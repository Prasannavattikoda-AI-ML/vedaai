import asyncio
import logging
from collections import defaultdict
from adapters.base import RawMessage, BaseAdapter
from core.agent import ClaudeAgent
from core.conversation import ConversationManager
from core.user_detector import UserDetector
from core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class Gateway:
    def __init__(
        self,
        adapters: dict,
        agent: ClaudeAgent,
        conversation_mgr: ConversationManager,
        user_detector: UserDetector,
        rate_limiter: RateLimiter,
    ):
        self._adapters = adapters
        self._agent = agent
        self._conv = conversation_mgr
        self._detector = user_detector
        self._rate_limiter = rate_limiter
        # Lambda factory for Python 3.10+ compatibility
        self._chat_locks: dict = defaultdict(lambda: asyncio.Lock())

    async def handle(self, raw: RawMessage) -> None:
        if await self._conv.is_duplicate(raw.channel, raw.message_id):
            logger.debug("Duplicate message %s — skipping", raw.message_id)
            return

        message = self._detector.resolve(raw)

        if message.is_group and not message.is_owner:
            logger.debug("Group message from non-owner — ignoring")
            return

        if not message.is_owner and self._rate_limiter.is_throttled(
            message.channel, message.sender_id
        ):
            logger.debug("Rate-limited sender %s — dropping", message.sender_id)
            return

        if not message.is_owner:
            self._rate_limiter.record(message.channel, message.sender_id)

        lock_key = f"{message.channel}:{message.chat_id}"
        async with self._chat_locks[lock_key]:
            if message.is_owner:
                response = await self._agent.assistant_mode(message)
            else:
                response = await self._agent.auto_reply_mode(message)

            await self._conv.save(message, role="user")
            await self._conv.save_response(message, response)

            adapter = self._adapters.get(message.channel)
            if adapter is None:
                raise RuntimeError(
                    f"No adapter registered for channel: {message.channel}"
                )
            await adapter.send_message(message.chat_id, response)
