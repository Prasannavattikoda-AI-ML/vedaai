import asyncio
import logging
from datetime import datetime, timezone
import aiohttp
from adapters.base import BaseAdapter, RawMessage, MessageCallback

logger = logging.getLogger(__name__)


class WhatsAppAdapter(BaseAdapter):
    @property
    def channel(self) -> str:
        return "whatsapp"

    def __init__(
        self,
        bridge_url: str,
        bridge_script: str,
        polling_interval: float = 2.0,
    ):
        self._bridge_url = bridge_url.rstrip("/")
        self._bridge_script = bridge_script
        self._polling_interval = polling_interval
        self._callback: MessageCallback | None = None
        self._proc = None
        self._poll_task = None
        self._running = False

    async def on_message(self, callback: MessageCallback) -> None:
        self._callback = callback

    async def connect(self) -> None:
        """Start the Node.js bridge subprocess."""
        self._proc = await asyncio.create_subprocess_exec(
            "node", self._bridge_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await self._wait_for_ready(timeout=30)
        except Exception:
            self._proc.terminate()
            raise
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("WhatsApp adapter connected")

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            await asyncio.gather(self._poll_task, return_exceptions=True)
        if self._proc:
            self._proc.terminate()

    async def send_message(self, chat_id: str, text: str) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._bridge_url}/send",
                json={"chat_id": chat_id, "message": text},
            ) as resp:
                data = await resp.json()
                if data.get("status") != "sent":
                    logger.error("WhatsApp send failed: %s", data.get("reason"))

    async def _wait_for_ready(self, timeout: int) -> None:
        if not self._proc or not self._proc.stdout:
            raise RuntimeError("Bridge process not started")

        async def _read_until_ready():
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("Bridge process exited unexpectedly")
                if b"WhatsApp ready" in line:
                    return

        await asyncio.wait_for(_read_until_ready(), timeout=timeout)

    async def _poll_loop(self) -> None:
        while self._running:
            await self._poll_once()
            await asyncio.sleep(self._polling_interval)

    async def _poll_once(self) -> None:
        if not self._callback:
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._bridge_url}/messages/pending"
                ) as resp:
                    messages = await resp.json()
            for m in messages:
                try:
                    raw = RawMessage(
                        channel=self.channel,
                        chat_id=m["chat_id"],
                        message_id=m["message_id"],
                        sender_id=m["from"],
                        sender_name=m.get("from_name", m["from"]),
                        text=m["body"],
                        timestamp=datetime.fromtimestamp(
                            m["timestamp"], tz=timezone.utc
                        ),
                        is_group=m.get("is_group", False),
                    )
                    await self._callback(raw)
                except Exception as e:
                    logger.warning("WhatsApp: failed to dispatch message %s: %s", m.get("message_id"), e)
        except Exception as e:
            logger.warning("WhatsApp polling error: %s", e)
