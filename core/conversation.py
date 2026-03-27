from adapters.base import Message
from storage.db import Database, ConversationRow


class ConversationManager:
    def __init__(self, db: Database):
        self._db = db

    async def get_history(
        self, channel: str, chat_id: str, limit: int = 20
    ) -> list[dict]:
        """Returns last `limit` messages for a (channel, chat_id) pair. Always per-chat."""
        with self._db.session() as s:
            rows = (
                s.query(ConversationRow)
                .filter_by(channel=channel, chat_id=chat_id)
                .order_by(ConversationRow.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    async def save(self, message: Message, role: str) -> None:
        """Saves an incoming message to conversation history."""
        with self._db.session() as s:
            s.add(ConversationRow(
                channel=message.channel,
                chat_id=message.chat_id,
                message_id=message.message_id,
                sender_id=message.sender_id,
                role=role,
                content=message.text,
            ))

    async def save_response(self, message: Message, response: str) -> None:
        """Saves an outgoing assistant response. Role is always 'assistant' — hardcoded."""
        with self._db.session() as s:
            s.add(ConversationRow(
                channel=message.channel,
                chat_id=message.chat_id,
                message_id=None,
                sender_id="assistant",
                role="assistant",
                content=response,
            ))

    async def is_duplicate(self, channel: str, message_id: str) -> bool:
        """Returns True if message_id was already processed for this channel."""
        if message_id is None:
            return False
        with self._db.session() as s:
            return s.query(ConversationRow).filter_by(
                channel=channel, message_id=message_id
            ).first() is not None
