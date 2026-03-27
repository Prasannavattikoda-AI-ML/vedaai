from contextlib import contextmanager
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class ConversationRow(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    channel = Column(String, nullable=False)
    chat_id = Column(String, nullable=False)
    message_id = Column(String, nullable=True)   # nullable for assistant responses
    sender_id = Column(String, nullable=False)
    role = Column(String, nullable=False)         # "user" | "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("channel", "message_id", name="uq_channel_message_id"),
    )


class KnowledgeDocRow(Base):
    __tablename__ = "knowledge_docs"
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_hash = Column(String, nullable=False)
    indexed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RateLimitRow(Base):
    __tablename__ = "rate_limits"
    channel = Column(String, primary_key=True)
    sender_id = Column(String, primary_key=True)
    last_seen = Column(DateTime, nullable=False)


class Database:
    def __init__(self, url: str):
        self._engine = create_engine(url)
        self._Session = sessionmaker(self._engine)

    def create_tables(self) -> None:
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self):
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()
