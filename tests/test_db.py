# tests/test_db.py
import pytest
from storage.db import Database, ConversationRow, KnowledgeDocRow, RateLimitRow

@pytest.fixture
def db():
    d = Database("sqlite:///:memory:")
    d.create_tables()
    return d

def test_tables_created(db):
    with db.session() as s:
        rows = s.query(ConversationRow).all()
    assert rows == []

def test_knowledge_doc_table_exists(db):
    with db.session() as s:
        rows = s.query(KnowledgeDocRow).all()
    assert rows == []

def test_rate_limit_table_exists(db):
    with db.session() as s:
        rows = s.query(RateLimitRow).all()
    assert rows == []

def test_conversation_row_insert(db):
    with db.session() as s:
        row = ConversationRow(
            channel="telegram",
            chat_id="chat1",
            message_id="msg1",
            sender_id="user1",
            role="user",
            content="Hello",
        )
        s.add(row)
    # session auto-commits on exit
    with db.session() as s:
        fetched = s.query(ConversationRow).filter_by(message_id="msg1").first()
        assert fetched.content == "Hello"
        assert fetched.role == "user"
