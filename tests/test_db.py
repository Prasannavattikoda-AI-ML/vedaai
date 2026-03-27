# tests/test_db.py
import pytest
from storage.db import Database, ConversationRow, KnowledgeDocRow, RateLimitRow

@pytest.fixture
def db():
    d = Database("sqlite:///:memory:")
    d.create_tables()
    return d

def test_tables_created(db):
    session = db.session()
    rows = session.query(ConversationRow).all()
    assert rows == []
    session.close()

def test_knowledge_doc_table_exists(db):
    session = db.session()
    rows = session.query(KnowledgeDocRow).all()
    assert rows == []
    session.close()

def test_rate_limit_table_exists(db):
    session = db.session()
    rows = session.query(RateLimitRow).all()
    assert rows == []
    session.close()

def test_conversation_row_insert(db):
    from datetime import datetime
    session = db.session()
    row = ConversationRow(
        channel="telegram",
        chat_id="chat1",
        message_id="msg1",
        sender_id="user1",
        role="user",
        content="Hello",
    )
    session.add(row)
    session.commit()
    fetched = session.query(ConversationRow).filter_by(message_id="msg1").first()
    assert fetched.content == "Hello"
    assert fetched.role == "user"
    session.close()
