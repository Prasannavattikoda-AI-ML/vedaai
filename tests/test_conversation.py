import pytest
from datetime import datetime
from adapters.base import Message
from core.conversation import ConversationManager
from storage.db import Database

@pytest.fixture
def conv():
    db = Database("sqlite:///:memory:")
    db.create_tables()
    return ConversationManager(db)

def make_msg(channel="telegram", chat_id="chat1", message_id="m1", sender_id="u1", is_owner=False):
    return Message(
        channel=channel, chat_id=chat_id, message_id=message_id,
        sender_id=sender_id, sender_name="Test",
        text="hello", timestamp=datetime(2026,1,1),
        is_group=False, is_owner=is_owner,
    )

@pytest.mark.asyncio
async def test_save_and_get_history(conv):
    msg = make_msg()
    await conv.save(msg, role="user")
    history = await conv.get_history("telegram", "chat1")
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"

@pytest.mark.asyncio
async def test_save_response(conv):
    msg = make_msg()
    await conv.save_response(msg, "world")
    history = await conv.get_history("telegram", "chat1")
    assert history[0]["role"] == "assistant"
    assert history[0]["content"] == "world"

@pytest.mark.asyncio
async def test_history_is_per_chat(conv):
    await conv.save(make_msg(chat_id="chat1", message_id="m1"), role="user")
    await conv.save(make_msg(chat_id="chat2", message_id="m2"), role="user")
    h1 = await conv.get_history("telegram", "chat1")
    h2 = await conv.get_history("telegram", "chat2")
    assert len(h1) == 1 and len(h2) == 1

@pytest.mark.asyncio
async def test_history_limit(conv):
    for i in range(25):
        await conv.save(make_msg(message_id=f"m{i}"), role="user")
    history = await conv.get_history("telegram", "chat1", limit=20)
    assert len(history) == 20

@pytest.mark.asyncio
async def test_is_duplicate_false_for_new(conv):
    assert await conv.is_duplicate("telegram", "m1") is False

@pytest.mark.asyncio
async def test_is_duplicate_true_after_save(conv):
    msg = make_msg(message_id="m1")
    await conv.save(msg, role="user")
    assert await conv.is_duplicate("telegram", "m1") is True

@pytest.mark.asyncio
async def test_is_duplicate_none_message_id_returns_false(conv):
    # None message_id (assistant responses) must never be treated as duplicates
    assert await conv.is_duplicate("telegram", None) is False
