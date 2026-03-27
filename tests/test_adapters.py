import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from adapters.base import RawMessage, Message, BaseAdapter

def test_raw_message_fields():
    raw = RawMessage(
        channel="telegram", chat_id="123", message_id="msg1",
        sender_id="456", sender_name="Alice",
        text="Hello", timestamp=datetime(2026, 1, 1), is_group=False,
    )
    assert raw.channel == "telegram"
    assert raw.is_group is False

def test_message_has_is_owner():
    msg = Message(
        channel="telegram", chat_id="123", message_id="msg1",
        sender_id="456", sender_name="Alice",
        text="Hello", timestamp=datetime(2026, 1, 1),
        is_group=False, is_owner=True,
    )
    assert msg.is_owner is True

def test_base_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseAdapter()


# ── Task 11: TelegramAdapter ──────────────────────────────────────────────────
from adapters.telegram import TelegramAdapter

@pytest.fixture
def tg():
    return TelegramAdapter(bot_token="fake_token", owner_id="999")

@pytest.mark.asyncio
async def test_telegram_channel_name(tg):
    assert tg.channel == "telegram"

@pytest.mark.asyncio
async def test_telegram_on_message_registers_callback(tg):
    callback = AsyncMock()
    await tg.on_message(callback)
    assert tg._callback == callback

@pytest.mark.asyncio
async def test_telegram_handles_text_message(tg):
    callback = AsyncMock()
    await tg.on_message(callback)
    mock_update = MagicMock()
    mock_update.message.text = "Hello"
    mock_update.message.message_id = 42
    mock_update.message.chat.id = "chat1"
    mock_update.message.from_user.id = 123
    mock_update.message.from_user.first_name = "Alice"
    mock_update.message.chat.type = "private"
    mock_update.message.date.timestamp.return_value = 1711500000.0
    await tg._handle_update(mock_update, None)
    callback.assert_called_once()
    raw = callback.call_args[0][0]
    assert raw.channel == "telegram"
    assert raw.text == "Hello"
    assert raw.sender_id == "123"
    assert raw.is_group is False

@pytest.mark.asyncio
async def test_telegram_group_message_sets_is_group(tg):
    callback = AsyncMock()
    await tg.on_message(callback)
    mock_update = MagicMock()
    mock_update.message.text = "Group msg"
    mock_update.message.message_id = 43
    mock_update.message.chat.id = "group1"
    mock_update.message.from_user.id = 456
    mock_update.message.from_user.first_name = "Bob"
    mock_update.message.chat.type = "group"
    mock_update.message.date.timestamp.return_value = 1711500000.0
    await tg._handle_update(mock_update, None)
    raw = callback.call_args[0][0]
    assert raw.is_group is True

@pytest.mark.asyncio
async def test_telegram_non_text_message_produces_placeholder(tg):
    callback = AsyncMock()
    await tg.on_message(callback)
    mock_update = MagicMock()
    mock_update.message.text = None  # e.g. a photo
    mock_update.message.message_id = 44
    mock_update.message.chat.id = "chat1"
    mock_update.message.from_user.id = 789
    mock_update.message.from_user.first_name = "Carol"
    mock_update.message.chat.type = "private"
    mock_update.message.date.timestamp.return_value = 1711500000.0
    await tg._handle_update(mock_update, None)
    callback.assert_called_once()
    raw = callback.call_args[0][0]
    assert raw.text == "[non-text message]"
