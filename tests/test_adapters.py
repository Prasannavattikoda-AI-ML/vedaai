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
    return TelegramAdapter(bot_token="fake_token")

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

@pytest.mark.asyncio
async def test_telegram_ignores_channel_post_without_from_user(tg):
    """Channel posts have no from_user — should be silently ignored."""
    callback = AsyncMock()
    await tg.on_message(callback)
    mock_update = MagicMock()
    mock_update.message.text = "Channel post"
    mock_update.message.from_user = None  # channel posts have no sender
    await tg._handle_update(mock_update, None)
    callback.assert_not_called()


# ── Task 13: WhatsAppAdapter ──────────────────────────────────────────────────
from adapters.whatsapp import WhatsAppAdapter

@pytest.fixture
def wa():
    return WhatsAppAdapter(
        bridge_url="http://localhost:3000",
        bridge_script="bridge/index.js",
        polling_interval=0.01,  # fast for tests
    )

@pytest.fixture
def wa_adapter():
    return WhatsAppAdapter(
        bridge_url="http://localhost:3000",
        bridge_script="bridge/index.js",
        polling_interval=0.01,
    )

@pytest.mark.asyncio
async def test_whatsapp_channel_name(wa):
    assert wa.channel == "whatsapp"

@pytest.mark.asyncio
async def test_whatsapp_on_message_registers_callback(wa):
    callback = AsyncMock()
    await wa.on_message(callback)
    assert wa._callback == callback

@pytest.mark.asyncio
async def test_whatsapp_polls_and_dispatches(wa):
    callback = AsyncMock()
    await wa.on_message(callback)
    bridge_payload = [{
        "message_id": "WA_001",
        "from": "+91111",
        "from_name": "Bob",
        "chat_id": "91111@c.us",
        "body": "Hey",
        "timestamp": 1711500000,
        "is_group": False,
    }]
    # Mock aiohttp ClientSession.get
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=bridge_payload)
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_cm)
    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)
    with patch("aiohttp.ClientSession", return_value=mock_session_cm):
        await wa._poll_once()
    callback.assert_called_once()
    raw = callback.call_args[0][0]
    assert raw.channel == "whatsapp"
    assert raw.text == "Hey"
    assert raw.message_id == "WA_001"
    assert raw.is_group is False

@pytest.mark.asyncio
async def test_whatsapp_poll_once_no_callback_does_nothing(wa):
    # _poll_once with no callback registered should not raise
    await wa._poll_once()  # no callback registered

@pytest.mark.asyncio
async def test_whatsapp_poll_error_does_not_crash(wa):
    """Network errors during polling should be swallowed gracefully."""
    callback = AsyncMock()
    await wa.on_message(callback)
    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)
    with patch("aiohttp.ClientSession", return_value=mock_session_cm):
        await wa._poll_once()  # must not raise
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_whatsapp_connect_raises_on_missing_node(wa_adapter):
    """connect() raises when node binary is not found (FileNotFoundError)."""
    wa_adapter._bridge_script = "bridge/index.js"
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("node not found")):
        with pytest.raises(FileNotFoundError):
            await wa_adapter.connect()
    assert wa_adapter._running is False


@pytest.mark.asyncio
async def test_whatsapp_disconnect_terminates_proc(wa_adapter):
    mock_proc = MagicMock()
    wa_adapter._proc = mock_proc
    wa_adapter._running = True
    await wa_adapter.disconnect()
    assert wa_adapter._running is False
    mock_proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_whatsapp_wait_for_ready_timeout(wa_adapter):
    """_wait_for_ready raises asyncio.TimeoutError if bridge never prints ready."""
    import asyncio as aio
    mock_proc = MagicMock()
    mock_proc.stdout = MagicMock()

    async def never_ready():
        # yield to event loop so asyncio.wait_for timeout can fire
        await aio.sleep(0)
        return b"some other output\n"

    mock_proc.stdout.readline = never_ready
    wa_adapter._proc = mock_proc
    with pytest.raises(aio.TimeoutError):
        await wa_adapter._wait_for_ready(timeout=0.05)
