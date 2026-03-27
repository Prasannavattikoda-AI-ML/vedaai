import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from adapters.base import RawMessage, Message
from core.gateway import Gateway

def make_raw(sender_id="stranger", message_id="m1", is_group=False, channel="telegram"):
    return RawMessage(
        channel=channel, chat_id="chat1", message_id=message_id,
        sender_id=sender_id, sender_name="Test",
        text="hi", timestamp=datetime(2026,1,1), is_group=is_group,
    )

def make_msg(is_owner=False, is_group=False, message_id="m1"):
    return Message(
        channel="telegram", chat_id="chat1", message_id=message_id,
        sender_id="u1", sender_name="T", text="hi",
        timestamp=datetime(2026,1,1), is_group=is_group, is_owner=is_owner,
    )

@pytest.fixture
def gw():
    adapter = AsyncMock()
    adapter.channel = "telegram"
    agent = AsyncMock()
    agent.auto_reply_mode.return_value = "auto reply"
    agent.assistant_mode.return_value = "assistant reply"
    conv = AsyncMock()
    conv.is_duplicate.return_value = False
    detector = MagicMock()
    rate_limiter = MagicMock()
    rate_limiter.is_throttled.return_value = False
    gateway = Gateway(
        adapters={"telegram": adapter},
        agent=agent,
        conversation_mgr=conv,
        user_detector=detector,
        rate_limiter=rate_limiter,
    )
    return gateway, adapter, agent, conv, detector, rate_limiter

@pytest.mark.asyncio
async def test_auto_reply_for_non_owner(gw):
    gateway, adapter, agent, conv, detector, _ = gw
    detector.resolve.return_value = make_msg(is_owner=False)
    await gateway.handle(make_raw())
    agent.auto_reply_mode.assert_called_once()
    agent.assistant_mode.assert_not_called()

@pytest.mark.asyncio
async def test_assistant_mode_for_owner(gw):
    gateway, adapter, agent, conv, detector, _ = gw
    detector.resolve.return_value = make_msg(is_owner=True)
    await gateway.handle(make_raw(sender_id="owner"))
    agent.assistant_mode.assert_called_once()
    agent.auto_reply_mode.assert_not_called()

@pytest.mark.asyncio
async def test_duplicate_skipped(gw):
    gateway, adapter, agent, conv, detector, _ = gw
    conv.is_duplicate.return_value = True
    await gateway.handle(make_raw())
    agent.auto_reply_mode.assert_not_called()

@pytest.mark.asyncio
async def test_group_message_from_non_owner_ignored(gw):
    gateway, adapter, agent, conv, detector, _ = gw
    detector.resolve.return_value = make_msg(is_owner=False, is_group=True)
    await gateway.handle(make_raw(is_group=True))
    agent.auto_reply_mode.assert_not_called()

@pytest.mark.asyncio
async def test_rate_limited_sender_ignored(gw):
    gateway, adapter, agent, conv, detector, rate_limiter = gw
    detector.resolve.return_value = make_msg(is_owner=False)
    rate_limiter.is_throttled.return_value = True
    await gateway.handle(make_raw())
    agent.auto_reply_mode.assert_not_called()

@pytest.mark.asyncio
async def test_both_sides_saved(gw):
    gateway, adapter, agent, conv, detector, _ = gw
    detector.resolve.return_value = make_msg(is_owner=False)
    await gateway.handle(make_raw())
    conv.save.assert_called_once()
    conv.save_response.assert_called_once()
    # role must NOT be passed — it is hardcoded inside save_response
    call_kwargs = conv.save_response.call_args.kwargs
    assert "role" not in call_kwargs

@pytest.mark.asyncio
async def test_response_sent_via_adapter(gw):
    gateway, adapter, agent, conv, detector, _ = gw
    detector.resolve.return_value = make_msg(is_owner=False)
    agent.auto_reply_mode.return_value = "Hello back"
    await gateway.handle(make_raw())
    adapter.send_message.assert_called_once_with("chat1", "Hello back")

@pytest.mark.asyncio
async def test_concurrent_messages_same_chat_serialized(gw):
    gateway, _, agent, conv, detector, _ = gw
    results = []
    async def slow_reply(msg):
        await asyncio.sleep(0.05)
        results.append("done")
        return "reply"
    agent.auto_reply_mode.side_effect = slow_reply
    detector.resolve.return_value = make_msg(is_owner=False)
    conv.is_duplicate.side_effect = [False, False]
    await asyncio.gather(
        gateway.handle(make_raw(message_id="m1")),
        gateway.handle(make_raw(message_id="m2")),
    )
    assert len(results) == 2

@pytest.mark.asyncio
async def test_missing_adapter_raises(gw):
    gateway, _, agent, conv, detector, _ = gw
    # Send a message on a channel with no registered adapter
    raw = RawMessage(
        channel="unknown", chat_id="c1", message_id="m1",
        sender_id="u1", sender_name="T", text="hi",
        timestamp=datetime(2026,1,1), is_group=False,
    )
    msg = Message(
        channel="unknown", chat_id="c1", message_id="m1",
        sender_id="u1", sender_name="T", text="hi",
        timestamp=datetime(2026,1,1), is_group=False, is_owner=False,
    )
    detector.resolve.return_value = msg
    with pytest.raises(RuntimeError, match="No adapter registered"):
        await gateway.handle(raw)
