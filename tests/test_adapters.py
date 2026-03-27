from datetime import datetime
from adapters.base import RawMessage, Message

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
