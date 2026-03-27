import pytest
from datetime import datetime
from adapters.base import RawMessage
from core.user_detector import UserDetector

OWNER_IDS = {"whatsapp": "+91111", "telegram": "999"}

def make_raw(channel, sender_id, is_group=False):
    return RawMessage(
        channel=channel, chat_id="chat1", message_id="m1",
        sender_id=sender_id, sender_name="Test",
        text="hi", timestamp=datetime(2026,1,1), is_group=is_group,
    )

def test_owner_detected():
    ud = UserDetector(OWNER_IDS)
    msg = ud.resolve(make_raw("telegram", "999"))
    assert msg.is_owner is True

def test_non_owner_detected():
    ud = UserDetector(OWNER_IDS)
    msg = ud.resolve(make_raw("telegram", "stranger"))
    assert msg.is_owner is False

def test_missing_owner_raises():
    with pytest.raises(ValueError, match="Missing owner.telegram"):
        UserDetector({"whatsapp": "+91111"})

def test_empty_owner_raises():
    with pytest.raises(ValueError, match="Missing owner.whatsapp"):
        UserDetector({"whatsapp": "", "telegram": "999"})

def test_resolve_copies_all_fields():
    ud = UserDetector(OWNER_IDS)
    raw = make_raw("telegram", "999", is_group=True)
    msg = ud.resolve(raw)
    assert msg.channel == "telegram"
    assert msg.is_group is True
    assert msg.is_owner is True
