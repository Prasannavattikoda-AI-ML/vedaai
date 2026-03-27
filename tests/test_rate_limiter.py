import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from core.rate_limiter import RateLimiter
from storage.db import Database

@pytest.fixture
def limiter():
    db = Database("sqlite:///:memory:")
    db.create_tables()
    return RateLimiter(db, cooldown_seconds=10)

def test_not_throttled_on_first_message(limiter):
    assert limiter.is_throttled("telegram", "user1") is False

def test_throttled_immediately_after_record(limiter):
    limiter.record("telegram", "user1")
    assert limiter.is_throttled("telegram", "user1") is True

def test_not_throttled_after_cooldown(limiter):
    limiter.record("telegram", "user1")
    future = datetime.now(timezone.utc) + timedelta(seconds=11)
    with patch("core.rate_limiter.datetime") as mock_dt:
        mock_dt.now.return_value = future
        assert limiter.is_throttled("telegram", "user1") is False

def test_different_senders_independent(limiter):
    limiter.record("telegram", "user1")
    assert limiter.is_throttled("telegram", "user2") is False
