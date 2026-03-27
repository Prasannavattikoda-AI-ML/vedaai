from datetime import datetime, timedelta, timezone
from storage.db import Database, RateLimitRow


class RateLimiter:
    def __init__(self, db: Database, cooldown_seconds: int = 10):
        self._db = db
        self._cooldown = timedelta(seconds=cooldown_seconds)

    def is_throttled(self, channel: str, sender_id: str) -> bool:
        with self._db.session() as s:
            row = s.query(RateLimitRow).filter_by(
                channel=channel, sender_id=sender_id
            ).first()
            if row is None:
                return False
            last_seen = row.last_seen
            # Use astimezone to handle both naive (SQLite) and aware (PostgreSQL) datetimes
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            else:
                last_seen = last_seen.astimezone(timezone.utc)
            return datetime.now(timezone.utc) - last_seen < self._cooldown

    def record(self, channel: str, sender_id: str) -> None:
        with self._db.session() as s:
            row = s.query(RateLimitRow).filter_by(
                channel=channel, sender_id=sender_id
            ).first()
            now = datetime.now(timezone.utc)
            if row:
                row.last_seen = now
            else:
                s.add(RateLimitRow(channel=channel, sender_id=sender_id, last_seen=now))
