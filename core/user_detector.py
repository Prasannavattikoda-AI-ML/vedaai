from dataclasses import asdict
from adapters.base import RawMessage, Message


class UserDetector:
    def __init__(self, owner_ids: dict[str, str]):
        for channel in ["whatsapp", "telegram"]:
            if channel not in owner_ids or not owner_ids[channel]:
                raise ValueError(
                    f"Missing owner.{channel} in settings.yaml. "
                    "VedaAI cannot start without owner IDs configured."
                )
        self.owner_ids = owner_ids

    def resolve(self, raw: RawMessage) -> Message:
        """Enriches RawMessage with is_owner flag, returns full Message."""
        is_owner = self.owner_ids.get(raw.channel) == raw.sender_id
        return Message(**asdict(raw), is_owner=is_owner)
