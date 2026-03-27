from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Awaitable


@dataclass
class RawMessage:
    channel: str
    chat_id: str
    message_id: str
    sender_id: str
    sender_name: str
    text: str
    timestamp: datetime
    is_group: bool


@dataclass
class Message:
    channel: str
    chat_id: str
    message_id: str
    sender_id: str
    sender_name: str
    text: str
    timestamp: datetime
    is_group: bool
    is_owner: bool


MessageCallback = Callable[[RawMessage], Awaitable[None]]


class BaseAdapter(ABC):
    @property
    @abstractmethod
    def channel(self) -> str:
        """Channel name: 'whatsapp' or 'telegram'. Must be defined by concrete adapter."""
        ...

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None: ...

    @abstractmethod
    async def on_message(self, callback: MessageCallback) -> None:
        """Called ONCE at startup. Adapter invokes callback for each incoming message."""
        ...
