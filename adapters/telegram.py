import logging
from datetime import datetime
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from adapters.base import BaseAdapter, RawMessage, MessageCallback

logger = logging.getLogger(__name__)

GROUP_TYPES = {"group", "supergroup", "channel"}


class TelegramAdapter(BaseAdapter):
    @property
    def channel(self) -> str:
        return "telegram"

    def __init__(self, bot_token: str, owner_id: str):
        self._token = bot_token
        self._owner_id = owner_id
        self._callback: MessageCallback | None = None
        self._app: Application | None = None

    async def on_message(self, callback: MessageCallback) -> None:
        self._callback = callback

    async def connect(self) -> None:
        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(MessageHandler(filters.ALL, self._handle_update))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("Telegram adapter connected")

    async def disconnect(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send_message(self, chat_id: str, text: str) -> None:
        await self._app.bot.send_message(chat_id=chat_id, text=text)

    async def _handle_update(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message or not self._callback:
            return

        is_group = update.message.chat.type in GROUP_TYPES
        timestamp = datetime.utcfromtimestamp(update.message.date.timestamp())
        text = update.message.text if update.message.text else "[non-text message]"

        raw = RawMessage(
            channel=self.channel,
            chat_id=str(update.message.chat.id),
            message_id=str(update.message.message_id),
            sender_id=str(update.message.from_user.id),
            sender_name=update.message.from_user.first_name or "",
            text=text,
            timestamp=timestamp,
            is_group=is_group,
        )
        await self._callback(raw)
