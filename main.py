import asyncio
import logging
from core.config import load_settings
from storage.db import Database
from core.conversation import ConversationManager
from core.rate_limiter import RateLimiter
from core.user_detector import UserDetector
from knowledge.persona_engine import PersonaEngine
from knowledge.rag_engine import RAGEngine
from core.agent import ClaudeAgent
from core.gateway import Gateway
from adapters.telegram import TelegramAdapter
from adapters.whatsapp import WhatsAppAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("vedaai")


async def main():
    cfg = load_settings()

    # Storage
    db = Database(cfg["database"]["url"])
    db.create_tables()

    # Core components
    conv = ConversationManager(db)
    rate_limiter = RateLimiter(db, cooldown_seconds=cfg["rate_limiting"]["auto_reply_cooldown_seconds"])
    user_detector = UserDetector(cfg["owner"])
    persona = PersonaEngine("config/persona.yaml")
    rag = RAGEngine(
        documents_path=cfg["rag"]["documents_path"],
        chroma_path=cfg["rag"]["chroma_path"],
        top_k=cfg["rag"]["top_k"],
        min_similarity=cfg["rag"]["min_similarity"],
    )

    # Index documents on startup (async — uses thread executor internally)
    logger.info("Indexing knowledge documents...")
    await rag.index_documents()
    logger.info("Indexing complete.")

    agent = ClaudeAgent(
        api_key=cfg["anthropic"]["api_key"],
        model=cfg["anthropic"]["model"],
        conversation=conv,
        rag=rag,
        persona=persona,
    )

    # Adapters
    tg = TelegramAdapter(bot_token=cfg["telegram"]["bot_token"])
    wa = WhatsAppAdapter(
        bridge_url=cfg["whatsapp"]["bridge_url"],
        bridge_script=cfg["whatsapp"]["bridge_script"],
        polling_interval=cfg["whatsapp"]["polling_interval_seconds"],
    )

    gateway = Gateway(
        adapters={"telegram": tg, "whatsapp": wa},
        agent=agent,
        conversation_mgr=conv,
        user_detector=user_detector,
        rate_limiter=rate_limiter,
    )

    # Register message handlers
    await tg.on_message(gateway.handle)
    await wa.on_message(gateway.handle)

    # Connect adapters — WhatsApp raises on failure; main.py handles graceful degradation
    logger.info("Connecting adapters...")
    try:
        await wa.connect()
        logger.info("WhatsApp adapter connected.")
    except Exception as e:
        logger.warning("WhatsApp bridge failed to start (%s) — continuing with Telegram only.", e)

    await tg.connect()
    logger.info("VedaAI is running. Press Ctrl+C to stop.")

    try:
        await asyncio.Event().wait()   # run forever
    finally:
        logger.info("Shutting down...")
        await tg.disconnect()
        await wa.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
