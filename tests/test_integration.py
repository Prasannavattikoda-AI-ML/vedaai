# tests/test_integration.py
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from adapters.base import RawMessage
from core.conversation import ConversationManager
from core.rate_limiter import RateLimiter
from core.user_detector import UserDetector
from core.gateway import Gateway
from core.agent import ClaudeAgent
from knowledge.persona_engine import PersonaEngine
from knowledge.rag_engine import RAGEngine
from storage.db import Database

OWNER_IDS = {"whatsapp": "+91999", "telegram": "owner123"}


@pytest.fixture
async def full_system(tmp_path):
    db = Database("sqlite:///:memory:")
    db.create_tables()
    conv = ConversationManager(db)
    rate_limiter = RateLimiter(db, cooldown_seconds=0)   # no cooldown in tests
    user_detector = UserDetector(OWNER_IDS)

    persona_yaml = tmp_path / "persona.yaml"
    persona_yaml.write_text(
        'name: "Pandu"\ntone: "friendly"\nlanguage: "English"\nboundaries: []\nrules: []'
    )
    persona = PersonaEngine(str(persona_yaml))

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "bio.txt").write_text("Pandu is an AI engineer.")
    rag = RAGEngine(str(docs), str(tmp_path / ".chroma"), top_k=1, min_similarity=0.0)
    await rag.index_documents()

    mock_adapter = AsyncMock()
    mock_adapter.channel = "telegram"

    # AsyncAnthropic client — mock must be AsyncMock
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello stranger!", type="text")]
    mock_response.stop_reason = "end_turn"

    agent = ClaudeAgent(
        api_key="test", model="claude-sonnet-4-6",
        conversation=conv, rag=rag, persona=persona,
    )
    agent._client = MagicMock()
    agent._client.messages = MagicMock()
    agent._client.messages.create = AsyncMock(return_value=mock_response)

    gateway = Gateway(
        adapters={"telegram": mock_adapter},
        agent=agent,
        conversation_mgr=conv,
        user_detector=user_detector,
        rate_limiter=rate_limiter,
    )
    return gateway, mock_adapter, conv, agent


@pytest.mark.asyncio
async def test_full_auto_reply_flow(full_system):
    gateway, adapter, conv, _ = full_system
    raw = RawMessage(
        channel="telegram", chat_id="chat1", message_id="int_m1",
        sender_id="stranger", sender_name="Bob",
        text="Hi Pandu!", timestamp=datetime(2026, 1, 1),
        is_group=False,
    )
    await gateway.handle(raw)
    adapter.send_message.assert_called_once_with("chat1", "Hello stranger!")
    history = await conv.get_history("telegram", "chat1")
    assert len(history) == 2   # user + assistant


@pytest.mark.asyncio
async def test_full_owner_assistant_flow(full_system):
    gateway, adapter, conv, agent = full_system
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Here is what I know!", type="text")]
    mock_response.stop_reason = "end_turn"
    agent._client.messages.create = AsyncMock(return_value=mock_response)

    raw = RawMessage(
        channel="telegram", chat_id="owner_chat", message_id="owner_m1",
        sender_id="owner123", sender_name="Pandu",
        text="What do you know about me?", timestamp=datetime(2026, 1, 1),
        is_group=False,
    )
    await gateway.handle(raw)
    adapter.send_message.assert_called_once_with("owner_chat", "Here is what I know!")
    # Verify tools were passed (assistant mode uses tools; auto-reply does not)
    call_kwargs = agent._client.messages.create.call_args.kwargs
    assert "tools" in call_kwargs


@pytest.mark.asyncio
async def test_duplicate_message_not_processed_twice(full_system):
    gateway, adapter, conv, _ = full_system
    raw = RawMessage(
        channel="telegram", chat_id="chat1", message_id="dup_m1",
        sender_id="stranger", sender_name="Bob",
        text="Hi", timestamp=datetime(2026, 1, 1), is_group=False,
    )
    await gateway.handle(raw)
    await gateway.handle(raw)  # same message_id
    assert adapter.send_message.call_count == 1
