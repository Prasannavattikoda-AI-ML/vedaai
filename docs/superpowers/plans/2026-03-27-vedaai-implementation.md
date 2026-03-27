# VedaAI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build VedaAI — a Python personal AI assistant that auto-replies to WhatsApp and Telegram messages using Claude + RAG + persona rules, and acts as a personal assistant when the owner messages it.

**Architecture:** Single Python asyncio process. A Gateway routes messages from WhatsApp/Telegram adapters through UserDetector → Claude Agent (with RAG and persona). All state stored in SQLite via SQLAlchemy. WhatsApp uses a local Node.js bridge; Telegram uses python-telegram-bot v20.

**Tech Stack:** Python 3.11+, Anthropic SDK (`anthropic>=0.50.0`), python-telegram-bot v20, ChromaDB, sentence-transformers, SQLAlchemy 2, aiohttp, pytest-asyncio

---

## File Map

| File | Responsibility |
|------|---------------|
| `main.py` | Entry point — wires all components, starts adapters |
| `config/settings.yaml` | All configuration (API keys via env vars) |
| `config/persona.yaml` | Persona name, tone, boundaries, trigger rules |
| `adapters/base.py` | `RawMessage`, `Message` dataclasses + `BaseAdapter` ABC |
| `adapters/telegram.py` | Telegram adapter using python-telegram-bot |
| `adapters/whatsapp.py` | WhatsApp adapter — manages Node.js bridge subprocess + polling |
| `bridge/index.js` | Node.js whatsapp-web.js bridge HTTP server |
| `bridge/package.json` | Node.js dependencies |
| `core/user_detector.py` | `UserDetector` — owner ID resolution, validates config at startup |
| `core/conversation.py` | `ConversationManager` — history read/write/dedup |
| `core/gateway.py` | `Gateway` — message routing, rate limiting, per-chat locking |
| `core/agent.py` | `ClaudeAgent` — auto-reply and assistant modes, tool loop |
| `core/rate_limiter.py` | `RateLimiter` — per-sender cooldown using SQLite |
| `knowledge/persona_engine.py` | Loads persona.yaml, builds system prompt fragment |
| `knowledge/rag_engine.py` | Document indexing (ChromaDB + sentence-transformers) + retrieval |
| `storage/db.py` | SQLAlchemy engine, session factory, ORM models |
| `tests/test_user_detector.py` | UserDetector unit tests |
| `tests/test_conversation.py` | ConversationManager unit tests |
| `tests/test_gateway.py` | Gateway routing, rate limiting, dedup, concurrency tests |
| `tests/test_agent.py` | ClaudeAgent unit tests (mocked Anthropic client) |
| `tests/test_rag.py` | RAG engine unit tests |
| `tests/test_persona.py` | PersonaEngine unit tests |
| `tests/test_adapters.py` | Adapter unit tests (mocked external libraries) |
| `tests/test_integration.py` | Full message flow integration test |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config/settings.yaml`
- Create: `config/persona.yaml`
- Create: `pytest.ini`

- [ ] **Step 1: Create requirements.txt**

```
anthropic>=0.50.0
python-telegram-bot>=20.0,<21
chromadb>=0.5.0,<1.0.0
sentence-transformers>=2.2.0
sqlalchemy>=2.0
alembic>=1.13.0
pyyaml>=6.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
pytest>=8.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: Create .env.example**

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

- [ ] **Step 3: Create config/settings.yaml**

```yaml
owner:
  whatsapp: "+91XXXXXXXXXX"
  telegram: "YOUR_TELEGRAM_USER_ID"

anthropic:
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-sonnet-4-6"

telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"

whatsapp:
  mode: "bridge"
  bridge_url: "http://localhost:3000"
  bridge_script: "bridge/index.js"
  polling_interval_seconds: 2

database:
  url: "sqlite:///vedaai.db"

rag:
  documents_path: "knowledge/documents"
  chroma_path: ".chroma"
  top_k: 3
  min_similarity: 0.4

conversation:
  history_window: 20

rate_limiting:
  auto_reply_cooldown_seconds: 10
```

- [ ] **Step 4: Create config/persona.yaml**

```yaml
name: "Pandu"
tone: "friendly, professional, concise"
language: "English"
boundaries:
  - "Don't share personal phone number"
  - "Don't commit to meetings without checking"
rules:
  - trigger: "availability"
    response_hint: "Check calendar context before responding"
  - trigger: "pricing|cost|rate"
    response_hint: "Redirect to email for business discussions"
```

- [ ] **Step 5: Create pytest.ini**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 6: Install dependencies**

```bash
cd /Users/anilkumar/Prasanna/vedaai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 7: Create empty package __init__ files**

```bash
mkdir -p adapters core knowledge storage tests knowledge/documents
touch adapters/__init__.py core/__init__.py knowledge/__init__.py storage/__init__.py tests/__init__.py
```

- [ ] **Step 8: Commit**

```bash
git add .
git commit -m "feat: project scaffold — requirements, config, directory structure"
```

---

## Task 2: Storage Layer

**Files:**
- Create: `storage/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_db.py
import pytest
from storage.db import Database, ConversationRow

@pytest.fixture
def db():
    d = Database("sqlite:///:memory:")
    d.create_tables()
    return d

def test_tables_created(db):
    # If tables were not created, this would raise
    session = db.session()
    rows = session.query(ConversationRow).all()
    assert rows == []
    session.close()
```

Run: `pytest tests/test_db.py -v`
Expected: FAIL — `storage.db` not found

- [ ] **Step 2: Create storage/db.py**

```python
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime,
    UniqueConstraint, Index
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class ConversationRow(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    channel = Column(String, nullable=False)
    chat_id = Column(String, nullable=False)
    message_id = Column(String, nullable=True)   # nullable for assistant responses
    sender_id = Column(String, nullable=False)
    role = Column(String, nullable=False)         # "user" | "assistant"
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Deduplicate by (channel, message_id) — partial unique index handled via UniqueConstraint
        # SQLite supports partial index via raw DDL; for simplicity we enforce in app logic
        UniqueConstraint("channel", "message_id", name="uq_channel_message_id"),
    )


class KnowledgeDocRow(Base):
    __tablename__ = "knowledge_docs"
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(String, nullable=False)
    chunk_hash = Column(String, nullable=False)
    indexed_at = Column(DateTime, default=datetime.utcnow)


class RateLimitRow(Base):
    __tablename__ = "rate_limits"
    channel = Column(String, primary_key=True)
    sender_id = Column(String, primary_key=True)
    last_seen = Column(DateTime, nullable=False)


class Database:
    def __init__(self, url: str):
        self._engine = create_engine(url)
        self._Session = sessionmaker(bind=self._engine)

    def create_tables(self) -> None:
        Base.metadata.create_all(self._engine)

    def session(self):
        return self._Session()
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add storage/db.py tests/test_db.py
git commit -m "feat: storage layer — SQLAlchemy ORM with conversations, knowledge_docs, rate_limits tables"
```

---

## Task 3: Data Models (RawMessage, Message, BaseAdapter)

**Files:**
- Create: `adapters/base.py`
- Test: `tests/test_adapters.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_adapters.py
from datetime import datetime
from adapters.base import RawMessage, Message

def test_raw_message_fields():
    raw = RawMessage(
        channel="telegram",
        chat_id="123",
        message_id="msg1",
        sender_id="456",
        sender_name="Alice",
        text="Hello",
        timestamp=datetime(2026, 1, 1),
        is_group=False,
    )
    assert raw.channel == "telegram"
    assert raw.is_group is False

def test_message_has_is_owner():
    msg = Message(
        channel="telegram",
        chat_id="123",
        message_id="msg1",
        sender_id="456",
        sender_name="Alice",
        text="Hello",
        timestamp=datetime(2026, 1, 1),
        is_group=False,
        is_owner=True,
    )
    assert msg.is_owner is True
```

Run: `pytest tests/test_adapters.py -v`
Expected: FAIL

- [ ] **Step 2: Create adapters/base.py**

```python
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
    channel: str

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None: ...

    @abstractmethod
    async def on_message(self, callback: MessageCallback) -> None:
        """
        Called ONCE at startup to register the message handler.
        The adapter stores the callback and invokes it for each
        incoming message for the lifetime of the process.
        """
        ...
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_adapters.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add adapters/base.py tests/test_adapters.py
git commit -m "feat: data models — RawMessage, Message dataclasses and BaseAdapter ABC"
```

---

## Task 4: UserDetector

**Files:**
- Create: `core/user_detector.py`
- Test: `tests/test_user_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_user_detector.py
import pytest
from datetime import datetime
from adapters.base import RawMessage
from core.user_detector import UserDetector

OWNER_IDS = {"whatsapp": "+91111", "telegram": "999"}

def make_raw(channel, sender_id, is_group=False):
    return RawMessage(
        channel=channel, chat_id="chat1", message_id="m1",
        sender_id=sender_id, sender_name="Test",
        text="hi", timestamp=datetime(2026,1,1), is_group=is_group
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
```

Run: `pytest tests/test_user_detector.py -v`
Expected: FAIL

- [ ] **Step 2: Create core/user_detector.py**

```python
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
        is_owner = self.owner_ids.get(raw.channel) == raw.sender_id
        return Message(**asdict(raw), is_owner=is_owner)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_user_detector.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add core/user_detector.py tests/test_user_detector.py
git commit -m "feat: UserDetector — owner resolution with startup validation"
```

---

## Task 5: ConversationManager

**Files:**
- Create: `core/conversation.py`
- Test: `tests/test_conversation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_conversation.py
import pytest
from datetime import datetime
from adapters.base import Message
from core.conversation import ConversationManager
from storage.db import Database

@pytest.fixture
def conv():
    db = Database("sqlite:///:memory:")
    db.create_tables()
    return ConversationManager(db)

def make_msg(channel="telegram", chat_id="chat1", message_id="m1", sender_id="u1", is_owner=False):
    return Message(
        channel=channel, chat_id=chat_id, message_id=message_id,
        sender_id=sender_id, sender_name="Test",
        text="hello", timestamp=datetime(2026,1,1),
        is_group=False, is_owner=is_owner
    )

@pytest.mark.asyncio
async def test_save_and_get_history(conv):
    msg = make_msg()
    await conv.save(msg, role="user")
    history = await conv.get_history("telegram", "chat1")
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"

@pytest.mark.asyncio
async def test_save_response(conv):
    msg = make_msg()
    await conv.save_response(msg, "world")
    history = await conv.get_history("telegram", "chat1")
    assert history[0]["role"] == "assistant"
    assert history[0]["content"] == "world"

@pytest.mark.asyncio
async def test_history_is_per_chat(conv):
    await conv.save(make_msg(chat_id="chat1", message_id="m1"), role="user")
    await conv.save(make_msg(chat_id="chat2", message_id="m2"), role="user")
    h1 = await conv.get_history("telegram", "chat1")
    h2 = await conv.get_history("telegram", "chat2")
    assert len(h1) == 1
    assert len(h2) == 1

@pytest.mark.asyncio
async def test_history_limit(conv):
    for i in range(25):
        await conv.save(make_msg(message_id=f"m{i}"), role="user")
    history = await conv.get_history("telegram", "chat1", limit=20)
    assert len(history) == 20

@pytest.mark.asyncio
async def test_is_duplicate_false_for_new(conv):
    assert await conv.is_duplicate("telegram", "m1") is False

@pytest.mark.asyncio
async def test_is_duplicate_true_after_save(conv):
    msg = make_msg(message_id="m1")
    await conv.save(msg, role="user")
    assert await conv.is_duplicate("telegram", "m1") is True
```

Run: `pytest tests/test_conversation.py -v`
Expected: FAIL

- [ ] **Step 2: Create core/conversation.py**

```python
from adapters.base import Message
from storage.db import Database, ConversationRow


class ConversationManager:
    def __init__(self, db: Database):
        self._db = db

    async def get_history(
        self, channel: str, chat_id: str, limit: int = 20
    ) -> list[dict]:
        session = self._db.session()
        try:
            rows = (
                session.query(ConversationRow)
                .filter_by(channel=channel, chat_id=chat_id)
                .order_by(ConversationRow.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [{"role": r.role, "content": r.content} for r in reversed(rows)]
        finally:
            session.close()

    async def save(self, message: Message, role: str) -> None:
        session = self._db.session()
        try:
            row = ConversationRow(
                channel=message.channel,
                chat_id=message.chat_id,
                message_id=message.message_id,
                sender_id=message.sender_id,
                role=role,
                content=message.text,
            )
            session.add(row)
            session.commit()
        finally:
            session.close()

    async def save_response(self, message: Message, response: str) -> None:
        """Role is always 'assistant' — hardcoded to prevent misuse."""
        session = self._db.session()
        try:
            row = ConversationRow(
                channel=message.channel,
                chat_id=message.chat_id,
                message_id=None,          # assistant responses have no platform message_id
                sender_id="assistant",
                role="assistant",
                content=response,
            )
            session.add(row)
            session.commit()
        finally:
            session.close()

    async def is_duplicate(self, channel: str, message_id: str) -> bool:
        session = self._db.session()
        try:
            exists = (
                session.query(ConversationRow)
                .filter_by(channel=channel, message_id=message_id)
                .first()
            )
            return exists is not None
        finally:
            session.close()
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_conversation.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add core/conversation.py tests/test_conversation.py
git commit -m "feat: ConversationManager — per-chat history, save, deduplication"
```

---

## Task 6: RateLimiter

**Files:**
- Create: `core/rate_limiter.py`
- Test: `tests/test_rate_limiter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rate_limiter.py
import pytest
from datetime import datetime, timedelta
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
    past = datetime.utcnow() - timedelta(seconds=11)
    with patch("core.rate_limiter.datetime") as mock_dt:
        mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(seconds=11)
        assert limiter.is_throttled("telegram", "user1") is False

def test_different_senders_independent(limiter):
    limiter.record("telegram", "user1")
    assert limiter.is_throttled("telegram", "user2") is False
```

Run: `pytest tests/test_rate_limiter.py -v`
Expected: FAIL

- [ ] **Step 2: Create core/rate_limiter.py**

```python
from datetime import datetime, timedelta
from storage.db import Database, RateLimitRow


class RateLimiter:
    def __init__(self, db: Database, cooldown_seconds: int = 10):
        self._db = db
        self._cooldown = timedelta(seconds=cooldown_seconds)

    def is_throttled(self, channel: str, sender_id: str) -> bool:
        session = self._db.session()
        try:
            row = session.query(RateLimitRow).filter_by(
                channel=channel, sender_id=sender_id
            ).first()
            if row is None:
                return False
            return datetime.utcnow() - row.last_seen < self._cooldown
        finally:
            session.close()

    def record(self, channel: str, sender_id: str) -> None:
        session = self._db.session()
        try:
            row = session.query(RateLimitRow).filter_by(
                channel=channel, sender_id=sender_id
            ).first()
            now = datetime.utcnow()
            if row:
                row.last_seen = now
            else:
                session.add(RateLimitRow(channel=channel, sender_id=sender_id, last_seen=now))
            session.commit()
        finally:
            session.close()
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_rate_limiter.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add core/rate_limiter.py tests/test_rate_limiter.py
git commit -m "feat: RateLimiter — per-sender cooldown using SQLite"
```

---

## Task 7: PersonaEngine

**Files:**
- Create: `knowledge/persona_engine.py`
- Test: `tests/test_persona.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_persona.py
import pytest
import textwrap
from knowledge.persona_engine import PersonaEngine

PERSONA_YAML = textwrap.dedent("""
    name: "Pandu"
    tone: "friendly, professional"
    language: "English"
    boundaries:
      - "Don't share phone number"
    rules:
      - trigger: "availability"
        response_hint: "Check calendar"
      - trigger: "pricing|cost"
        response_hint: "Redirect to email"
""")

@pytest.fixture
def engine(tmp_path):
    f = tmp_path / "persona.yaml"
    f.write_text(PERSONA_YAML)
    return PersonaEngine(str(f))

def test_name(engine):
    assert engine.name == "Pandu"

def test_build_prompt_contains_name(engine):
    prompt = engine.build_prompt("hello")
    assert "Pandu" in prompt

def test_build_prompt_contains_boundary(engine):
    prompt = engine.build_prompt("hello")
    assert "Don't share phone number" in prompt

def test_trigger_hint_included(engine):
    prompt = engine.build_prompt("what is your availability?")
    assert "Check calendar" in prompt

def test_no_hint_when_no_trigger_match(engine):
    prompt = engine.build_prompt("hello how are you")
    assert "Check calendar" not in prompt
    assert "Redirect to email" not in prompt

def test_multiple_triggers_fire(engine):
    prompt = engine.build_prompt("what is your availability and pricing?")
    assert "Check calendar" in prompt
    assert "Redirect to email" in prompt

def test_trigger_case_insensitive(engine):
    prompt = engine.build_prompt("AVAILABILITY please")
    assert "Check calendar" in prompt

def test_pipe_trigger_alternation(engine):
    prompt = engine.build_prompt("what is the cost?")
    assert "Redirect to email" in prompt
```

Run: `pytest tests/test_persona.py -v`
Expected: FAIL

- [ ] **Step 2: Create knowledge/persona_engine.py**

```python
import re
import yaml


class PersonaEngine:
    def __init__(self, persona_path: str):
        with open(persona_path) as f:
            data = yaml.safe_load(f)
        self.name: str = data["name"]
        self._tone: str = data.get("tone", "friendly")
        self._language: str = data.get("language", "English")
        self._boundaries: list[str] = data.get("boundaries", [])
        self._rules: list[dict] = data.get("rules", [])

    def build_prompt(self, incoming_text: str) -> str:
        boundaries_text = "\n".join(f"- {b}" for b in self._boundaries)
        hints = self._matching_hints(incoming_text)
        hints_text = ("\nContextual guidance:\n" + "\n".join(f"- {h}" for h in hints)) if hints else ""

        return (
            f"You are {self.name}, a personal AI assistant.\n"
            f"Tone: {self._tone}\n"
            f"Language: {self._language}\n"
            f"Boundaries (never violate these):\n{boundaries_text}"
            f"{hints_text}"
        )

    def _matching_hints(self, text: str) -> list[str]:
        hints = []
        for rule in self._rules:
            if re.search(rule["trigger"], text, re.IGNORECASE):
                hints.append(rule["response_hint"])
        return hints
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_persona.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add knowledge/persona_engine.py tests/test_persona.py
git commit -m "feat: PersonaEngine — loads persona.yaml, builds system prompt with trigger-based hints"
```

---

## Task 8: RAG Engine

**Files:**
- Create: `knowledge/rag_engine.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rag.py
import pytest
from knowledge.rag_engine import RAGEngine

@pytest.fixture
def engine(tmp_path):
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "bio.txt").write_text(
        "Pandu is a software engineer specializing in AI and machine learning. "
        "He works at a startup in Hyderabad building recommendation systems."
    )
    chroma_path = str(tmp_path / ".chroma")
    return RAGEngine(
        documents_path=str(docs_path),
        chroma_path=chroma_path,
        top_k=2,
        min_similarity=0.0,   # set to 0 so test content is always returned
    )

@pytest.mark.asyncio
async def test_index_and_search(engine):
    engine.index_documents()
    results = await engine.search("What does Pandu do for work?")
    assert len(results) > 0
    combined = " ".join(results)
    assert "software engineer" in combined.lower() or "AI" in combined

@pytest.mark.asyncio
async def test_search_returns_empty_above_threshold(tmp_path):
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "bio.txt").write_text("Pandu likes hiking and photography.")
    engine = RAGEngine(
        documents_path=str(docs_path),
        chroma_path=str(tmp_path / ".chroma"),
        top_k=3,
        min_similarity=0.99,  # impossibly high threshold
    )
    engine.index_documents()
    results = await engine.search("quantum physics")
    assert results == []

@pytest.mark.asyncio
async def test_search_formats_as_string(engine):
    engine.index_documents()
    result = await engine.search_as_string("AI work")
    assert isinstance(result, str)
```

Run: `pytest tests/test_rag.py -v`
Expected: FAIL (Note: first run downloads the embedding model ~80MB)

- [ ] **Step 2: Create knowledge/rag_engine.py**

```python
import hashlib
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class RAGEngine:
    def __init__(
        self,
        documents_path: str,
        chroma_path: str,
        top_k: int = 3,
        min_similarity: float = 0.4,
    ):
        self._docs_path = Path(documents_path)
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._client = chromadb.PersistentClient(path=chroma_path)
        # cosine space ensures distances match spec's stated cosine similarity requirement
        self._collection = self._client.get_or_create_collection(
            "vedaai_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    def index_documents(self) -> None:
        """Index all .txt and .md files in documents_path. Skips unchanged chunks."""
        for file_path in self._docs_path.glob("**/*"):
            if file_path.suffix not in (".txt", ".md"):
                continue
            text = file_path.read_text(encoding="utf-8")
            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path.name}::{i}"
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
                # Skip if already indexed with same content
                existing = self._collection.get(ids=[chunk_id])
                if existing["ids"] and existing["metadatas"][0].get("hash") == chunk_hash:
                    continue
                embedding = self._model.encode(chunk).tolist()
                self._collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"filename": file_path.name, "chunk": i, "hash": chunk_hash}],
                )

    async def search(self, query: str) -> list[str]:
        """Returns list of relevant text chunks above min_similarity threshold."""
        if self._collection.count() == 0:
            return []
        query_embedding = self._model.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self._top_k, self._collection.count()),
            include=["documents", "distances"],
        )
        chunks = []
        for doc, distance in zip(results["documents"][0], results["distances"][0]):
            # ChromaDB returns L2 distance; convert to similarity score
            similarity = 1 / (1 + distance)
            if similarity >= self._min_similarity:
                chunks.append(doc)
        return chunks

    async def search_as_string(self, query: str) -> str:
        """Returns chunks joined as a single string for prompt injection."""
        """Returns chunks joined as a single string for prompt injection."""
        chunks = await self.search(query)
        return "\n\n".join(chunks) if chunks else ""
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_rag.py -v`
Expected: All PASS (first run downloads ~80MB embedding model)

- [ ] **Step 4: Commit**

```bash
git add knowledge/rag_engine.py tests/test_rag.py
git commit -m "feat: RAGEngine — ChromaDB + sentence-transformers with similarity threshold filtering"
```

---

## Task 9: ClaudeAgent

**Files:**
- Create: `core/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from adapters.base import Message
from core.agent import ClaudeAgent

def make_msg(is_owner=False, text="hello"):
    return Message(
        channel="telegram", chat_id="c1", message_id="m1",
        sender_id="u1", sender_name="Alice",
        text=text, timestamp=datetime(2026,1,1),
        is_group=False, is_owner=is_owner,
    )

@pytest.fixture
def agent():
    conv = AsyncMock()
    conv.get_history.return_value = []
    rag = AsyncMock()
    rag.search_as_string.return_value = "Pandu is an AI engineer."
    persona = MagicMock()
    persona.name = "Pandu"
    persona.build_prompt.return_value = "You are Pandu."
    return ClaudeAgent(
        api_key="test_key",
        model="claude-sonnet-4-6",
        conversation=conv,
        rag=rag,
        persona=persona,
    )

@pytest.mark.asyncio
async def test_auto_reply_calls_rag(agent):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hi there!", type="text")]
    mock_response.stop_reason = "end_turn"
    with patch.object(agent._client.messages, "create", new=AsyncMock(return_value=mock_response)):
        result = await agent.auto_reply_mode(make_msg(is_owner=False))
    agent._rag.search_as_string.assert_called_once()
    assert result == "Hi there!"

@pytest.mark.asyncio
async def test_auto_reply_does_not_use_rag_if_empty(agent):
    agent._rag.search_as_string.return_value = ""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Reply", type="text")]
    mock_response.stop_reason = "end_turn"
    with patch.object(agent._client.messages, "create", new=AsyncMock(return_value=mock_response)):
        result = await agent.auto_reply_mode(make_msg())
    assert result == "Reply"

@pytest.mark.asyncio
async def test_assistant_mode_does_not_auto_inject_rag(agent):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Answer", type="text")]
    mock_response.stop_reason = "end_turn"
    with patch.object(agent._client.messages, "create", new=AsyncMock(return_value=mock_response)):
        result = await agent.assistant_mode(make_msg(is_owner=True))
    # RAG should NOT be called automatically in assistant mode
    agent._rag.search_as_string.assert_not_called()
    assert result == "Answer"
```

Run: `pytest tests/test_agent.py -v`
Expected: FAIL

- [ ] **Step 2: Create core/agent.py**

```python
import asyncio
import anthropic
from adapters.base import Message
from core.conversation import ConversationManager
from knowledge.rag_engine import RAGEngine
from knowledge.persona_engine import PersonaEngine


SEARCH_KNOWLEDGE_TOOL = {
    "name": "search_knowledge",
    "description": "Search Pandu's personal knowledge base for relevant information.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"],
    },
}

GET_HISTORY_TOOL = {
    "name": "get_conversation_history",
    "description": "Retrieve recent conversation history for a specific chat.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chat_id": {"type": "string"},
            "channel": {"type": "string"},
            "n": {"type": "integer", "description": "Number of messages to retrieve", "default": 10},
        },
        "required": ["chat_id", "channel"],
    },
}


class ClaudeAgent:
    def __init__(
        self,
        api_key: str,
        model: str,
        conversation: ConversationManager,
        rag: RAGEngine,
        persona: PersonaEngine,
    ):
        # AsyncAnthropic required — sync client blocks the event loop
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._conv = conversation
        self._rag = rag
        self._persona = persona

    async def auto_reply_mode(self, message: Message) -> str:
        history = await self._conv.get_history(message.channel, message.chat_id)
        rag_context = await self._rag.search_as_string(message.text)
        persona_prompt = self._persona.build_prompt(message.text)

        rag_section = f"\nRelevant knowledge:\n{rag_context}" if rag_context else ""
        system = f"{persona_prompt}{rag_section}"

        try:
            response = await self._client.messages.create(
                model=self._model,
                system=system,
                messages=history + [{"role": "user", "content": message.text}],
                max_tokens=1024,
            )
            return response.content[0].text
        except Exception:
            await asyncio.sleep(2)
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    system=system,
                    messages=history + [{"role": "user", "content": message.text}],
                    max_tokens=1024,
                )
                return response.content[0].text
            except Exception:
                return "I'm unavailable right now, please try again shortly."

    async def assistant_mode(self, message: Message) -> str:
        history = await self._conv.get_history(message.channel, message.chat_id)
        system = f"You are a personal AI assistant for {self._persona.name}. Be concise and direct."

        try:
            response = await self._client.messages.create(
                model=self._model,
                system=system,
                messages=history + [{"role": "user", "content": message.text}],
                tools=[SEARCH_KNOWLEDGE_TOOL, GET_HISTORY_TOOL],
                max_tokens=2048,
            )
            return await self._run_tool_loop(response, message)
        except Exception:
            await asyncio.sleep(2)
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    system=system,
                    messages=history + [{"role": "user", "content": message.text}],
                    tools=[SEARCH_KNOWLEDGE_TOOL, GET_HISTORY_TOOL],
                    max_tokens=2048,
                )
                return await self._run_tool_loop(response, message)
            except Exception:
                return "I'm unavailable right now, please try again shortly."

    async def _run_tool_loop(self, response, message: Message) -> str:
        messages = [{"role": "user", "content": message.text}]

        while response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = await self._execute_tool(block.name, block.input, message)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            response = await self._client.messages.create(
                model=self._model,
                system=f"You are a personal AI assistant for {self._persona.name}.",
                messages=messages,
                tools=[SEARCH_KNOWLEDGE_TOOL, GET_HISTORY_TOOL],
                max_tokens=2048,
            )

        for block in response.content:
            if block.type == "text":
                return block.text
        return "I could not generate a response."

    async def _execute_tool(self, name: str, inputs: dict, message: Message) -> str:
        if name == "search_knowledge":
            return await self._rag.search_as_string(inputs["query"])
        if name == "get_conversation_history":
            history = await self._conv.get_history(
                inputs["channel"], inputs["chat_id"], inputs.get("n", 10)
            )
            return str(history)
        return f"Unknown tool: {name}"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_agent.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add core/agent.py tests/test_agent.py
git commit -m "feat: ClaudeAgent — auto-reply with RAG injection, assistant mode with tool loop, retry on failure"
```

---

## Task 10: Gateway

**Files:**
- Create: `core/gateway.py`
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gateway.py
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

@pytest.fixture
def gateway():
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
    gw = Gateway(
        adapters={"telegram": adapter},
        agent=agent,
        conversation_mgr=conv,
        user_detector=detector,
        rate_limiter=rate_limiter,
    )
    return gw, adapter, agent, conv, detector, rate_limiter

@pytest.mark.asyncio
async def test_auto_reply_for_non_owner(gateway):
    gw, adapter, agent, conv, detector, _ = gateway
    msg = Message(channel="telegram", chat_id="c1", message_id="m1",
                  sender_id="s1", sender_name="S", text="hi",
                  timestamp=datetime(2026,1,1), is_group=False, is_owner=False)
    detector.resolve.return_value = msg
    await gw.handle(make_raw())
    agent.auto_reply_mode.assert_called_once()
    agent.assistant_mode.assert_not_called()

@pytest.mark.asyncio
async def test_assistant_mode_for_owner(gateway):
    gw, adapter, agent, conv, detector, _ = gateway
    msg = Message(channel="telegram", chat_id="c1", message_id="m1",
                  sender_id="owner", sender_name="O", text="hi",
                  timestamp=datetime(2026,1,1), is_group=False, is_owner=True)
    detector.resolve.return_value = msg
    await gw.handle(make_raw(sender_id="owner"))
    agent.assistant_mode.assert_called_once()

@pytest.mark.asyncio
async def test_duplicate_skipped(gateway):
    gw, adapter, agent, conv, detector, _ = gateway
    conv.is_duplicate.return_value = True
    await gw.handle(make_raw())
    agent.auto_reply_mode.assert_not_called()

@pytest.mark.asyncio
async def test_group_message_from_non_owner_ignored(gateway):
    gw, adapter, agent, conv, detector, _ = gateway
    msg = Message(channel="telegram", chat_id="c1", message_id="m1",
                  sender_id="s1", sender_name="S", text="hi",
                  timestamp=datetime(2026,1,1), is_group=True, is_owner=False)
    detector.resolve.return_value = msg
    await gw.handle(make_raw(is_group=True))
    agent.auto_reply_mode.assert_not_called()

@pytest.mark.asyncio
async def test_rate_limited_sender_ignored(gateway):
    gw, adapter, agent, conv, detector, rate_limiter = gateway
    msg = Message(channel="telegram", chat_id="c1", message_id="m1",
                  sender_id="s1", sender_name="S", text="hi",
                  timestamp=datetime(2026,1,1), is_group=False, is_owner=False)
    detector.resolve.return_value = msg
    rate_limiter.is_throttled.return_value = True
    await gw.handle(make_raw())
    agent.auto_reply_mode.assert_not_called()

@pytest.mark.asyncio
async def test_both_sides_saved(gateway):
    gw, adapter, agent, conv, detector, _ = gateway
    msg = Message(channel="telegram", chat_id="c1", message_id="m1",
                  sender_id="s1", sender_name="S", text="hi",
                  timestamp=datetime(2026,1,1), is_group=False, is_owner=False)
    detector.resolve.return_value = msg
    await gw.handle(make_raw())
    conv.save.assert_called_once()
    conv.save_response.assert_called_once()
    # Verify role is NOT passed — it is hardcoded inside save_response per spec
    call_kwargs = conv.save_response.call_args.kwargs
    assert "role" not in call_kwargs

@pytest.mark.asyncio
async def test_concurrent_messages_same_chat_serialized(gateway):
    gw, adapter, agent, conv, detector, _ = gateway
    results = []
    async def slow_reply(msg):
        await asyncio.sleep(0.05)
        results.append("done")
        return "reply"
    agent.auto_reply_mode.side_effect = slow_reply
    msg = Message(channel="telegram", chat_id="c1", message_id="m1",
                  sender_id="s1", sender_name="S", text="hi",
                  timestamp=datetime(2026,1,1), is_group=False, is_owner=False)
    detector.resolve.return_value = msg
    raw1 = make_raw(message_id="m1")
    raw2 = make_raw(message_id="m2")
    conv.is_duplicate.side_effect = [False, False]
    await asyncio.gather(gw.handle(raw1), gw.handle(raw2))
    assert len(results) == 2
```

Run: `pytest tests/test_gateway.py -v`
Expected: FAIL

- [ ] **Step 2: Create core/gateway.py**

```python
import asyncio
import logging
from collections import defaultdict
from adapters.base import RawMessage, BaseAdapter
from core.agent import ClaudeAgent
from core.conversation import ConversationManager
from core.user_detector import UserDetector
from core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class Gateway:
    def __init__(
        self,
        adapters: dict[str, BaseAdapter],
        agent: ClaudeAgent,
        conversation_mgr: ConversationManager,
        user_detector: UserDetector,
        rate_limiter: RateLimiter,
    ):
        self._adapters = adapters
        self._agent = agent
        self._conv = conversation_mgr
        self._detector = user_detector
        self._rate_limiter = rate_limiter
        # Lambda factory for Python 3.10+ compatibility
        self._chat_locks: dict[str, asyncio.Lock] = defaultdict(lambda: asyncio.Lock())

    async def handle(self, raw: RawMessage) -> None:
        if await self._conv.is_duplicate(raw.channel, raw.message_id):
            logger.debug("Duplicate message %s — skipping", raw.message_id)
            return

        message = self._detector.resolve(raw)

        if message.is_group and not message.is_owner:
            logger.debug("Group message from non-owner — ignoring")
            return

        if not message.is_owner and self._rate_limiter.is_throttled(message.channel, message.sender_id):
            logger.debug("Rate-limited sender %s — dropping", message.sender_id)
            return

        if not message.is_owner:
            self._rate_limiter.record(message.channel, message.sender_id)

        lock_key = f"{message.channel}:{message.chat_id}"
        async with self._chat_locks[lock_key]:
            if message.is_owner:
                response = await self._agent.assistant_mode(message)
            else:
                response = await self._agent.auto_reply_mode(message)

            await self._conv.save(message, role="user")
            await self._conv.save_response(message, response)

            adapter = self._adapters.get(message.channel)
            if adapter is None:
                raise RuntimeError(f"No adapter registered for channel: {message.channel}")
            await adapter.send_message(message.chat_id, response)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_gateway.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add core/gateway.py tests/test_gateway.py
git commit -m "feat: Gateway — message routing with dedup, group filtering, rate limiting, per-chat locking"
```

---

## Task 11: Telegram Adapter

**Files:**
- Create: `adapters/telegram.py`
- Modify: `tests/test_adapters.py` (add Telegram tests)

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_adapters.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from adapters.telegram import TelegramAdapter

@pytest.fixture
def tg_adapter():
    return TelegramAdapter(bot_token="fake_token", owner_id="999")

@pytest.mark.asyncio
async def test_telegram_on_message_registers_callback(tg_adapter):
    callback = AsyncMock()
    # on_message should store the callback without error
    await tg_adapter.on_message(callback)
    assert tg_adapter._callback == callback

@pytest.mark.asyncio
async def test_telegram_handles_text_message(tg_adapter):
    callback = AsyncMock()
    await tg_adapter.on_message(callback)
    # Simulate incoming update
    mock_update = MagicMock()
    mock_update.message.text = "Hello"
    mock_update.message.message_id = 42
    mock_update.message.chat.id = "chat1"
    mock_update.message.from_user.id = 123
    mock_update.message.from_user.first_name = "Alice"
    mock_update.message.chat.type = "private"
    mock_update.message.date.timestamp.return_value = 1711500000.0
    await tg_adapter._handle_update(mock_update, None)
    callback.assert_called_once()
    raw = callback.call_args[0][0]
    assert raw.channel == "telegram"
    assert raw.text == "Hello"
    assert raw.sender_id == "123"
    assert raw.is_group is False

@pytest.mark.asyncio
async def test_telegram_ignores_non_text(tg_adapter):
    callback = AsyncMock()
    await tg_adapter.on_message(callback)
    mock_update = MagicMock()
    mock_update.message.text = None  # e.g. a photo message
    await tg_adapter._handle_update(mock_update, None)
    callback.assert_not_called()
```

Run: `pytest tests/test_adapters.py -v`
Expected: FAIL on new Telegram tests

- [ ] **Step 2: Create adapters/telegram.py**

```python
import logging
from datetime import datetime
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from adapters.base import BaseAdapter, RawMessage, MessageCallback

logger = logging.getLogger(__name__)

GROUP_TYPES = {"group", "supergroup"}


class TelegramAdapter(BaseAdapter):
    channel = "telegram"

    def __init__(self, bot_token: str, owner_id: str):
        self._token = bot_token
        self._owner_id = owner_id
        self._callback: MessageCallback | None = None
        self._app: Application | None = None

    async def on_message(self, callback: MessageCallback) -> None:
        self._callback = callback

    async def connect(self) -> None:
        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(
            MessageHandler(filters.ALL, self._handle_update)
        )
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

    async def _handle_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            if update.message and not update.message.text:
                # Non-text message — respond with limitation notice via callback
                if self._callback:
                    raw = RawMessage(
                        channel=self.channel,
                        chat_id=str(update.message.chat.id),
                        message_id=str(update.message.message_id),
                        sender_id=str(update.message.from_user.id),
                        sender_name=update.message.from_user.first_name or "",
                        text="[non-text message]",
                        timestamp=datetime.utcfromtimestamp(update.message.date.timestamp()),
                        is_group=update.message.chat.type in GROUP_TYPES,
                    )
                    await self._callback(raw)
            return

        if not self._callback:
            return

        raw = RawMessage(
            channel=self.channel,
            chat_id=str(update.message.chat.id),
            message_id=str(update.message.message_id),
            sender_id=str(update.message.from_user.id),
            sender_name=update.message.from_user.first_name or "",
            text=update.message.text,
            timestamp=datetime.utcfromtimestamp(update.message.date.timestamp()),
            is_group=update.message.chat.type in GROUP_TYPES,
        )
        await self._callback(raw)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_adapters.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add adapters/telegram.py tests/test_adapters.py
git commit -m "feat: TelegramAdapter — python-telegram-bot v20, group detection, non-text handling"
```

---

## Task 12: WhatsApp Bridge (Node.js)

**Files:**
- Create: `bridge/package.json`
- Create: `bridge/index.js`
- Create: `.gitignore` (to exclude bridge/node_modules and session files)

- [ ] **Step 1: Create bridge/package.json**

```json
{
  "name": "vedaai-whatsapp-bridge",
  "version": "1.0.0",
  "description": "WhatsApp Web bridge for VedaAI",
  "main": "index.js",
  "dependencies": {
    "whatsapp-web.js": "^1.23.0",
    "qrcode-terminal": "^0.12.0",
    "express": "^4.18.0"
  }
}
```

- [ ] **Step 2: Install Node.js dependencies**

```bash
cd /Users/anilkumar/Prasanna/vedaai/bridge
npm install
```

- [ ] **Step 3: Create bridge/index.js**

```javascript
const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const express = require('express');

const app = express();
app.use(express.json());

const PORT = process.env.BRIDGE_PORT || 3000;
const pendingMessages = [];

const client = new Client({
    authStrategy: new LocalAuth({ dataPath: './session' }),
    puppeteer: { args: ['--no-sandbox', '--disable-setuid-sandbox'] }
});

client.on('qr', (qr) => {
    console.log('Scan this QR code with WhatsApp:');
    qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
    console.log('WhatsApp ready');
});

client.on('message', async (msg) => {
    if (!msg.body) return;
    const chat = await msg.getChat();
    pendingMessages.push({
        message_id: msg.id._serialized,
        from: msg.from,
        from_name: msg._data.notifyName || msg.from,
        chat_id: msg.from,
        body: msg.body,
        timestamp: msg.timestamp,
        is_group: chat.isGroup,
    });
});

// Python adapter polls this endpoint
app.get('/messages/pending', (req, res) => {
    const msgs = [...pendingMessages];
    pendingMessages.length = 0;
    res.json(msgs);
});

// Python adapter sends messages via this endpoint
app.post('/send', async (req, res) => {
    const { chat_id, message } = req.body;
    try {
        await client.sendMessage(chat_id, message);
        res.json({ status: 'sent' });
    } catch (err) {
        res.status(500).json({ status: 'error', reason: err.message });
    }
});

app.listen(PORT, () => {
    console.log(`Bridge listening on port ${PORT}`);
});

client.initialize();
```

- [ ] **Step 4: Create .gitignore**

```
venv/
__pycache__/
*.pyc
.env
vedaai.db
.chroma/
bridge/node_modules/
bridge/session/
.DS_Store
```

- [ ] **Step 5: Commit**

```bash
git add bridge/ .gitignore
git commit -m "feat: WhatsApp bridge — whatsapp-web.js Node.js HTTP server with QR auth and polling API"
```

---

## Task 13: WhatsApp Python Adapter

**Files:**
- Create: `adapters/whatsapp.py`
- Modify: `tests/test_adapters.py` (add WhatsApp tests)

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_adapters.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from adapters.whatsapp import WhatsAppAdapter

@pytest.fixture
def wa_adapter():
    return WhatsAppAdapter(
        bridge_url="http://localhost:3000",
        bridge_script="bridge/index.js",
        polling_interval=0.01,  # fast polling for tests
    )

@pytest.mark.asyncio
async def test_whatsapp_on_message_registers_callback(wa_adapter):
    callback = AsyncMock()
    await wa_adapter.on_message(callback)
    assert wa_adapter._callback == callback

@pytest.mark.asyncio
async def test_whatsapp_polls_and_dispatches(wa_adapter):
    callback = AsyncMock()
    await wa_adapter.on_message(callback)
    bridge_payload = [{
        "message_id": "WA_001",
        "from": "+91111",
        "from_name": "Bob",
        "chat_id": "91111@c.us",
        "body": "Hey",
        "timestamp": 1711500000,
        "is_group": False,
    }]
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=bridge_payload)
        mock_get.return_value.__aenter__.return_value = mock_resp
        await wa_adapter._poll_once()
    callback.assert_called_once()
    raw = callback.call_args[0][0]
    assert raw.channel == "whatsapp"
    assert raw.text == "Hey"
    assert raw.message_id == "WA_001"
```

Run: `pytest tests/test_adapters.py -v`
Expected: FAIL on WhatsApp tests

- [ ] **Step 2: Create adapters/whatsapp.py**

```python
import asyncio
import logging
from datetime import datetime
import aiohttp
from adapters.base import BaseAdapter, RawMessage, MessageCallback

logger = logging.getLogger(__name__)


class WhatsAppAdapter(BaseAdapter):
    channel = "whatsapp"

    def __init__(self, bridge_url: str, bridge_script: str, polling_interval: float = 2.0):
        self._bridge_url = bridge_url.rstrip("/")
        self._bridge_script = bridge_script
        self._polling_interval = polling_interval
        self._callback: MessageCallback | None = None
        self._proc = None
        self._running = False

    async def on_message(self, callback: MessageCallback) -> None:
        self._callback = callback

    async def connect(self) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "node", self._bridge_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await self._wait_for_ready(timeout=30)
            self._running = True
            asyncio.create_task(self._poll_loop())
            logger.info("WhatsApp adapter connected")
        except (RuntimeError, FileNotFoundError) as e:
            logger.warning("WhatsApp bridge failed to start: %s — continuing without WhatsApp", e)

    async def disconnect(self) -> None:
        self._running = False
        if self._proc:
            self._proc.terminate()

    async def send_message(self, chat_id: str, text: str) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._bridge_url}/send",
                json={"chat_id": chat_id, "message": text},
            ) as resp:
                data = await resp.json()
                if data.get("status") != "sent":
                    logger.error("WhatsApp send failed: %s", data.get("reason"))

    async def _wait_for_ready(self, timeout: int) -> None:
        if not self._proc:
            raise RuntimeError("Bridge process not started")
        async def _read():
            while True:
                if self._proc.stdout is None:
                    break
                line = await self._proc.stdout.readline()
                if b"WhatsApp ready" in line:
                    return
        try:
            await asyncio.wait_for(_read(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError("WhatsApp bridge failed to start within 30 seconds")

    async def _poll_loop(self) -> None:
        while self._running:
            await self._poll_once()
            await asyncio.sleep(self._polling_interval)

    async def _poll_once(self) -> None:
        if not self._callback:
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._bridge_url}/messages/pending") as resp:
                    messages = await resp.json()
            for m in messages:
                raw = RawMessage(
                    channel=self.channel,
                    chat_id=m["chat_id"],
                    message_id=m["message_id"],
                    sender_id=m["from"],
                    sender_name=m.get("from_name", m["from"]),
                    text=m["body"],
                    timestamp=datetime.utcfromtimestamp(m["timestamp"]),
                    is_group=m.get("is_group", False),
                )
                await self._callback(raw)
        except Exception as e:
            logger.warning("WhatsApp polling error: %s", e)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_adapters.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add adapters/whatsapp.py tests/test_adapters.py
git commit -m "feat: WhatsAppAdapter — polling bridge with graceful degradation on startup failure"
```

---

## Task 14: Config Loader

**Files:**
- Create: `core/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
import os
import pytest
from unittest.mock import patch
from core.config import load_settings

SAMPLE_YAML = """
anthropic:
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-sonnet-4-6"
owner:
  telegram: "999"
  whatsapp: "+91111"
nested:
  list:
    - "${SOME_VAR}"
    - "plain"
"""

@pytest.fixture
def yaml_file(tmp_path):
    f = tmp_path / "settings.yaml"
    f.write_text(SAMPLE_YAML)
    return str(f)

def test_plain_value_unchanged(yaml_file):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        cfg = load_settings(yaml_file)
    assert cfg["anthropic"]["model"] == "claude-sonnet-4-6"

def test_env_var_substituted(yaml_file):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        cfg = load_settings(yaml_file)
    assert cfg["anthropic"]["api_key"] == "sk-test"

def test_missing_env_var_resolves_to_empty(yaml_file):
    with patch.dict(os.environ, {}, clear=True):
        cfg = load_settings(yaml_file)
    assert cfg["anthropic"]["api_key"] == ""

def test_nested_list_resolved(yaml_file):
    with patch.dict(os.environ, {"SOME_VAR": "hello"}):
        cfg = load_settings(yaml_file)
    assert cfg["nested"]["list"][0] == "hello"
    assert cfg["nested"]["list"][1] == "plain"
```

Run: `pytest tests/test_config.py -v`
Expected: FAIL

- [ ] **Step 2: Create core/config.py**

```python
import os
import re
import yaml
from dotenv import load_dotenv

load_dotenv()

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value):
    if isinstance(value, str):
        def replace(match):
            env_val = os.environ.get(match.group(1), "")
            return env_val
        return _ENV_VAR_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _resolve_env_vars(raw)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 4: Alembic — defer migrations to v2**

Alembic is included in `requirements.txt` for future use but migrations are not initialised in v1. The app creates tables directly via `db.create_tables()`. Remove the `alembic` line from `requirements.txt` to avoid confusion, and add it back when the first migration is needed:

```bash
# Remove alembic from requirements.txt (v1 uses create_tables() directly)
sed -i '' '/^alembic/d' requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add core/config.py tests/test_config.py requirements.txt
git commit -m "feat: config loader — resolves \${ENV_VAR} substitution in settings.yaml"
```

---

## Task 15: main.py — Wire Everything Together

**Files:**
- Create: `main.py`

- [ ] **Step 1: Create main.py**

```python
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

    # Index documents on startup
    logger.info("Indexing knowledge documents...")
    rag.index_documents()
    logger.info("Indexing complete.")

    agent = ClaudeAgent(
        api_key=cfg["anthropic"]["api_key"],
        model=cfg["anthropic"]["model"],
        conversation=conv,
        rag=rag,
        persona=persona,
    )

    # Adapters
    tg = TelegramAdapter(
        bot_token=cfg["telegram"]["bot_token"],
        owner_id=cfg["owner"]["telegram"],
    )
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

    # Connect adapters (WhatsApp gracefully degrades on failure)
    logger.info("Connecting adapters...")
    await tg.connect()
    await wa.connect()
    logger.info("VedaAI is running. Press Ctrl+C to stop.")

    try:
        await asyncio.Event().wait()   # run forever
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
        await tg.disconnect()
        await wa.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run full test suite to confirm nothing is broken**

```bash
pytest tests/ -v --ignore=tests/test_integration.py
```

Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: main.py — wires all components and starts VedaAI"
```

---

## Task 16: Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
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
def full_system(tmp_path):
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
    rag.index_documents()

    mock_adapter = AsyncMock()
    mock_adapter.channel = "telegram"

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello stranger!", type="text")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response

    agent = ClaudeAgent(
        api_key="test", model="claude-sonnet-4-6",
        conversation=conv, rag=rag, persona=persona,
    )
    agent._client = mock_client

    gateway = Gateway(
        adapters={"telegram": mock_adapter},
        agent=agent,
        conversation_mgr=conv,
        user_detector=user_detector,
        rate_limiter=rate_limiter,
    )
    return gateway, mock_adapter, conv

@pytest.mark.asyncio
async def test_full_auto_reply_flow(full_system):
    gateway, adapter, conv = full_system
    raw = RawMessage(
        channel="telegram", chat_id="chat1", message_id="int_m1",
        sender_id="stranger", sender_name="Bob",
        text="Hi Pandu!", timestamp=datetime(2026,1,1),
        is_group=False,
    )
    await gateway.handle(raw)
    adapter.send_message.assert_called_once_with("chat1", "Hello stranger!")
    history = await conv.get_history("telegram", "chat1")
    assert len(history) == 2   # user + assistant

@pytest.mark.asyncio
async def test_full_owner_assistant_flow(full_system):
    gateway, adapter, conv = full_system
    # Owner messages bot — should use assistant mode (tools included in API call)
    mock_client = gateway._agent._client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Here is what I know!", type="text")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    raw = RawMessage(
        channel="telegram", chat_id="owner_chat", message_id="owner_m1",
        sender_id="owner123", sender_name="Pandu",
        text="What do you know about me?", timestamp=datetime(2026,1,1),
        is_group=False,
    )
    await gateway.handle(raw)
    adapter.send_message.assert_called_once_with("owner_chat", "Here is what I know!")
    # Verify tools were passed (assistant mode uses tools; auto-reply does not)
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "tools" in call_kwargs

@pytest.mark.asyncio
async def test_duplicate_message_not_processed_twice(full_system):
    gateway, adapter, conv = full_system
    raw = RawMessage(
        channel="telegram", chat_id="chat1", message_id="dup_m1",
        sender_id="stranger", sender_name="Bob",
        text="Hi", timestamp=datetime(2026,1,1), is_group=False,
    )
    await gateway.handle(raw)
    await gateway.handle(raw)  # same message_id
    assert adapter.send_message.call_count == 1
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 4: Final commit and push**

```bash
git add tests/test_integration.py
git commit -m "test: integration test — full auto-reply flow and deduplication"
git push origin main
```

---

## Task 17: Manual Telegram Smoke Test

**Prerequisite:** Create a Telegram bot via @BotFather and get your bot token.

- [ ] **Step 1: Copy and fill in your config**

```bash
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY and TELEGRAM_BOT_TOKEN
```

- [ ] **Step 2: Update settings.yaml with your Telegram user ID**

To find your Telegram user ID: message @userinfobot on Telegram.

Edit `config/settings.yaml`:
```yaml
owner:
  telegram: "YOUR_ACTUAL_TELEGRAM_USER_ID"
```

- [ ] **Step 3: Add a sample knowledge document**

```bash
echo "Pandu is a software engineer who specializes in AI. He is available on weekdays 10am-6pm IST." \
  > knowledge/documents/bio.txt
```

- [ ] **Step 4: Start VedaAI**

```bash
source venv/bin/activate
python main.py
```

Expected output:
```
INFO vedaai — Indexing knowledge documents...
INFO vedaai — Indexing complete.
INFO vedaai — Connecting adapters...
INFO vedaai — VedaAI is running. Press Ctrl+C to stop.
```

- [ ] **Step 5: Test assistant mode**

Message your bot from your own Telegram account:
> "What do you know about me?"

Expected: Claude responds using the bio document.

- [ ] **Step 6: Test auto-reply mode**

From a different Telegram account (or ask a friend) message the bot:
> "Hello, is Pandu available?"

Expected: Claude responds as Pandu using persona rules.

---

## Done

All tasks complete. VedaAI is running locally with:
- Telegram connected and tested
- WhatsApp bridge ready (connect by running bridge separately and scanning QR)
- Full test suite passing
- Code pushed to https://github.com/Prasannavattikoda-AI-ML/vedaai
