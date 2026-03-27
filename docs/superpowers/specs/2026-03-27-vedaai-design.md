# VedaAI — Design Specification
**Date:** 2026-03-27
**Author:** Pandu
**Status:** Approved

---

## Overview

VedaAI is a Python-based personal AI assistant powered by Anthropic's Claude Agent SDK. It connects to WhatsApp and Telegram, operates in hybrid mode (auto-replies to others on your behalf + personal assistant when you message it directly), and uses a combination of persona rules and RAG-based knowledge to respond intelligently.

---

## Goals

- Act as a 24/7 intelligent representative on WhatsApp and Telegram
- Answer incoming messages using persona rules + indexed personal knowledge
- Serve as a personal AI assistant when the owner messages the bot directly
- Run locally on macOS, with a clear path to cloud deployment
- Be extensible: adding new channels should require only one new file

---

## Non-Goals (v1)

- Instagram and LinkedIn integrations (deferred to v2)
- Voice or screen control
- Multi-user support (single owner only)
- Web dashboard or UI

---

## Architecture

### Pattern: Monolithic Gateway

A single Python process hosts all components. Adapters connect concurrently via `asyncio`. A central gateway routes messages to the correct processing pipeline.

```
[WhatsApp Adapter]──┐
                    ├──▶ [Gateway / Message Router]
[Telegram Adapter]──┘           │
                          [User Detector]
                         /              \
                  [Auto-Reply]      [Assistant]
                  (others)           (owner)
                       │                │
               [Persona Engine]   [Claude Agent]
               [RAG Engine]       [Tools: RAG, history]
                       └────────────────┘
                                │
                         [Claude Agent SDK]
                                │
                         [Response Router]
                        /               \
               [WhatsApp Adapter]  [Telegram Adapter]
```

---

## Project Structure

```
vedaai/
├── main.py                        # Entry point — starts gateway + all adapters
├── config/
│   ├── settings.yaml              # API keys, ports, adapter configs
│   └── persona.yaml               # Persona rules, tone, boundaries
├── core/
│   ├── gateway.py                 # Central message router
│   ├── agent.py                   # Claude Agent SDK wrapper
│   ├── user_detector.py           # Identifies owner vs. others
│   └── conversation.py            # Conversation history + context manager
├── adapters/
│   ├── base.py                    # Abstract BaseAdapter interface
│   ├── whatsapp.py                # WhatsApp via pywa or web.js bridge
│   └── telegram.py                # Telegram via python-telegram-bot
├── knowledge/
│   ├── rag_engine.py              # Document chunking, embedding, retrieval
│   ├── persona_engine.py          # Loads and applies persona.yaml rules
│   └── documents/                 # Owner's files/notes for RAG indexing
├── storage/
│   ├── db.py                      # SQLAlchemy ORM (SQLite locally)
│   └── migrations/                # Alembic migrations
├── tests/
│   ├── test_gateway.py
│   ├── test_agent.py
│   ├── test_rag.py
│   └── test_adapters.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Core Components

### 1. BaseAdapter (`adapters/base.py`)

Abstract interface every channel adapter must implement:

```python
class BaseAdapter(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None: ...

    @abstractmethod
    async def on_message(self, callback: Callable[[Message], Awaitable[None]]) -> None: ...

    @abstractmethod
    def is_owner(self, sender_id: str) -> bool: ...
```

**Standard Message object:**
```python
@dataclass
class Message:
    channel: str          # "whatsapp" | "telegram"
    chat_id: str          # conversation identifier
    sender_id: str        # who sent it
    sender_name: str
    text: str
    timestamp: datetime
    is_owner: bool        # populated by user_detector
```

### 2. Gateway (`core/gateway.py`)

Receives normalized `Message` objects from all adapters, determines mode (auto-reply vs. assistant), dispatches to Claude Agent, sends response back.

```python
async def route(message: Message) -> None:
    if message.is_owner:
        response = await agent.assistant_mode(message)
    else:
        response = await agent.auto_reply_mode(message)
    await adapters[message.channel].send_message(message.chat_id, response)
```

### 3. Claude Agent (`core/agent.py`)

Wraps the Anthropic Agent SDK. Builds dynamic system prompts and calls tools.

**Auto-reply mode system prompt:**
```
You are {persona.name}, a helpful AI assistant representing them.
Tone: {persona.tone}
Language: {persona.language}
Boundaries: {persona.boundaries}
Relevant knowledge: {rag_context}
Conversation history: {last_20_messages}
```

**Assistant mode:** Full Claude agent with tools:
- `search_knowledge(query)` — RAG search over documents
- `get_conversation_history(chat_id, n)` — retrieve past messages
- Extensible for future tools (reminders, calendar, etc.)

**Model:** `claude-sonnet-4-6` (fast + cost-effective for frequent messages)

### 4. RAG Engine (`knowledge/rag_engine.py`)

- **Vector store:** ChromaDB (local, persistent)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Chunking:** 512 tokens, 50-token overlap
- **Retrieval:** Top-3 chunks by cosine similarity
- **Triggered:** Only in auto-reply mode (not assistant mode)

### 5. Persona Engine (`knowledge/persona_engine.py`)

Loads `persona.yaml` and produces a structured system prompt fragment. Supports:
- Tone and language settings
- Hard boundaries (never do X)
- Trigger-based response hints (if message contains "availability", hint the model to check context)

**`config/persona.yaml`:**
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

### 6. Storage (`storage/db.py`)

SQLite via SQLAlchemy ORM. Two tables:

```sql
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY,
    channel     TEXT NOT NULL,
    chat_id     TEXT NOT NULL,
    sender_id   TEXT NOT NULL,
    role        TEXT NOT NULL,  -- "user" | "assistant"
    content     TEXT NOT NULL,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE knowledge_docs (
    id              INTEGER PRIMARY KEY,
    filename        TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding_hash  TEXT NOT NULL,
    indexed_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Cloud swap:** Change SQLite URL to PostgreSQL in `settings.yaml` — SQLAlchemy handles the rest.

---

## Channel Adapters

### WhatsApp (`adapters/whatsapp.py`)

- **Library:** `pywa` (Python WhatsApp Cloud API wrapper)
- **Auth:** Meta WhatsApp Business API token (or local `whatsapp-web.js` bridge via subprocess for personal accounts)
- **Note:** For personal use without a business account, the bridge approach is used — a Node.js subprocess exposes a local HTTP server that `whatsapp.py` communicates with

### Telegram (`adapters/telegram.py`)

- **Library:** `python-telegram-bot` (v20+, async)
- **Auth:** Bot token from @BotFather
- **Most reliable:** Official API, generous rate limits

---

## User Detection (`core/user_detector.py`)

Determines if the sender is the owner:

```python
class UserDetector:
    def __init__(self, owner_ids: dict[str, str]):
        # owner_ids = {"whatsapp": "+91XXXXXXXXXX", "telegram": "123456789"}
        self.owner_ids = owner_ids

    def is_owner(self, channel: str, sender_id: str) -> bool:
        return self.owner_ids.get(channel) == sender_id
```

Owner IDs configured in `settings.yaml`.

---

## Configuration (`config/settings.yaml`)

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
  mode: "bridge"  # "bridge" | "cloud_api"
  bridge_url: "http://localhost:3000"

database:
  url: "sqlite:///vedaai.db"

rag:
  documents_path: "knowledge/documents"
  chroma_path: ".chroma"
  top_k: 3

conversation:
  history_window: 20  # messages to include in context
```

---

## Data Flow (End-to-End)

**Incoming message from someone else (auto-reply):**
1. Telegram/WhatsApp adapter receives message, normalizes to `Message`
2. Gateway receives `Message`, `user_detector` flags `is_owner=False`
3. Gateway calls `agent.auto_reply_mode(message)`
4. Agent loads last 20 messages from SQLite
5. Agent calls `rag_engine.search(message.text)` → top 3 chunks
6. Agent builds system prompt from persona + RAG context
7. Claude generates response
8. Gateway sends response via adapter
9. Both message and response saved to SQLite

**Owner messages bot (assistant mode):**
1. Adapter receives message, `is_owner=True`
2. Gateway calls `agent.assistant_mode(message)`
3. Claude has full tool access: RAG search, conversation history
4. Claude responds directly
5. Saved to SQLite

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Claude API error | Retry once, then send fallback: "I'm unavailable right now, please try again shortly." |
| Adapter disconnect | Log error, attempt reconnect with exponential backoff (max 5 retries) |
| RAG retrieval failure | Fall back to persona-only response (no RAG context) |
| Unknown message type (media, sticker) | Politely acknowledge: "I can only process text messages currently." |

---

## Testing Strategy

- **Unit tests:** Each component tested in isolation (agent, RAG, persona engine, user detector)
- **Adapter tests:** Mocked channel libraries (no real WhatsApp/Telegram needed)
- **Integration test:** Full message flow from raw adapter input to response output
- **Manual testing:** Real Telegram bot (easy to set up) used for live verification

---

## Deployment Path

```
Phase 1 (now):   python main.py  →  runs locally on Mac
Phase 2 (later): Dockerfile + docker-compose
Phase 3 (cloud): Railway / Render / EC2 — just change DB URL + env vars
```

All secrets via environment variables, never hardcoded.

---

## Dependencies

```
anthropic>=0.40.0          # Claude Agent SDK
python-telegram-bot>=20.0  # Telegram adapter
pywa>=1.0.0                # WhatsApp adapter
chromadb>=0.4.0            # Vector store
sentence-transformers       # Local embeddings
sqlalchemy>=2.0            # ORM
alembic                    # DB migrations
pyyaml                     # Config loading
python-dotenv              # .env support
pytest                     # Testing
pytest-asyncio             # Async test support
```

---

## Future Enhancements (v2+)

- Instagram DM adapter
- LinkedIn message adapter
- Reminder/scheduling tool
- Web dashboard for conversation review
- Voice message transcription (Whisper)
- Multi-language persona support
