# VedaAI — Design Specification
**Date:** 2026-03-27
**Author:** Pandu
**Status:** Approved
**GitHub:** https://github.com/Prasannavattikoda-AI-ML/vedaai

---

## Overview

VedaAI is a Python-based personal AI assistant powered by Anthropic's Claude Agent SDK. It connects to WhatsApp and Telegram, operates in hybrid mode (auto-replies to others on your behalf + personal assistant when you message it directly), and uses a combination of persona rules and RAG-based knowledge to respond intelligently.

---

## Goals

- Act as a 24/7 intelligent representative on WhatsApp and Telegram
- Answer incoming messages using persona rules + RAG-indexed personal knowledge
- Serve as a personal AI assistant when the owner messages the bot directly
- Run locally on macOS, with a clear path to cloud deployment
- Be extensible: adding new channels requires only one new adapter file

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
                     [UserDetector.resolve(message)]
                         /              \
                  [Auto-Reply]      [Assistant]
                  (others)           (owner)
                       │                │
               [Persona Engine]    [Claude Agent]
               [RAG Engine]        [Tools: explicit RAG,
               [auto-injected]      conversation history]
                       └────────────────┘
                                │
                         [Claude Agent SDK]
                                │
                      [ConversationManager.save()]
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
│   ├── user_detector.py           # Resolves sender identity (owner vs. other)
│   └── conversation.py            # ConversationManager — history read/write
├── adapters/
│   ├── base.py                    # Abstract BaseAdapter interface
│   ├── whatsapp.py                # WhatsApp via whatsapp-web.js HTTP bridge
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

Abstract interface every channel adapter must implement. `is_owner` is NOT on the adapter — owner detection is centralised in `UserDetector` (see below). The adapter only normalises messages and sends responses.

```python
class BaseAdapter(ABC):
    channel: str  # "whatsapp" | "telegram"

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None: ...

    @abstractmethod
    async def on_message(
        self,
        callback: Callable[[RawMessage], Awaitable[None]]
    ) -> None:
        """
        Called ONCE at startup to register the message handler.
        The adapter stores the callback and invokes it for each
        incoming message for the lifetime of the process.
        """
        ...
```

**Standard Message object** (produced by Gateway after enrichment):
```python
@dataclass
class Message:
    channel: str          # "whatsapp" | "telegram"
    chat_id: str          # conversation identifier (unique per channel)
    message_id: str       # platform-assigned ID (for deduplication)
    sender_id: str        # who sent it
    sender_name: str
    text: str
    timestamp: datetime
    is_group: bool        # True if the message came from a group chat
    is_owner: bool        # populated by UserDetector inside Gateway
```

**RawMessage** (produced by adapters before enrichment):
```python
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
```

### 2. UserDetector (`core/user_detector.py`)

Single source of truth for owner detection. Called by the Gateway — never by adapters.

```python
class UserDetector:
    def __init__(self, owner_ids: dict[str, str]):
        # Validated at startup — raises ValueError if any owner ID is missing
        # owner_ids = {"whatsapp": "+91XXXXXXXXXX", "telegram": "123456789"}
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
```

### 3. Gateway (`core/gateway.py`)

Receives `RawMessage` objects from adapters, enriches them via `UserDetector`, manages concurrency per chat, and dispatches to the agent.

```python
class Gateway:
    def __init__(self, adapters, agent, conversation_mgr, user_detector, rate_limiter):
        self._adapters: dict[str, BaseAdapter] = adapters
        self._agent = agent
        self._conv = conversation_mgr
        self._detector = user_detector
        self._rate_limiter = rate_limiter
        # Per chat_id lock to prevent concurrent responses to same conversation
        # Lambda factory required for Python 3.10+ compatibility (do not use defaultdict(asyncio.Lock))
        self._chat_locks: dict[str, asyncio.Lock] = defaultdict(lambda: asyncio.Lock())

    async def handle(self, raw: RawMessage) -> None:
        # Deduplication: skip if message_id already processed
        if await self._conv.is_duplicate(raw.channel, raw.message_id):
            return

        message = self._detector.resolve(raw)

        # Group chat policy: only respond if owner is messaging (assistant mode)
        # Ignore all auto-reply triggers in group chats
        if message.is_group and not message.is_owner:
            return

        # Serialize responses per conversation (prevent race conditions)
        async with self._chat_locks[f"{message.channel}:{message.chat_id}"]:
            if message.is_owner:
                response = await self._agent.assistant_mode(message)
            else:
                response = await self._agent.auto_reply_mode(message)

            # Save both sides of the conversation
            await self._conv.save(message, role="user")
            await self._conv.save_response(message, response, role="assistant")

            adapter = self._adapters.get(message.channel)
            if adapter is None:
                raise RuntimeError(f"No adapter registered for channel: {message.channel}")
            await adapter.send_message(message.chat_id, response)
```

**Per-sender rate limiting** is enforced inside `handle()` before the lock:
```python
# Max 1 response per 10 seconds per sender (auto-reply only)
if not message.is_owner:
    if self._rate_limiter.is_throttled(message.channel, message.sender_id):
        return  # silently drop; do not reply to throttled senders
    self._rate_limiter.record(message.channel, message.sender_id)
```

### 4. ConversationManager (`core/conversation.py`)

Encapsulates all conversation history operations. `agent.py` uses this interface — it never queries `db.py` directly.

```python
class ConversationManager:
    def __init__(self, db: Database):
        self._db = db

    async def get_history(
        self,
        channel: str,
        chat_id: str,
        limit: int = 20
    ) -> list[dict]:
        """
        Returns the last `limit` messages for a specific (channel, chat_id) pair.
        History is always per-chat, never global.
        Returns list of {"role": "user"|"assistant", "content": str} dicts
        compatible with the Claude messages format.
        """
        ...

    async def save(self, message: Message, role: str) -> None:
        """Saves an incoming message to the conversation history."""
        ...

    async def save_response(self, message: Message, response: str) -> None:
        """Saves an outgoing assistant response to the conversation history.
        Role is always 'assistant' — hardcoded internally to prevent misuse."""
        ...

    async def is_duplicate(self, channel: str, message_id: str) -> bool:
        """
        Returns True if message_id has already been processed for this channel.
        Used to handle webhook replay / double-delivery from WhatsApp Cloud API.
        """
        ...
```

### 5. Claude Agent (`core/agent.py`)

Wraps the Anthropic Agent SDK. Builds dynamic system prompts and calls tools.

**Auto-reply mode** — RAG is automatically injected; owner does not choose what knowledge to surface:
```python
async def auto_reply_mode(self, message: Message) -> str:
    history = await self._conv.get_history(message.channel, message.chat_id)
    rag_context = await self._rag.search(message.text)  # auto-injected
    persona_prompt = self._persona.build_prompt()

    system = f"""
{persona_prompt}

Relevant knowledge about {self._persona.name}:
{rag_context}

Conversation so far:
{format_history(history)}
"""
    response = await self._client.messages.create(
        model=self._model,
        system=system,
        messages=[{"role": "user", "content": message.text}],
        max_tokens=1024,
    )
    return response.content[0].text
```

**Assistant mode** — Full agent with explicit tools; owner can ask Claude to search knowledge:
```python
async def assistant_mode(self, message: Message) -> str:
    history = await self._conv.get_history(message.channel, message.chat_id)
    # RAG is NOT auto-injected here — owner invokes it via tool call
    # Claude decides whether to search knowledge based on the question
    response = await self._client.messages.create(
        model=self._model,
        system="You are a personal AI assistant for Pandu. Be concise and direct.",
        messages=history + [{"role": "user", "content": message.text}],
        tools=[search_knowledge_tool, get_conversation_history_tool],
        max_tokens=2048,
    )
    # Handle tool use loop (tool calls → results → final response)
    return await self._run_tool_loop(response, message)
```

**Tools available in assistant mode:**
- `search_knowledge(query: str)` — explicit RAG search over documents
- `get_conversation_history(chat_id: str, channel: str, n: int)` — retrieve past messages

**Model:** `claude-sonnet-4-6`

### 6. RAG Engine (`knowledge/rag_engine.py`)

- **Vector store:** ChromaDB (local, persistent to `.chroma/` directory)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Chunking:** 512 tokens, 50-token overlap
- **Retrieval:** Top-3 chunks with cosine similarity score ≥ 0.4 (below threshold = not included)
- **Auto-reply mode:** injected automatically into system prompt
- **Assistant mode:** accessible via `search_knowledge` tool only

**Dual storage clarification:** ChromaDB holds vectors + chunk text. SQLite `knowledge_docs` table holds metadata (filename, chunk index, hash of chunk text) to detect when files change and need re-indexing. They are kept in sync: when a document is re-indexed, the ChromaDB collection entry and the SQLite metadata row are both updated. ChromaDB is the query-time source of truth; SQLite tracks index state.

### 7. Persona Engine (`knowledge/persona_engine.py`)

Loads `persona.yaml` and produces a structured system prompt fragment.

**Trigger matching:** Uses `re.search(pattern, text, re.IGNORECASE)` where `pattern` is the exact string from `trigger`. Pipe characters are treated as regex alternation (e.g., `pricing|cost|rate` matches any of the three words). Matching is case-insensitive and substring (not whole-word). Multiple triggers can fire for a single message — all matching `response_hint` values are appended to the prompt.

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

### 8. Storage (`storage/db.py`)

SQLite via SQLAlchemy ORM. Three tables:

```sql
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY,
    channel     TEXT NOT NULL,
    chat_id     TEXT NOT NULL,
    message_id  TEXT,                    -- platform message ID (nullable for assistant responses)
    sender_id   TEXT NOT NULL,
    role        TEXT NOT NULL,           -- "user" | "assistant"
    content     TEXT NOT NULL,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Deduplication index: ensures each platform message is processed once
CREATE UNIQUE INDEX idx_dedup ON conversations(channel, message_id)
    WHERE message_id IS NOT NULL;

CREATE TABLE knowledge_docs (
    id              INTEGER PRIMARY KEY,
    filename        TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    chunk_hash      TEXT NOT NULL,       -- SHA256 of chunk_text (for change detection)
    indexed_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE rate_limits (
    channel     TEXT NOT NULL,
    sender_id   TEXT NOT NULL,
    last_seen   DATETIME NOT NULL,
    PRIMARY KEY (channel, sender_id)
);
```

**Cloud swap:** Change `database.url` in `settings.yaml` to a PostgreSQL DSN — SQLAlchemy handles the rest.

---

## Channel Adapters

### WhatsApp (`adapters/whatsapp.py`) — Bridge Mode

For personal WhatsApp accounts (no Meta Business account needed), the adapter communicates with a local `whatsapp-web.js` Node.js bridge via HTTP.

**Bridge HTTP API contract:**

The bridge runs on `http://localhost:3000` (configurable). The Python adapter:

1. **Receiving messages** — registers a webhook by polling `GET /messages/pending` every 2 seconds OR the bridge POSTs to `http://localhost:8000/whatsapp/webhook` (configurable). Payload:
```json
{
  "message_id": "MSGID_123",
  "from": "+91XXXXXXXXXX",
  "from_name": "John",
  "chat_id": "91XXXXXXXXXX@c.us",
  "body": "Hello",
  "timestamp": 1711500000,
  "is_group": false
}
```

2. **Sending messages** — `POST /send`:
```json
{
  "chat_id": "91XXXXXXXXXX@c.us",
  "message": "Hello back!"
}
```
Response: `{"status": "sent", "message_id": "MSGID_124"}` or `{"status": "error", "reason": "..."}`

**Session management:** The bridge handles QR code authentication. On first run, the bridge prints the QR code to its terminal. After scanning, the session is persisted to `bridge/session/` and reused on subsequent starts. The Python adapter does NOT manage the QR flow — it only connects after the bridge is ready.

**Bridge lifecycle:** `whatsapp.py` starts the bridge as a subprocess on `adapter.connect()`:
```python
async def connect(self):
    self._proc = await asyncio.create_subprocess_exec(
        "node", "bridge/index.js",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # Wait for bridge to emit "WhatsApp ready" on stdout (max 30s)
    await self._wait_for_ready(timeout=30)
    # If timeout: raise RuntimeError("WhatsApp bridge failed to start")
```

**Startup failure:** If the bridge fails to start within 30 seconds, `connect()` raises `RuntimeError`. `main.py` catches this and logs a warning — the system starts with Telegram-only (graceful degradation, not a hard crash).

**Note:** Using `whatsapp-web.js` for automation violates WhatsApp's Terms of Service. For personal/development use this is acceptable, but the risk of account suspension exists if message volumes are high. `pywa` is NOT used in bridge mode and is excluded from `requirements.txt` for personal deployments.

### Telegram (`adapters/telegram.py`)

- **Library:** `python-telegram-bot` v20+ (native async)
- **Auth:** Bot token from @BotFather, stored in `settings.yaml` / env var
- **Group chat handling:** Bot ignores all group messages unless `is_owner=True` (owner can use assistant mode from a group)
- **Most reliable channel:** Official API, generous rate limits, no ToS risk

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
  mode: "bridge"              # only "bridge" supported in v1
  bridge_url: "http://localhost:3000"
  bridge_script: "bridge/index.js"
  polling_interval_seconds: 2  # how often to poll GET /messages/pending

database:
  url: "sqlite:///vedaai.db"  # swap to postgresql://... for cloud

rag:
  documents_path: "knowledge/documents"
  chroma_path: ".chroma"
  top_k: 3
  min_similarity: 0.4         # chunks below this score are not injected

conversation:
  history_window: 20          # per-chat message pairs to include in context

rate_limiting:
  auto_reply_cooldown_seconds: 10  # min seconds between responses to same sender
```

All `${VAR}` values are loaded from environment variables at startup.

---

## Data Flow (End-to-End)

### Incoming message from someone else (auto-reply):

1. Adapter receives raw message, normalises to `RawMessage`
2. `Gateway.handle(raw)` called
3. `ConversationManager.is_duplicate()` check — if seen before, return (handles webhook replay)
4. `UserDetector.resolve(raw)` → `Message` with `is_owner=False`
5. Group check: if `is_group=True` and not owner, return
6. Rate limit check: if sender throttled, return
7. Acquire per-chat lock (prevents race conditions for same chat_id)
8. `agent.auto_reply_mode(message)` called
9. Agent fetches last 20 messages for this `(channel, chat_id)` from `ConversationManager`
10. Agent calls `rag_engine.search(message.text)` → top-3 chunks above similarity threshold
11. Agent builds system prompt: persona + RAG context + history
12. Claude generates response
13. `ConversationManager.save(message, role="user")` — saves incoming message
14. `ConversationManager.save_response(message, response, role="assistant")` — saves response
15. `adapter.send_message(message.chat_id, response)`
16. Release lock

### Owner messages bot (assistant mode):

1-7. Same as above (dedup, resolve, group check, rate limit, lock)
8. `agent.assistant_mode(message)` called
9. Agent fetches last 20 messages for this `(channel, chat_id)` — per-chat, not global
10. Claude has tool access; decides whether to call `search_knowledge` based on the question
11. If tool called: RAG search executed, result injected as tool response
12. Claude generates final response
13. Both incoming message AND response saved to `ConversationManager` (same as auto-reply)
14-16. Same as above

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Claude API error | Retry once with 2s delay. On second failure: send "I'm unavailable right now, please try again shortly." |
| Adapter runtime disconnect | Log error, attempt reconnect with exponential backoff: 5s, 10s, 20s, 40s, 80s. After 5 failures: log critical error, continue running other adapters |
| WhatsApp bridge startup failure | Log warning, start without WhatsApp. Telegram continues normally |
| RAG retrieval failure | Fall back to persona-only response (no RAG context injected) |
| No RAG results above threshold | Use persona-only response. Do not inject low-confidence content |
| Unknown message type (media, sticker, etc.) | Respond: "I can only process text messages currently." Save to history |
| Missing adapter for channel | `RuntimeError` — this is a programming error, not a runtime condition; crash loudly |
| Owner IDs missing at startup | `ValueError` — refuse to start with clear message |
| Webhook replay / duplicate message | `is_duplicate()` check at gateway entry — silently skip |
| Rate-limited sender | Silently drop — no response sent, no error |
| Group message from non-owner | Silently ignore |

---

## Testing Strategy

- **Unit tests:** Each component in isolation (agent, RAG, persona engine, user detector, conversation manager)
- **Adapter tests:** Mock `python-telegram-bot` and bridge HTTP server (no real accounts needed)
- **Integration test:** Full message flow — raw adapter input → gateway → agent → response
- **Deduplication test:** Send same `message_id` twice, verify only one response
- **Race condition test:** Send two messages to same `chat_id` concurrently, verify sequential processing
- **Manual testing:** Real Telegram bot for live verification during development

---

## Deployment Path

```
Phase 1 (now):   python main.py  →  runs locally on Mac
Phase 2 (later): Dockerfile + docker-compose (bridge + Python in one compose file)
Phase 3 (cloud): Railway / Render / EC2 — change DB URL + env vars only
```

All secrets via environment variables, never hardcoded.

---

## Dependencies

```
anthropic>=0.50.0              # Claude Agent SDK (claude-sonnet-4-6)
python-telegram-bot>=20.0,<21  # Telegram adapter
chromadb>=0.5.0,<1.0.0         # Vector store (0.5+ API required)
sentence-transformers>=2.2.0   # Local embeddings
sqlalchemy>=2.0                # ORM
alembic>=1.13.0                # DB migrations
pyyaml>=6.0                    # Config loading
python-dotenv>=1.0.0           # .env support
aiohttp>=3.9.0                 # HTTP client for WhatsApp bridge
pytest>=8.0                    # Testing
pytest-asyncio>=0.23.0         # Async test support
```

---

## Future Enhancements (v2+)

- Instagram DM adapter
- LinkedIn message adapter
- Reminder/scheduling tool
- Web dashboard for conversation review
- Voice message transcription (Whisper)
- Multi-language persona support
- WhatsApp Cloud API mode (for production, replaces bridge)
