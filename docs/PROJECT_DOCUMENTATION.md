# VedaAI — Project Documentation

## 1. Project Overview

**VedaAI** is a personal AI assistant that connects to WhatsApp and Telegram, operating in hybrid mode:
- **Auto-reply mode** (strangers): Responds using persona rules + RAG knowledge base
- **Assistant mode** (owner): Full Claude AI with tools to search knowledge and retrieve conversation history

The system runs as a single Python asyncio process using a monolithic gateway pattern.

---

## 2. System Architecture

```
                    ┌─────────────────┐   ┌─────────────────┐
                    │  WhatsApp Adapter│   │ Telegram Adapter │
                    │  (Node.js Bridge)│   │(python-telegram) │
                    └────────┬────────┘   └────────┬────────┘
                             │                      │
                             └──────────┬───────────┘
                                        │
                              ┌─────────▼─────────┐
                              │      Gateway       │
                              │  Dedup + RateLimit │
                              │  + Per-chat Lock   │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼──────────┐
                              │    UserDetector     │
                              │  Owner vs Stranger  │
                              └────┬──────────┬────┘
                                   │          │
                          Stranger │          │ Owner
                                   │          │
                        ┌──────────▼──┐  ┌────▼──────────┐
                        │ Auto-Reply  │  │   Assistant    │
                        │ Persona+RAG │  │ Tools + Claude │
                        └──────┬──────┘  └────┬──────────┘
                               │               │
                               └───────┬───────┘
                                       │
                             ┌─────────▼──────────┐
                             │    Claude API       │
                             │  AsyncAnthropic     │
                             │  claude-sonnet-4-6  │
                             └─────────┬──────────┘
                                       │
                            ┌──────────┴──────────┐
                            │                     │
                   ┌────────▼────────┐  ┌─────────▼────────┐
                   │     SQLite      │  │     ChromaDB     │
                   │  SQLAlchemy 2.x │  │ sentence-         │
                   │  Conversations  │  │ transformers      │
                   │  Rate Limits    │  │ RAG Documents     │
                   └─────────────────┘  └──────────────────┘
```

---

## 3. Component Details

### 3.1 Adapters (`adapters/`)

| Component | File | Purpose |
|-----------|------|---------|
| BaseAdapter | `base.py` | ABC defining `connect()`, `disconnect()`, `send_message()`, `on_message()` |
| RawMessage | `base.py` | Dataclass — adapter-produced message before owner resolution |
| Message | `base.py` | Dataclass — enriched message with `is_owner` flag |
| TelegramAdapter | `telegram.py` | python-telegram-bot v20, native async, group detection |
| WhatsAppAdapter | `whatsapp.py` | Manages Node.js bridge subprocess, HTTP polling |

**Key design decisions:**
- `channel` is an `@property @abstractmethod` — enforced at class level
- `is_owner` is NOT on adapters — owner detection centralized in `UserDetector`
- WhatsApp uses polling (`GET /messages/pending`), not webhooks, for v1 simplicity
- Graceful degradation: WhatsApp bridge failure doesn't crash the system

### 3.2 Gateway (`core/gateway.py`)

The central message router. Receives `RawMessage` from adapters, processes through the pipeline:

1. **Deduplication** — `ConversationManager.is_duplicate()` checks if `message_id` was already seen
2. **Owner Resolution** — `UserDetector.resolve()` enriches with `is_owner` flag
3. **Group Policy** — ignores non-owner messages in group chats
4. **Rate Limiting** — per-sender cooldown (10s default) for auto-reply mode only
5. **Per-chat Locking** — `defaultdict(lambda: asyncio.Lock())` prevents race conditions
6. **Agent Dispatch** — routes to `auto_reply_mode()` or `assistant_mode()`
7. **Adapter Lookup** — verifies adapter exists BEFORE saving conversation
8. **Persistence** — saves both user message and assistant response
9. **Delivery** — sends response via the correct adapter

### 3.3 Claude Agent (`core/agent.py`)

Two modes of operation:

**Auto-Reply Mode** (strangers):
- RAG context auto-injected into system prompt
- Persona rules define tone and boundaries
- No tools — Claude responds directly
- Retry once on API failure, then return fallback message

**Assistant Mode** (owner):
- Tools available: `search_knowledge`, `get_conversation_history`
- RAG NOT auto-injected (accessible via tool only)
- Full tool loop: Claude calls tools, agent executes, returns results
- Tool loop wrapped in try/except — returns fallback on any failure

**Critical implementation detail:** Uses `anthropic.AsyncAnthropic` (not sync `Anthropic`) — the sync client blocks the event loop in asyncio applications.

### 3.4 Knowledge Layer (`knowledge/`)

**PersonaEngine** (`persona_engine.py`):
- Loads `config/persona.yaml` with name, tone, language, boundaries, trigger rules
- Trigger matching uses `re.search(rule["trigger"], text, re.IGNORECASE)`
- `name` is a `@property` backed by `self._name`

**RAGEngine** (`rag_engine.py`):
- ChromaDB with `metadata={"hnsw:space": "cosine"}` for proper similarity scoring
- `sentence-transformers` (all-MiniLM-L6-v2) for embeddings
- `index_documents()` is async — delegates to `asyncio.to_thread` to avoid blocking
- `search()` also uses `asyncio.to_thread` for the encode + query operations
- Chunk IDs use `file_path.relative_to(docs_path)::chunk_index` to prevent collisions
- Skip-unchanged optimization via SHA-256 content hashing
- Similarity formula: `1.0 - distance / 2.0` (cosine distance range [0, 2])

### 3.5 Storage (`storage/db.py`)

SQLAlchemy 2.x ORM with three models:
- `ConversationRow` — chat history with `UniqueConstraint(channel, message_id)`
- `KnowledgeDocRow` — document metadata for RAG indexing
- `RateLimitRow` — per-sender last-seen timestamps

Session management uses `@contextmanager`:
```python
@contextmanager
def session(self):
    s = self._Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
```

### 3.6 WhatsApp Bridge (`bridge/`)

Node.js Express server wrapping `whatsapp-web.js`:
- `GET /messages/pending` — drains and returns message queue
- `POST /send` — sends message, returns `{status, message_id}` or error
- `LocalAuth` for session persistence (QR scan required only on first run)
- Input validation on `/send` endpoint
- Started as a subprocess by `WhatsAppAdapter.connect()`

---

## 4. Configuration

### `config/settings.yaml`
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

All `${VAR}` values are resolved from environment variables at startup.

### `config/persona.yaml`
```yaml
name: "Pandu"
tone: "friendly, professional"
language: "English"
boundaries:
  - "Do not share personal phone numbers"
  - "Do not make commitments on behalf of the owner"
rules:
  - trigger: "available|free|schedule"
    response_hint: "Check availability and respond politely"
  - trigger: "who are you|what do you do"
    response_hint: "Introduce as Pandu's AI assistant"
```

---

## 5. Testing Strategy

**Methodology:** Test-Driven Development (TDD) — tests written before implementation for every component.

**Test count:** 84 tests (81 unit + 3 integration)

| Test File | Count | What it covers |
|-----------|-------|---------------|
| `test_user_detector.py` | 5 | Owner detection, validation, field copying |
| `test_conversation.py` | 7 | History read/write, dedup, None message_id |
| `test_rate_limiter.py` | 3 | Throttling, cooldown expiry, sender isolation |
| `test_gateway.py` | 15 | Full routing, group policy, rate limiting, concurrency |
| `test_agent.py` | 12 | Auto-reply, assistant mode, tool loop, retry, fallback |
| `test_rag.py` | 12 | Indexing, search, similarity threshold, skip-unchanged |
| `test_persona.py` | 7 | YAML loading, prompt building, trigger matching |
| `test_adapters.py` | 17 | Telegram + WhatsApp adapters, connect, disconnect, poll |
| `test_config.py` | 4 | ENV substitution, missing file, invalid YAML |
| `test_integration.py` | 3 | Full auto-reply flow, owner mode, dedup end-to-end |

**Key testing patterns:**
- `AsyncMock` for all async interfaces
- In-memory SQLite (`sqlite:///:memory:`) for database tests
- `tmp_path` fixture for file-dependent tests (persona, RAG)
- `asyncio.sleep(0)` in mocks to ensure event loop yields for timeout tests

---

## 6. Error Handling & Edge Cases

| Scenario | Handling |
|----------|----------|
| WhatsApp bridge fails to start | `main.py` catches exception, continues with Telegram only |
| Bridge process exits unexpectedly | `_wait_for_ready` raises `RuntimeError` |
| Claude API timeout | Retry once after 2s sleep, then return `FALLBACK_MESSAGE` |
| Tool loop exception | Caught inside `_run_tool_loop`, returns `FALLBACK_MESSAGE` |
| Duplicate message (webhook replay) | `is_duplicate()` check at gateway entry |
| `message_id` is None | `is_duplicate` returns `False` (no false positives) |
| Callback throws mid-batch (WhatsApp) | Per-message try/except, remaining messages still dispatched |
| Rate-limited sender | Silently dropped (no response sent) |
| Group chat from stranger | Silently dropped (only owner gets responses in groups) |
| Adapter missing for channel | Check happens before saving conversation |
| Empty YAML config file | `raw or {}` prevents None propagation |
| SQLite naive vs aware timestamps | Conditional `replace(tzinfo=UTC)` vs `astimezone(UTC)` |

---

## 7. Development Process

**Methodology:** Subagent-Driven Development with two-stage reviews

1. **Spec phase** — Collaborative brainstorming, design document, spec review loop
2. **Plan phase** — 17-task TDD implementation plan with file map
3. **Build phase** — Fresh subagent per task, tests written first
4. **Review phase** — Two-stage review per task:
   - Stage 1: Spec compliance (does it match the design?)
   - Stage 2: Code quality (bugs, edge cases, error handling)
5. **Fix phase** — Issues from reviews fixed and re-verified
6. **Final review** — Full codebase review across all modules

**Key metrics:**
- 27 commits across the implementation
- 25+ issues caught in spec review before any code was written
- 30+ issues caught in two-stage code reviews across all tasks
- 0 test failures in final suite

---

## 8. Future Roadmap (v2)

- [ ] Instagram and LinkedIn channel adapters
- [ ] Alembic database migrations
- [ ] Web dashboard for monitoring conversations
- [ ] Cloud deployment (Docker + cloud provider)
- [ ] Webhook mode for WhatsApp (replace polling)
- [ ] Multi-user support
- [ ] Voice message transcription
- [ ] Scheduled message sending
