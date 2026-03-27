# VedaAI

**Personal AI assistant that auto-replies on WhatsApp & Telegram using Claude + RAG + persona rules, and acts as your personal assistant when you message it directly.**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude-Sonnet_4-orange?logo=anthropic&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-84_passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## How It Works

VedaAI runs as a single Python process. Messages from WhatsApp and Telegram flow through a central gateway that decides how to respond:

- **Stranger messages you** &rarr; Auto-reply using your persona + RAG knowledge base
- **You message the bot** &rarr; Full AI assistant with tools (search knowledge, conversation history)
- **Group chats** &rarr; Only responds when the owner messages (ignores strangers in groups)

```
WhatsApp Adapter ──┐
                   ├──▶ Gateway ──▶ UserDetector
Telegram Adapter ──┘                    │
                              ┌─────────┴─────────┐
                         Stranger              Owner
                              │                    │
                       Auto-Reply Mode      Assistant Mode
                       (Persona + RAG)      (Tools + RAG)
                              │                    │
                              └─────────┬──────────┘
                                   Claude API
                                        │
                                  ┌─────┴─────┐
                               SQLite      ChromaDB
```

## Features

- **Hybrid mode** &mdash; auto-replies to others on your behalf, personal assistant when you message it
- **RAG knowledge base** &mdash; ChromaDB + sentence-transformers index your documents for context-aware replies
- **Persona engine** &mdash; YAML-configured personality, tone, boundaries, and trigger rules
- **Per-chat concurrency** &mdash; asyncio locks prevent race conditions in parallel conversations
- **Rate limiting** &mdash; per-sender cooldown prevents spam responses
- **Message deduplication** &mdash; handles webhook replays and double-delivery
- **Graceful degradation** &mdash; WhatsApp bridge failure doesn't crash Telegram
- **84 automated tests** &mdash; built entirely with TDD, including 3 integration tests

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI | Anthropic Claude API (AsyncAnthropic) |
| RAG | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| WhatsApp | whatsapp-web.js Node.js bridge + aiohttp polling |
| Telegram | python-telegram-bot v20 (native async) |
| Database | SQLAlchemy 2.x + SQLite |
| Persona | YAML config with regex trigger matching |
| Testing | pytest-asyncio, 84 tests, TDD methodology |
| Runtime | Python asyncio single-process gateway |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Prasannavattikoda-AI-ML/vedaai.git
cd vedaai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and TELEGRAM_BOT_TOKEN
# Edit config/settings.yaml with your Telegram user ID

# Add knowledge documents
echo "Your bio or notes here" > knowledge/documents/bio.txt

# Run
python main.py
```

See [`docs/SMOKE_TEST.md`](docs/SMOKE_TEST.md) for the full setup walkthrough.

## Project Structure

```
vedaai/
├── main.py                    # Entry point — wires all components
├── config/
│   ├── settings.yaml          # API keys, ports, adapter configs
│   └── persona.yaml           # Persona name, tone, boundaries, triggers
├── core/
│   ├── gateway.py             # Central message router with per-chat locking
│   ├── agent.py               # Claude API wrapper — auto-reply & assistant modes
│   ├── user_detector.py       # Owner vs. stranger resolution
│   ├── conversation.py        # Per-chat history management
│   ├── rate_limiter.py        # Per-sender cooldown
│   └── config.py              # YAML loader with ENV var substitution
├── adapters/
│   ├── base.py                # BaseAdapter ABC + RawMessage/Message dataclasses
│   ├── telegram.py            # Telegram adapter (python-telegram-bot v20)
│   └── whatsapp.py            # WhatsApp adapter (Node.js bridge + polling)
├── bridge/
│   ├── index.js               # whatsapp-web.js HTTP bridge server
│   └── package.json           # Node.js dependencies
├── knowledge/
│   ├── rag_engine.py          # ChromaDB + sentence-transformers RAG
│   ├── persona_engine.py      # YAML persona loader with regex triggers
│   └── documents/             # Your files for RAG indexing
├── storage/
│   └── db.py                  # SQLAlchemy ORM models + session factory
└── tests/                     # 84 tests — unit + integration
```

## Architecture Decisions

- **AsyncAnthropic** (not sync `Anthropic`) &mdash; sync client blocks the event loop in asyncio apps
- **Per-chat locks** via `defaultdict(lambda: asyncio.Lock())` &mdash; prevents race conditions
- **RAG auto-injected** in auto-reply mode, **tool-accessible** in assistant mode
- **Adapter pattern** &mdash; adding a new channel requires one new file implementing `BaseAdapter`
- **WhatsApp bridge** as subprocess &mdash; isolates Node.js from Python's event loop

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

---

Built by [Prasanna Vattikunda](https://github.com/Prasannavattikoda-AI-ML)
