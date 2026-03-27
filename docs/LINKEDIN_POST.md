# LinkedIn Post Content for VedaAI

> **Attach image:** `docs/assets/vedaai-architecture-linkedin.png`

---

## Post Option 1 (Recommended — Technical + Story)

I built an AI assistant that answers my WhatsApp and Telegram messages when I'm busy.

Not a chatbot template. Not a wrapper around an API. A real system:

- Strangers message me? VedaAI auto-replies using my persona + a RAG knowledge base
- I message the bot? Full AI assistant with tools to search my knowledge base
- Per-chat locking so parallel conversations don't race
- Rate limiting so nobody gets spammed
- Message dedup so webhook replays don't cause double-replies
- Graceful degradation: if WhatsApp fails, Telegram keeps running

The architecture: a single Python asyncio process with a Gateway pattern routing messages through adapters, UserDetector, and Claude's API.

Built it test-first: 84 automated tests before any production code. Two-stage code reviews on every task. 15 modules across 27 commits.

Tech stack:
- Anthropic Claude API (AsyncAnthropic)
- ChromaDB + sentence-transformers for RAG
- python-telegram-bot v20
- whatsapp-web.js Node.js bridge
- SQLAlchemy 2.x + SQLite
- pytest-asyncio

The hardest part wasn't the AI. It was the systems engineering: making async adapters play nice, handling every edge case (what if the bridge crashes? what if message_id is null? what if the Claude API times out mid-tool-loop?), and keeping everything testable.

Open source: https://github.com/Prasannavattikoda-AI-ML/vedaai

#AI #Python #AsyncIO #Claude #RAG #WhatsApp #Telegram #SystemsEngineering #OpenSource #TDD

---

## Post Option 2 (Shorter — Impact-focused)

What if your AI assistant could answer messages on your behalf — with your personality, your knowledge, and your boundaries?

That's VedaAI.

It connects to WhatsApp and Telegram, auto-replies to strangers using RAG + persona rules, and switches to full assistant mode when I message it directly.

Under the hood:
- Python asyncio gateway with per-chat locking
- Claude API for intelligence
- ChromaDB for knowledge retrieval
- 84 tests, built test-first
- Modular adapters — adding a new channel = one file

The whole system runs as a single process. If WhatsApp crashes, Telegram keeps going. If Claude times out, it retries once then sends a graceful fallback.

This is the project that taught me: the AI part is 20% of the work. The other 80% is error handling, concurrency, and testing.

Code: https://github.com/Prasannavattikoda-AI-ML/vedaai

#AI #Python #Claude #RAG #OpenSource #SoftwareEngineering

---

## Post Option 3 (Short + Punchy — Maximum engagement)

I'm a 2nd-year IT student and I just built a production-grade AI assistant from scratch.

VedaAI auto-replies on WhatsApp & Telegram using Claude + RAG. 84 tests. 15 modules. 27 commits. Full async architecture.

Not a tutorial project. A real system with rate limiting, message dedup, per-chat concurrency locks, and graceful degradation.

The architecture diagram tells the story better than I can. (see image)

Repo: https://github.com/Prasannavattikoda-AI-ML/vedaai

What should I build next?

#AI #BuildInPublic #Python #Claude #RAG #OpenSource

---

## Hashtag Strategy

**Always include:** #AI #Python #Claude #RAG #OpenSource
**For reach:** #BuildInPublic #SoftwareEngineering #TDD
**For recruiters:** #InternshipReady #SystemsDesign #AsyncProgramming

## Posting Tips

1. **Attach the image** — LinkedIn posts with images get 2x more engagement
2. **Post Tuesday-Thursday, 8-10am IST** — peak LinkedIn engagement window
3. **Reply to every comment** within the first 2 hours — the algorithm rewards it
4. **Tag relevant people** — if you know anyone at Anthropic or in the AI space, tag them
5. **First comment trick** — post the GitHub link as the first comment (some say links in the main post reduce reach)
