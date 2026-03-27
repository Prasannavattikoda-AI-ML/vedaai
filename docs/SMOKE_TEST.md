# Manual Telegram Smoke Test

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
