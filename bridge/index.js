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
        from_name: msg.notifyName || msg.from,
        chat_id: msg.from,
        body: msg.body,
        timestamp: msg.timestamp,
        is_group: chat.isGroup,
    });
});

// Python adapter polls this to drain the queue
app.get('/messages/pending', (req, res) => {
    const msgs = [...pendingMessages];
    pendingMessages.length = 0;
    res.json(msgs);
});

// Python adapter sends messages via this
app.post('/send', async (req, res) => {
    const { chat_id, message } = req.body;
    if (!chat_id || !message) {
        return res.status(400).json({ status: 'error', reason: 'chat_id and message are required' });
    }
    try {
        const result = await client.sendMessage(chat_id, message);
        res.json({ status: 'sent', message_id: result.id._serialized });
    } catch (err) {
        res.status(500).json({ status: 'error', reason: err.message });
    }
});

app.listen(PORT, () => {
    console.log(`Bridge listening on port ${PORT}`);
});

client.initialize();
