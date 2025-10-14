# Telegram Notification Setup Guide

Get training updates on your phone after each epoch!

## Quick Setup (5 minutes)

### Step 1: Create Your Telegram Bot

1. Open Telegram on your phone
2. Search for: `@BotFather`
3. Start a chat and send: `/newbot`
4. Choose a name for your bot (e.g., "My Training Bot")
5. Choose a username (e.g., "my_training_bot")
6. **Save the token** you receive (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Get Your Chat ID

1. Search for: `@userinfobot` on Telegram
2. Send: `/start`
3. **Save your Chat ID** (a number like: `987654321`)

### Step 3: Set Environment Variables

Open terminal and run:

```bash
export TELEGRAM_BOT_TOKEN="paste_your_bot_token_here"
export TELEGRAM_CHAT_ID="paste_your_chat_id_here"
```

**To make it permanent** (add to ~/.bashrc):
```bash
echo 'export TELEGRAM_BOT_TOKEN="your_token"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="your_chat_id"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Install requests library

```bash
conda activate orchestra
pip install requests
```

### Step 5: Run Training

```bash
python main.py --max-epochs 10 --batch-size 256 --use-contrastive --lambda-contrast 0.1
```

## What You'll Receive

After each epoch, you'll get a message like:

```
📊 Epoch 1 Complete

📉 Loss: 2.3456
📈 Accuracy: 0.7234
🔵 Supervised Loss: 4.5678
🔴 Contrastive Loss: 0.1234
🎯 Win Rate: 63.45%
```

When training finishes:
```
🎉 Training Complete!

✅ Completed 10 epochs
📁 Check: logs/checkpoints/
```

## Troubleshooting

If notifications don't work:
1. Check environment variables are set: `echo $TELEGRAM_BOT_TOKEN`
2. Make sure you started a chat with your bot
3. Check terminal for error messages
4. Verify requests is installed: `pip list | grep requests`

## Privacy Note

- Your bot token is private - don't share it
- Only you can message your bot
- Messages are sent through Telegram's API (encrypted)
