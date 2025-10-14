"""Telegram notification callback for PyTorch Lightning."""

import os
from pathlib import Path
import requests
from lightning.pytorch.callbacks import Callback


class TelegramNotificationCallback(Callback):
    """Send Telegram notifications after each epoch.

    Setup:
    1. Create a Telegram bot: Message @BotFather on Telegram -> /newbot
    2. Get your chat ID: Message @userinfobot on Telegram
    3. Add to .env file in project root:
       TELEGRAM_BOT_TOKEN=your_bot_token
       TELEGRAM_CHAT_ID=your_chat_id
    """

    def __init__(self, send_on_epoch_end=True, send_on_train_end=True):
        super().__init__()

        # Load from .env file if exists
        self._load_env_file()

        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        self.send_on_epoch_end = send_on_epoch_end
        self.send_on_train_end = send_on_train_end

        if not self.bot_token or not self.chat_id:
            print("⚠️  Telegram not configured. Add to .env file or set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            self.enabled = False
        else:
            self.enabled = True
            print(f"✅ Telegram notifications enabled (Chat ID: {self.chat_id})")

    def _load_env_file(self):
        """Load environment variables from .env file."""
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value

    def send_message(self, message):
        """Send message to Telegram."""
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        try:
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                print("📱 Telegram notification sent")
            else:
                print(f"⚠️  Telegram error: {response.text}")
        except Exception as e:
            print(f"⚠️  Failed to send Telegram: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when training epoch ends."""
        if not self.send_on_epoch_end:
            return

        epoch = trainer.current_epoch + 1

        # Get metrics
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch", 0.0)
        train_acc = metrics.get("train_acc", 0.0)

        # Get contrastive metrics if available
        sup_loss = metrics.get("train_sup_loss_epoch", None)
        contrast_loss = metrics.get("train_contrast_loss_epoch", None)

        # Get hangman win rate if available
        win_rate = metrics.get("hangman_win_rate", None)

        # Build message
        msg = f"📊 *Epoch {epoch} Complete*\n\n"
        msg += f"📉 Loss: {train_loss:.4f}\n"
        msg += f"📈 Accuracy: {train_acc:.4f}\n"

        if sup_loss is not None:
            msg += f"🔵 Supervised Loss: {sup_loss:.4f}\n"
        if contrast_loss is not None:
            msg += f"🔴 Contrastive Loss: {contrast_loss:.4f}\n"
        if win_rate is not None:
            msg += f"🎯 Win Rate: {win_rate:.2%}\n"

        self.send_message(msg)

    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        if not self.send_on_train_end:
            return

        total_epochs = trainer.current_epoch + 1

        msg = f"🎉 *Training Complete!*\n\n"
        msg += f"✅ Completed {total_epochs} epochs\n"
        msg += f"📁 Check: logs/checkpoints/"

        self.send_message(msg)