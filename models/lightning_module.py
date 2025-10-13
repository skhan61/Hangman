"""PyTorch Lightning module wrapping Hangman architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import logging

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from models.architectures import BaseArchitecture
from models.metrics import MaskedAccuracy


logger = logging.getLogger(__name__)


@dataclass
class TrainingModuleConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class HangmanLightningModule(LightningModule):
    def __init__(
        self,
        model: BaseArchitecture,
        training_config: Optional[TrainingModuleConfig] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.model = self.model.to(self.device)
        self.training_config = training_config or TrainingModuleConfig()
        self.train_accuracy = MaskedAccuracy()
        # self.val_accuracy = MaskedAccuracy()

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.model(inputs, lengths)

    def _compute_loss_and_metrics(self, batch):
        # Batch tensors should already be on the correct device
        inputs = batch["inputs"]
        lengths = batch["lengths"]
        labels = batch["labels"]
        mask = batch["label_mask"]

        logger.debug(
            "Batch tensors — inputs=%s, lengths=%s, labels=%s, mask=%s",
            tuple(inputs.shape),
            tuple(lengths.shape),
            tuple(labels.shape),
            tuple(mask.shape),
        )

        logits = self(inputs, lengths)
        log_probs = F.log_softmax(logits, dim=-1)

        per_position_loss = -(labels * log_probs).sum(dim=-1)
        masked_loss = per_position_loss * mask

        total_mask = mask.sum()
        loss = masked_loss.sum() / total_mask.clamp_min(1.0)

        logger.debug(
            "Loss stats — masked_loss_sum=%s, total_mask=%s, loss=%s",
            masked_loss.sum().detach().cpu(),
            total_mask.detach().cpu(),
            loss.detach().cpu(),
        )

        # redundant
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            targets = labels.argmax(dim=-1)
            correct = ((preds == targets).float() * mask).sum()
            accuracy = correct / total_mask.clamp_min(1.0)

        logger.debug(
            "Accuracy stats — correct=%s, total_mask=%s, accuracy=%s",
            correct.detach().cpu(),
            total_mask.detach().cpu(),
            accuracy.detach().cpu(),
        )

        return loss, accuracy, logits

    def training_step(self, batch, batch_idx):
        # Move batch to device first
        device = self.device
        # print(device)
        batch_device = {
            "inputs": batch["inputs"].to(device),
            "labels": batch["labels"].to(device),
            "label_mask": batch["label_mask"].to(device),
            "lengths": batch["lengths"].to(device),
            # "words": batch["words"],
        }

        loss, batch_acc, logits = self._compute_loss_and_metrics(batch_device)
        self.train_accuracy.update(logits, batch_device["labels"], batch_device["label_mask"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_batch_acc", batch_acc, on_step=True, on_epoch=False, prog_bar=False
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        # Add learning rate scheduler
        # Reduces LR by half when win rate plateaus for 3 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',              # Maximize hangman_win_rate
            factor=0.5,              # Reduce LR to 50% when plateau
            patience=3,              # Wait 3 epochs before reducing LR
            min_lr=1e-6,             # Don't reduce below 0.000001
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "hangman_win_rate",  # Monitor win rate from callback
                "interval": "epoch",             # Check every epoch
                "frequency": 1,                  # Check every 1 epoch
            },
        }
