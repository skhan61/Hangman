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
    # Contrastive learning parameters
    use_contrastive: bool = False
    lambda_contrast: float = 0.1
    temperature: float = 0.07


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

        # Initialize contrastive loss if enabled
        if self.training_config.use_contrastive:
            from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss

            base_loss = NTXentLoss(temperature=self.training_config.temperature)
            self.contrastive_loss_fn = SelfSupervisedLoss(base_loss)
            logger.info(
                "Contrastive learning enabled with lambda=%.3f, temperature=%.3f",
                self.training_config.lambda_contrast,
                self.training_config.temperature,
            )
        else:
            self.contrastive_loss_fn = None

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.model(inputs, lengths, return_embeddings=return_embeddings)

    def _compute_supervised_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute supervised cross-entropy loss from logits.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: One-hot encoded labels [batch_size, seq_len, vocab_size]
            mask: Binary mask for valid positions [batch_size, seq_len]

        Returns:
            Tuple of (loss, total_mask_count)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        per_position_loss = -(labels * log_probs).sum(dim=-1)
        masked_loss = per_position_loss * mask
        total_mask = mask.sum()
        loss = masked_loss.sum() / total_mask.clamp_min(1.0)
        return loss, total_mask

    def training_step(self, batch, batch_idx):
        # Move batch to device first
        device = self.device
        batch_device = {
            "inputs": batch["inputs"].to(device),
            "labels": batch["labels"].to(device),
            "label_mask": batch["label_mask"].to(device),
            "lengths": batch["lengths"].to(device),
        }

        inputs = batch_device["inputs"]
        lengths = batch_device["lengths"]
        labels = batch_device["labels"]
        mask = batch_device["label_mask"]

        # Standard supervised loss computation
        if not self.training_config.use_contrastive:
            # Single forward pass without contrastive learning
            logits = self(inputs, lengths)
            loss, total_mask = self._compute_supervised_loss(logits, labels, mask)

            # Update metrics
            self.train_accuracy.update(logits, labels, mask)

            # Compute batch accuracy for logging
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                targets = labels.argmax(dim=-1)
                correct = ((preds == targets).float() * mask).sum()
                batch_acc = correct / total_mask.clamp_min(1.0)

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(
                "train_acc",
                self.train_accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "train_batch_acc",
                batch_acc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            return loss

        # Contrastive learning: dual forward passes with different dropout masks
        self.model.train()  # Ensure dropout is active

        # ========== View 1: First forward pass ==========
        logits1, embeddings1 = self(inputs, lengths, return_embeddings=True)

        # ========== View 2: Second forward pass (different dropout) ==========
        logits2, embeddings2 = self(inputs, lengths, return_embeddings=True)

        # ========== Supervised Loss ==========
        # Reuse the helper method instead of duplicating code
        sup_loss1, total_mask = self._compute_supervised_loss(logits1, labels, mask)
        sup_loss2, _ = self._compute_supervised_loss(logits2, labels, mask)

        # ========== Contrastive Loss ==========
        # SelfSupervisedLoss expects two separate embedding tensors
        # embeddings1[i] and embeddings2[i] are automatically treated as positive pairs
        contrast_loss = self.contrastive_loss_fn(embeddings1, embeddings2)

        # ========== Total Loss ==========
        total_loss = (
            sup_loss1 + sup_loss2 \
                + self.training_config.lambda_contrast * contrast_loss
        )

        # Update accuracy using first view
        self.train_accuracy.update(logits1, labels, mask)

        # Compute batch accuracy for logging
        with torch.no_grad():
            preds = logits1.argmax(dim=-1)
            targets = labels.argmax(dim=-1)
            correct = ((preds == targets).float() * mask).sum()
            batch_acc = correct / total_mask.clamp_min(1.0)

        # ========== Logging ==========
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_sup_loss", sup_loss1 + sup_loss2, on_step=True, on_epoch=True)
        self.log("train_contrast_loss", contrast_loss, on_step=True, on_epoch=True)
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

        return total_loss

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
            mode="max",  # Maximize hangman_win_rate
            factor=0.5,  # Reduce LR to 50% when plateau
            patience=3,  # Wait 3 epochs before reducing LR
            min_lr=1e-6,  # Don't reduce below 0.000001
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "hangman_win_rate",  # Monitor win rate from callback
                "interval": "epoch",  # Check every epoch
                "frequency": 1,  # Check every 1 epoch
            },
        }
