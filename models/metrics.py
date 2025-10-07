"""Shared metrics for Hangman models."""

from __future__ import annotations

import torch
from torchmetrics import Metric


class MaskedAccuracy(Metric):
    """Accuracy over positions that remain masked in the game state."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> None:
        preds = logits.argmax(dim=-1)
        target_ids = targets.argmax(dim=-1)
        mask_bool = mask > 0.0

        self.correct += ((preds == target_ids) & mask_bool).sum().to(self.correct.dtype)
        self.total += mask_bool.sum().to(self.total.dtype)

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.total.device)
        return self.correct / self.total


__all__ = ["MaskedAccuracy"]
