"""Base architecture class for Hangman models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class BaseConfig:
    """Base configuration for all Hangman architectures."""

    vocab_size: int = 26  # A-Z
    max_word_length: int = 45  # Longest word in English is 45 chars

    # Special token indices
    mask_idx: int = 26
    pad_idx: int = 27

    def get_vocab_size_with_special(self) -> int:
        """Get vocab size including special tokens (MASK + PAD)."""
        return self.vocab_size + 2  # +2 for MASK and PAD


class BaseArchitecture(nn.Module, ABC):
    """Abstract base class for Hangman architectures."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning per-position logits.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len]
            lengths: Length tensor of shape [batch_size]
            return_embeddings: If True, return (logits, embeddings) tuple

        Returns:
            If return_embeddings=False: logits of shape [batch_size, seq_len, vocab_size]
            If return_embeddings=True: (logits, embeddings) where embeddings is [batch_size, hidden_dim]
        """

    def get_num_parameters(self, only_trainable: bool = True) -> int:
        """Return number of model parameters."""

        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_model_summary(self) -> str:
        """Return string summary of model architecture."""

        total_params = self.get_num_parameters(only_trainable=False)
        trainable_params = self.get_num_parameters(only_trainable=True)

        summary = [
            f"Model: {self.__class__.__name__}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Vocabulary size: {self.config.vocab_size}",
            f"Max word length: {self.config.max_word_length}",
        ]
        return "\n".join(summary)
