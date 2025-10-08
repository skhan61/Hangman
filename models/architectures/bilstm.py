"""Bidirectional LSTM architecture for Hangman."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.architectures.base import BaseArchitecture, BaseConfig


@dataclass
class HangmanBiLSTMConfig(BaseConfig):
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3


logger = logging.getLogger(__name__)


class HangmanBiLSTM(BaseArchitecture):
    def __init__(self, config: HangmanBiLSTMConfig):
        super().__init__(config)

        vocab_with_special = config.get_vocab_size_with_special()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_with_special,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.output = nn.Linear(config.hidden_dim * 2, config.vocab_size)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        logger.debug(
            "Forward called with inputs shape=%s, lengths shape=%s",
            tuple(inputs.shape),
            tuple(lengths.shape),
        )

        embed = self.embedding(inputs)
        embed = self.dropout(embed)

        packed = pack_padded_sequence(
            embed, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=inputs.size(1)
        )

        logits = self.output(self.dropout(lstm_output))

        logger.debug("Logits shape=%s", tuple(logits.shape))
        return logits
