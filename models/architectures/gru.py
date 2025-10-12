"""Bidirectional GRU architecture for Hangman.

GRU (Gated Recurrent Unit) is a simplified version of LSTM with fewer gates.
It's faster than LSTM but still handles long-term dependencies well.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.architectures.base import BaseArchitecture, BaseConfig


@dataclass
class HangmanGRUConfig(BaseConfig):
    """Configuration for GRU-based Hangman model.

    GRU has fewer parameters than LSTM (2 gates vs 3) but similar performance.
    Good middle ground between CharRNN and BiLSTM.
    """
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    bidirectional: bool = True  # Use bidirectional GRU


logger = logging.getLogger(__name__)


class HangmanGRU(BaseArchitecture):
    """Bidirectional GRU for Hangman.

    Architecture:
    - Character embeddings (a-z + MASK + PAD)
    - Bidirectional GRU layers (faster than LSTM)
    - Dropout for regularization
    - Linear output layer for letter prediction

    GRU advantages over LSTM:
    - 33% fewer parameters (2 gates vs 3 gates)
    - Faster training and inference
    - Similar performance on most tasks

    GRU advantages over vanilla RNN:
    - Better long-term memory
    - Solves vanishing gradient problem
    """

    def __init__(self, config: HangmanGRUConfig):
        super().__init__(config)

        vocab_with_special = config.get_vocab_size_with_special()

        # Character embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_with_special,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

        # Output layer size depends on bidirectional or not
        output_input_size = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.output = nn.Linear(output_input_size, config.vocab_size)

        logger.info(
            "Initialized GRU with %d layers, hidden_dim=%d, bidirectional=%s",
            config.num_layers,
            config.hidden_dim,
            config.bidirectional,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass for GRU.

        Args:
            inputs: Token indices [batch_size, seq_len]
            lengths: Actual length of each sequence [batch_size]

        Returns:
            logits: Per-position letter predictions [batch_size, seq_len, 26]
        """
        logger.debug(
            "GRU forward called with inputs shape=%s, lengths shape=%s",
            tuple(inputs.shape),
            tuple(lengths.shape),
        )

        # Embed characters
        embed = self.embedding(inputs)
        embed = self.dropout(embed)

        # Pack sequences for efficient GRU processing
        packed = pack_padded_sequence(
            embed, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through GRU
        packed_output, _ = self.gru(packed)

        # Unpack sequences
        gru_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=inputs.size(1)
        )

        # Project to vocabulary for letter prediction
        logits = self.output(self.dropout(gru_output))

        logger.debug("GRU logits shape=%s", tuple(logits.shape))
        return logits