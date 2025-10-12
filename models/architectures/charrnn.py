"""Character-level RNN architecture for Hangman.

This implements a vanilla RNN (not LSTM/GRU) with character-level embeddings.
Simpler than BiLSTM but can still learn sequential patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.architectures.base import BaseArchitecture, BaseConfig


@dataclass
class HangmanCharRNNConfig(BaseConfig):
    """Configuration for Character RNN Hangman model.

    A simple RNN architecture with character embeddings.
    Can be unidirectional or bidirectional.
    """
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True  # Use bidirectional RNN


logger = logging.getLogger(__name__)


class HangmanCharRNN(BaseArchitecture):
    """Character-level RNN for Hangman.

    Architecture:
    - Character embeddings (a-z + MASK + PAD)
    - Vanilla RNN layers (can be bidirectional)
    - Dropout for regularization
    - Linear output layer for letter prediction

    Simpler than LSTM but faster and uses less memory.
    """

    def __init__(self, config: HangmanCharRNNConfig):
        super().__init__(config)

        vocab_with_special = config.get_vocab_size_with_special()

        # Character embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_with_special,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Vanilla RNN (simpler than LSTM)
        self.rnn = nn.RNN(
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
            "Initialized CharRNN with %d layers, hidden_dim=%d, bidirectional=%s",
            config.num_layers,
            config.hidden_dim,
            config.bidirectional,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass for Character RNN.

        Args:
            inputs: Token indices [batch_size, seq_len]
            lengths: Actual length of each sequence [batch_size]

        Returns:
            logits: Per-position letter predictions [batch_size, seq_len, 26]
        """
        logger.debug(
            "CharRNN forward called with inputs shape=%s, lengths shape=%s",
            tuple(inputs.shape),
            tuple(lengths.shape),
        )

        # Embed characters
        embed = self.embedding(inputs)
        embed = self.dropout(embed)

        # Pack sequences for efficient RNN processing
        packed = pack_padded_sequence(
            embed, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through RNN
        packed_output, _ = self.rnn(packed)

        # Unpack sequences
        rnn_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=inputs.size(1)
        )

        # Project to vocabulary for letter prediction
        logits = self.output(self.dropout(rnn_output))

        logger.debug("CharRNN logits shape=%s", tuple(logits.shape))
        return logits
