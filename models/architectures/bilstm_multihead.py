"""BiLSTM with Multi-Head prediction for Hangman.

Multiple prediction heads vote together for more robust predictions.
Ensemble-like behavior within a single model.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.architectures.base import BaseArchitecture, BaseConfig


@dataclass
class HangmanBiLSTMMultiHeadConfig(BaseConfig):
    """Configuration for BiLSTM with Multi-Head prediction.

    Multiple prediction heads create ensemble-like behavior.
    Each head learns slightly different patterns.
    """
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    num_heads: int = 3  # Number of prediction heads
    head_hidden_dim: int = 128  # Hidden dimension for each head


logger = logging.getLogger(__name__)


class MultiHeadPredictor(nn.Module):
    """Multiple prediction heads with voting mechanism.

    Each head has its own transformation and makes independent predictions.
    Final prediction is weighted average of all heads.
    """

    def __init__(self, input_dim: int, vocab_size: int, num_heads: int, head_hidden_dim: int, dropout: float):
        super().__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList()

        # Create multiple prediction heads
        for i in range(num_heads):
            head = nn.Sequential(
                nn.Linear(input_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, vocab_size),
            )
            self.heads.append(head)

        # Learnable weights for combining heads
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

        logger.info("Created MultiHeadPredictor with %d heads", num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multiple heads.

        Args:
            x: [batch_size, seq_len, hidden_dim]

        Returns:
            logits: [batch_size, seq_len, vocab_size] - weighted average of all heads
        """
        # Get predictions from each head
        head_outputs = []
        for head in self.heads:
            head_output = head(x)  # [batch, seq_len, vocab_size]
            head_outputs.append(head_output)

        # Stack all head outputs
        stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, seq_len, vocab_size]

        # Normalize head weights with softmax
        normalized_weights = torch.softmax(self.head_weights, dim=0)

        # Weighted average of head predictions
        weighted_output = torch.sum(
            stacked * normalized_weights.view(-1, 1, 1, 1),
            dim=0
        )  # [batch, seq_len, vocab_size]

        return weighted_output


class HangmanBiLSTMMultiHead(BaseArchitecture):
    """BiLSTM with Multi-Head prediction for Hangman.

    Architecture:
    - Character embeddings
    - Bidirectional LSTM layers
    - Multiple prediction heads (ensemble-like)
    - Weighted voting for final prediction

    Benefits over vanilla BiLSTM:
    - More robust predictions (ensemble within single model)
    - Each head can specialize in different patterns
    - Reduces overfitting through diversity
    - Expected: +2-4% win rate improvement
    """

    def __init__(self, config: HangmanBiLSTMMultiHeadConfig):
        super().__init__(config)

        vocab_with_special = config.get_vocab_size_with_special()

        # Character embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_with_special,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Multi-head predictor
        self.multihead_output = MultiHeadPredictor(
            input_dim=config.hidden_dim * 2,  # *2 for bidirectional
            vocab_size=config.vocab_size,
            num_heads=config.num_heads,
            head_hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )

        logger.info(
            "Initialized BiLSTM-MultiHead with %d layers, hidden_dim=%d, num_heads=%d",
            config.num_layers,
            config.hidden_dim,
            config.num_heads,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-head prediction.

        Args:
            inputs: Token indices [batch_size, seq_len]
            lengths: Actual length of each sequence [batch_size]

        Returns:
            logits: Per-position letter predictions [batch_size, seq_len, 26]
        """
        logger.debug(
            "BiLSTM-MultiHead forward called with inputs shape=%s, lengths shape=%s",
            tuple(inputs.shape),
            tuple(lengths.shape),
        )

        # Embed characters
        embed = self.embedding(inputs)
        embed = self.dropout(embed)

        # Pack sequences for LSTM
        packed = pack_padded_sequence(
            embed, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)

        # Unpack sequences
        lstm_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=inputs.size(1)
        )

        # Multi-head prediction with voting
        logits = self.multihead_output(self.dropout(lstm_output))

        logger.debug("BiLSTM-MultiHead logits shape=%s", tuple(logits.shape))

        return logits