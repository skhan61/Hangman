"""BiLSTM with Attention mechanism for Hangman.

Adds attention layer on top of BiLSTM to focus on important character positions.
Attention helps the model decide which positions are most relevant for prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.architectures.base import BaseArchitecture, BaseConfig


@dataclass
class HangmanBiLSTMAttentionConfig(BaseConfig):
    """Configuration for BiLSTM with Attention.

    Attention mechanism helps focus on relevant positions (e.g., revealed letters).
    """
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    attention_dim: int = 128  # Dimension for attention mechanism


logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Additive (Bahdanau-style) attention mechanism.

    Computes attention weights for each position, allowing the model
    to focus on important characters (e.g., revealed letters vs masked).
    """

    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()

        # Attention scoring network
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted output.

        Args:
            lstm_output: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len] - 1 for valid positions, 0 for padding

        Returns:
            attended_output: [batch_size, seq_len, hidden_dim] - attention-weighted
            attention_weights: [batch_size, seq_len, 1] - attention scores
        """
        # Compute attention scores for each position
        attention_scores = self.attention(lstm_output)  # [batch, seq_len, 1]

        # Mask out padding positions (set to very negative value)
        attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]

        # Apply attention weights to LSTM output
        attended_output = lstm_output * attention_weights  # [batch, seq_len, hidden_dim]

        return attended_output, attention_weights


class HangmanBiLSTMAttention(BaseArchitecture):
    """BiLSTM with Attention for Hangman.

    Architecture:
    - Character embeddings
    - Bidirectional LSTM layers
    - Attention mechanism (focus on important positions)
    - Output projection for letter prediction

    Benefits over vanilla BiLSTM:
    - Learns which positions are most informative
    - Better handling of long words
    - Can focus on revealed vs masked characters
    - Expected: +2-3% win rate improvement
    """

    def __init__(self, config: HangmanBiLSTMAttentionConfig):
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

        # Attention layer
        self.attention = AttentionLayer(
            hidden_dim=config.hidden_dim * 2,  # *2 for bidirectional
            attention_dim=config.attention_dim,
        )

        # Output projection (after attention)
        self.output = nn.Linear(config.hidden_dim * 2, config.vocab_size)

        logger.info(
            "Initialized BiLSTM-Attention with %d layers, hidden_dim=%d, attention_dim=%d",
            config.num_layers,
            config.hidden_dim,
            config.attention_dim,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention.

        Args:
            inputs: Token indices [batch_size, seq_len]
            lengths: Actual length of each sequence [batch_size]

        Returns:
            logits: Per-position letter predictions [batch_size, seq_len, 26]
        """
        logger.debug(
            "BiLSTM-Attention forward called with inputs shape=%s, lengths shape=%s",
            tuple(inputs.shape),
            tuple(lengths.shape),
        )

        batch_size, seq_len = inputs.shape

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
            packed_output, batch_first=True, total_length=seq_len
        )

        # Create attention mask (1 for valid positions, 0 for padding)
        mask = torch.arange(seq_len, device=inputs.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.float()

        # Apply attention
        attended_output, attention_weights = self.attention(lstm_output, mask)

        # Project to vocabulary
        logits = self.output(self.dropout(attended_output))

        logger.debug(
            "BiLSTM-Attention logits shape=%s, attention_weights shape=%s",
            tuple(logits.shape),
            tuple(attention_weights.shape),
        )

        return logits
