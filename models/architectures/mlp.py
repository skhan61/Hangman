"""Simple MLP (Multi-Layer Perceptron) architecture for Hangman.

This is a basic feed-forward neural network without recurrence.
It treats each position independently, making it the simplest baseline model.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn

from models.architectures.base import BaseArchitecture, BaseConfig


@dataclass
class HangmanMLPConfig(BaseConfig):
    """Configuration for MLP-based Hangman model.

    A simple feedforward network that processes each character position independently.
    No temporal/sequential modeling - serves as the simplest baseline.
    """
    embedding_dim: int = 128
    hidden_dims: tuple[int, ...] = (256, 256)  # Hidden layer sizes
    dropout: float = 0.3
    use_positional_encoding: bool = True  # Add position information


logger = logging.getLogger(__name__)


class HangmanMLP(BaseArchitecture):
    """Simple MLP for Hangman - no recurrence, treats positions independently.

    Architecture:
    - Character embeddings (a-z + MASK + PAD)
    - Optional positional embeddings (to give position information)
    - Multiple feedforward hidden layers with ReLU activation
    - Dropout for regularization
    - Linear output layer for letter prediction

    This is the simplest model - no memory of previous positions.
    Each character position is classified independently.

    Good for:
    - Fast baseline
    - Understanding if position matters
    - Debugging data pipeline

    Limitations:
    - Cannot learn patterns across positions (e.g., "q" followed by "u")
    - No understanding of word structure
    - Expected to perform worse than RNN/LSTM/Transformer
    """

    def __init__(self, config: HangmanMLPConfig):
        super().__init__(config)

        vocab_with_special = config.get_vocab_size_with_special()

        # Character embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_with_special,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )

        # Optional positional embeddings (to give each position unique info)
        self.use_positional_encoding = config.use_positional_encoding
        if self.use_positional_encoding:
            self.positional_embedding = nn.Embedding(
                num_embeddings=config.max_word_length,
                embedding_dim=config.embedding_dim,
            )
            mlp_input_dim = config.embedding_dim * 2  # Concatenate char + position
        else:
            mlp_input_dim = config.embedding_dim

        self.dropout = nn.Dropout(config.dropout)

        # Build MLP layers
        layers = []
        prev_dim = mlp_input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(prev_dim, config.vocab_size)

        logger.info(
            "Initialized MLP with hidden_dims=%s, positional_encoding=%s",
            config.hidden_dims,
            config.use_positional_encoding,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLP.

        Args:
            inputs: Token indices [batch_size, seq_len]
            lengths: Actual length of each sequence [batch_size] (not used in MLP)

        Returns:
            logits: Per-position letter predictions [batch_size, seq_len, 26]
        """
        logger.debug(
            "MLP forward called with inputs shape=%s, lengths shape=%s",
            tuple(inputs.shape),
            tuple(lengths.shape),
        )

        batch_size, seq_len = inputs.shape

        # Get character embeddings
        char_embed = self.embedding(inputs)  # [batch_size, seq_len, embedding_dim]

        # Optionally add positional information
        if self.use_positional_encoding:
            positions = torch.arange(seq_len, device=inputs.device, dtype=torch.long)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            pos_embed = self.positional_embedding(positions)  # [batch_size, seq_len, embedding_dim]

            # Concatenate character and position embeddings
            combined = torch.cat([char_embed, pos_embed], dim=-1)  # [batch_size, seq_len, 2*embedding_dim]
        else:
            combined = char_embed

        combined = self.dropout(combined)

        # Pass through MLP (processes each position independently)
        mlp_output = self.mlp(combined)  # [batch_size, seq_len, hidden_dim]

        # Project to vocabulary for letter prediction
        logits = self.output(mlp_output)  # [batch_size, seq_len, vocab_size]

        logger.debug("MLP logits shape=%s", tuple(logits.shape))
        return logits
