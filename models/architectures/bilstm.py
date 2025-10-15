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

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        return_embeddings: bool = False,
        num_embedding_layers: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        packed_output, (h_n, c_n) = self.lstm(packed)
        lstm_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=inputs.size(1)
        )

        logits = self.output(self.dropout(lstm_output))

        logger.debug("Logits shape=%s", tuple(logits.shape))

        if return_embeddings:
            # Extract hidden states from multiple layers
            # h_n shape: [num_layers * 2, batch_size, hidden_dim]
            # Each layer has forward (even indices) and backward (odd indices) states

            # Clamp num_embedding_layers to valid range
            num_layers_to_use = min(num_embedding_layers, self.config.num_layers)

            if num_layers_to_use == 1:
                # Use only the last layer (backward compatible)
                forward_hidden = h_n[-2]  # [batch_size, hidden_dim]
                backward_hidden = h_n[-1]  # [batch_size, hidden_dim]
                embeddings = torch.cat([forward_hidden, backward_hidden], dim=-1)
            else:
                # Use multiple layers from the top
                # For 4-layer BiLSTM: h_n has 8 elements (4 forward + 4 backward)
                # Indices: [0,1,2,3,4,5,6,7] where even=forward, odd=backward
                layer_embeddings = []
                for layer_offset in range(num_layers_to_use):
                    # Get forward and backward for this layer (from the top)
                    forward_idx = -(2 * num_layers_to_use) + (2 * layer_offset)
                    backward_idx = forward_idx + 1
                    forward_h = h_n[forward_idx]
                    backward_h = h_n[backward_idx]
                    layer_embeddings.append(torch.cat([forward_h, backward_h], dim=-1))

                # Concatenate all layer embeddings
                embeddings = torch.cat(layer_embeddings, dim=-1)
                # Shape: [batch_size, num_embedding_layers * hidden_dim * 2]

            logger.debug("Embeddings shape=%s (using %d layers)",
                        tuple(embeddings.shape), num_layers_to_use)
            return logits, embeddings

        return logits
