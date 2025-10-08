"""Transformer encoder architecture for Hangman."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn

from models.architectures.base import BaseArchitecture, BaseConfig


@dataclass
class HangmanTransformerConfig(BaseConfig):
    embedding_dim: int = 256
    num_heads: int = 8
    ff_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    use_sinusoidal_positional_encoding: bool = False


logger = logging.getLogger(__name__)


class HangmanTransformer(BaseArchitecture):
    def __init__(self, config: HangmanTransformerConfig):
        super().__init__(config)

        vocab_with_special = config.get_vocab_size_with_special()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_with_special,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )

        self.dropout = nn.Dropout(config.dropout)

        if config.use_sinusoidal_positional_encoding:
            self.positional_encoding = _SinusoidalPositionalEncoding(
                config.embedding_dim, config.max_word_length
            )
        else:
            self.positional_embedding = nn.Embedding(
                config.max_word_length,
                config.embedding_dim,
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.output = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        logger.debug(
            "Transformer forward called with inputs shape=%s, lengths shape=%s",
            tuple(inputs.shape),
            tuple(lengths.shape),
        )

        embed = self.embedding(inputs)

        if hasattr(self, "positional_embedding"):
            seq_len = inputs.size(1)
            # Clamp positions to avoid index out of bounds
            max_pos = self.positional_embedding.num_embeddings
            if seq_len > max_pos:
                logger.warning(
                    f"Sequence length {seq_len} exceeds max positional embeddings {max_pos}. "
                    f"Clamping to {max_pos}."
                )
                seq_positions = torch.arange(max_pos, device=inputs.device, dtype=torch.long)
                # Repeat the last position for overflow
                pos_embed = self.positional_embedding(seq_positions)
                pos_embed = pos_embed.unsqueeze(0)
                # Expand and pad
                pos_embed = torch.cat([
                    pos_embed.expand(embed.size(0), -1, -1),
                    pos_embed[:, -1:, :].expand(embed.size(0), seq_len - max_pos, -1)
                ], dim=1)
            else:
                seq_positions = torch.arange(seq_len, device=inputs.device, dtype=torch.long)
                pos_embed = self.positional_embedding(seq_positions)
                pos_embed = pos_embed.unsqueeze(0).expand_as(embed)
        else:
            pos_embed = self.positional_encoding(inputs.size(1), inputs.device)

        x = self.dropout(embed + pos_embed)

        key_padding_mask = torch.arange(inputs.size(1), device=inputs.device).unsqueeze(0) >= lengths.unsqueeze(1)
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)

        logits = self.output(self.dropout(encoded))
        logger.debug("Transformer logits shape=%s", tuple(logits.shape))
        return logits


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_length: int):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe = torch.zeros(max_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.pe[:seq_len].to(device)
