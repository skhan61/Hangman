"""Torch dataset for Hangman parquet states."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .encoder_utils import CharacterEncoder, DEFAULT_ALPHABET

# DEFAULT_ALPHABET = tuple(chr(i) for i in range(ord("A"), ord("Z") + 1))


@dataclass
class HangmanDatasetConfig:
    parquet_path: Path
    alphabet: Sequence[str] = DEFAULT_ALPHABET
    mask_token: str = "_"
    pad_token: str = "<PAD>"
    guessed_as_binary: bool = True


class HangmanDataset(Dataset):
    """Dataset that materializes hangman game states from a parquet file."""

    def __init__(self, config: HangmanDatasetConfig):
        self.config = config

        # Use shared encoder for consistency between training and inference
        self.encoder = CharacterEncoder(
            alphabet=config.alphabet,
            mask_token=config.mask_token,
            pad_token=config.pad_token,
        )

        # Keep for backwards compatibility
        self._alphabet_index = self.encoder._alphabet_index
        self._mask_idx = self.encoder._mask_idx
        self._pad_idx = self.encoder._pad_idx

        self.df = pd.read_parquet(config.parquet_path)
        self.samples = self.df.to_dict("records")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        state = sample["state"]
        guessed = sample["guessed"]
        targets = self._parse_targets(sample["targets"])

        # Use shared encoder methods
        inputs, lengths = self.encoder.encode_state(state)
        miss_vector = self.encoder.encode_guessed(
            guessed, as_binary=self.config.guessed_as_binary
        )
        labels, label_mask = self.encoder.encode_targets(targets, len(state))

        return {
            "inputs": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "label_mask": torch.tensor(label_mask, dtype=torch.float32),
            "miss_chars": torch.tensor(miss_vector, dtype=torch.float32),
            "length": torch.tensor(lengths, dtype=torch.long),
        }

    # Delegate to shared encoder (for backwards compatibility)
    def encode_state(self, state: List[Optional[str]]) -> Tuple[np.ndarray, int]:
        return self.encoder.encode_state(state)

    def _encode_guessed(self, guessed: Sequence[str]) -> np.ndarray:
        return self.encoder.encode_guessed(
            guessed, as_binary=self.config.guessed_as_binary
        )

    def _encode_targets(
        self, targets: Dict[int, str], length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.encoder.encode_targets(targets, length)

    @staticmethod
    def _parse_targets(raw: str) -> Dict[int, str]:
        if isinstance(raw, dict):
            return raw
        return {int(k): v for k, v in ast.literal_eval(raw).items()}
