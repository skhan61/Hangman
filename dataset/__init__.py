"""Dataset package exposing shared utilities for Hangman training."""

from .encoder_utils import CharacterEncoder, DEFAULT_ALPHABET
from .hangman_dataset import HangmanDataset, HangmanDatasetConfig

__all__ = [
    "CharacterEncoder",
    "DEFAULT_ALPHABET",
    "HangmanDataset",
    "HangmanDatasetConfig",
]
