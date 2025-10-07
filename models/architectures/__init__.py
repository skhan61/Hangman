"""Neural network architectures for Hangman."""

from .base import BaseArchitecture, BaseConfig
from .bilstm import HangmanBiLSTM, HangmanBiLSTMConfig

__all__ = [
    "BaseArchitecture",
    "BaseConfig",
    "HangmanBiLSTM",
    "HangmanBiLSTMConfig",
]
