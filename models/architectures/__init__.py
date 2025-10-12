"""Neural network architectures for Hangman."""

from .base import BaseArchitecture, BaseConfig
from .bert import HangmanBERT, HangmanBERTConfig
from .bilstm import HangmanBiLSTM, HangmanBiLSTMConfig
from .bilstm_attention import HangmanBiLSTMAttention, HangmanBiLSTMAttentionConfig
from .bilstm_multihead import HangmanBiLSTMMultiHead, HangmanBiLSTMMultiHeadConfig
from .charrnn import HangmanCharRNN, HangmanCharRNNConfig
from .gru import HangmanGRU, HangmanGRUConfig
from .mlp import HangmanMLP, HangmanMLPConfig
from .transformer import HangmanTransformer, HangmanTransformerConfig

__all__ = [
    "BaseArchitecture",
    "BaseConfig",
    "HangmanBERT",
    "HangmanBERTConfig",
    "HangmanBiLSTM",
    "HangmanBiLSTMConfig",
    "HangmanBiLSTMAttention",
    "HangmanBiLSTMAttentionConfig",
    "HangmanBiLSTMMultiHead",
    "HangmanBiLSTMMultiHeadConfig",
    "HangmanCharRNN",
    "HangmanCharRNNConfig",
    "HangmanGRU",
    "HangmanGRUConfig",
    "HangmanMLP",
    "HangmanMLPConfig",
    "HangmanTransformer",
    "HangmanTransformerConfig",
]
