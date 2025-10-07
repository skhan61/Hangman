"""
Shared encoding utilities for both training (HangmanDataset) and inference.
This ensures the EXACT SAME encoding pipeline is used everywhere.
"""
from typing import List, Optional, Sequence, Tuple
import numpy as np

DEFAULT_ALPHABET = tuple(chr(i) for i in range(ord("a"), ord("z") + 1))

class CharacterEncoder:
    """
    Shared character encoding logic used by both dataset and inference.
    This ensures consistency between training and inference.
    """
    def __init__(self, alphabet: Sequence[str],
                 mask_token: str = "_",
                 pad_token: str = "<PAD>"):
        self.alphabet = alphabet
        self.mask_token = mask_token
        self.pad_token = pad_token

        self._alphabet_index = {ch.upper(): idx for idx, ch in enumerate(alphabet)}
        self._mask_idx = len(alphabet)
        self._pad_idx = self._mask_idx + 1

    @staticmethod
    def _normalize(letter: Optional[str]) -> Optional[str]:
        if letter is None:
            return None
        return letter.upper()

    def encode_state(self, state: List[Optional[str]]) -> Tuple[np.ndarray, int]:
        """
        Encode a word state (list of characters or None for masked positions).
        """
        encoded = []
        for ch in state:
            norm = self._normalize(ch)
            if ch is None or ch == self.mask_token:
                encoded.append(self._mask_idx)
            else:
                encoded.append(self._alphabet_index.get(norm, self._mask_idx))
        return np.array(encoded, dtype=np.int64), len(encoded)

    def encode_guessed(self, guessed: Sequence[str], as_binary: bool = True) -> np.ndarray:
        """Encode guessed letters."""
        if not as_binary:
            return np.array(
                [self._alphabet_index[self._normalize(g)] for g in guessed],
                dtype=np.int64,
            )

        vector = np.zeros(len(self.alphabet), dtype=np.float32)
        for letter in guessed:
            idx = self._alphabet_index.get(self._normalize(letter))
            if idx is not None:
                vector[idx] = 1.0
        return vector

    def encode_targets(self, targets: dict, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Encode target positions and letters."""
        vocab_size = len(self.alphabet)
        labels = np.zeros((length, vocab_size), dtype=np.float32)
        mask = np.zeros(length, dtype=np.float32)

        for position, letter in targets.items():
            idx = self._alphabet_index.get(self._normalize(letter))
            if idx is not None:
                labels[position, idx] = 1.0
                mask[position] = 1.0

        return labels, mask
