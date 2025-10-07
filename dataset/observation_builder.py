"""
Observation builder for Hangman game inference.
This is a thin compatibility layer that uses the shared dataset encoder.
Provides a unified interface for both API and game engine.
"""
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.encoder_utils import CharacterEncoder, DEFAULT_ALPHABET

# Create shared encoder instance (same config as dataset)
_encoder = CharacterEncoder(
    alphabet=DEFAULT_ALPHABET,
    mask_token="_",
    pad_token="<PAD>"
)


def build_model_inputs(masked_word):
    """
    Build model inputs using shared dataset encoder.

    Args:
        masked_word: Current game state (e.g., "_pp_e")
        guessed_letters: List of guessed letters (e.g., ['a', 'e', 'p'])

    Returns:
        Tuple of (state_tensor, length_tensor, guessed_binary_tensor)
        All use the same encoding as the training dataset.
    """
    import torch

    # Encode masked word state
    state_list = list(masked_word)
    encoded_state, length = _encoder.encode_state(state_list)

    # Convert to tensors with batch dimension
    state_tensor = torch.tensor(encoded_state, dtype=torch.long).unsqueeze(0)
    length_tensor = torch.tensor([length], dtype=torch.long)
    # guessed_tensor = torch.from_numpy(guessed_binary).float().unsqueeze(0)

    return state_tensor, length_tensor
