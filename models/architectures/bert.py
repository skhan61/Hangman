"""BERT-based architecture for Hangman using Hugging Face transformers.

This uses a pretrained BERT model from Hugging Face transformers library,
fine-tuned for the Hangman task with character-level inputs.
Tokens represent: a-z (26 letters), MASK token (underscore), and PAD token.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from transformers import BertModel

from models.architectures.base import BaseArchitecture, BaseConfig


@dataclass
class HangmanBERTConfig(BaseConfig):
    """Configuration for BERT-based Hangman model using Hugging Face BERT.

    Hybrid approach: Uses pretrained BERT encoder with custom character embeddings.
    The vocabulary consists of 28 tokens: 26 letters (a-z), MASK (_), and PAD.
    """
    hidden_size: int = 768  # BERT base hidden size (fixed for pretrained)
    num_hidden_layers: int = 12  # BERT base layers (fixed for pretrained)
    num_attention_heads: int = 12  # BERT base attention heads (fixed for pretrained)
    intermediate_size: int = 3072  # BERT base FFN size (fixed for pretrained)
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    pretrained_model_name: str = "bert-base-uncased"  # Which BERT to use
    freeze_bert_layers: bool = False  # Whether to freeze pretrained BERT encoder
    num_layers_to_freeze: int = 0  # Number of bottom layers to freeze (0 = freeze none)


logger = logging.getLogger(__name__)


class HangmanBERT(BaseArchitecture):
    """BERT encoder for Hangman using Hugging Face transformers.

    Hybrid Architecture (Option 3):
    - Load pretrained BERT encoder (bert-base-uncased)
    - Replace word embeddings with custom character embeddings (28 tokens)
    - Keep pretrained positional embeddings and transformer layers
    - Add custom output head for letter prediction

    This combines transfer learning from pretrained BERT with task-specific
    character-level embeddings for optimal performance.
    """

    def __init__(self, config: HangmanBERTConfig):
        super().__init__(config)

        vocab_with_special = config.get_vocab_size_with_special()

        logger.info(
            "Loading pretrained BERT model: %s",
            config.pretrained_model_name,
        )

        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(config.pretrained_model_name)

        logger.info(
            "Replacing BERT word embeddings (vocab_size=%d) with custom character embeddings (vocab_size=%d)",
            self.bert.config.vocab_size,
            vocab_with_special,
        )

        # Replace the word embeddings with our custom character embeddings
        # Keep the original hidden_size to match pretrained transformer layers
        old_embeddings = self.bert.embeddings.word_embeddings
        new_embeddings = nn.Embedding(
            num_embeddings=vocab_with_special,
            embedding_dim=old_embeddings.embedding_dim,  # Must match BERT hidden_size
            padding_idx=config.pad_idx,
        )

        # Initialize new embeddings with small random values
        nn.init.normal_(new_embeddings.weight, mean=0.0, std=0.02)

        # Replace the embeddings in BERT
        self.bert.embeddings.word_embeddings = new_embeddings

        # Resize position embeddings if needed for longer sequences
        if config.max_word_length > self.bert.config.max_position_embeddings:
            logger.warning(
                "Hangman max_word_length (%d) > BERT max_position_embeddings (%d). "
                "Extending position embeddings.",
                config.max_word_length,
                self.bert.config.max_position_embeddings,
            )
            old_pos_embeddings = self.bert.embeddings.position_embeddings
            new_pos_embeddings = nn.Embedding(
                num_embeddings=config.max_word_length,
                embedding_dim=old_pos_embeddings.embedding_dim,
            )
            # Copy old weights and initialize new positions
            with torch.no_grad():
                new_pos_embeddings.weight[:old_pos_embeddings.num_embeddings] = old_pos_embeddings.weight
                nn.init.normal_(
                    new_pos_embeddings.weight[old_pos_embeddings.num_embeddings:],
                    mean=0.0,
                    std=0.02,
                )
            self.bert.embeddings.position_embeddings = new_pos_embeddings
            self.bert.embeddings.register_buffer(
                "position_ids",
                torch.arange(config.max_word_length).expand((1, -1)),
            )

        # Freeze BERT layers if requested
        if config.freeze_bert_layers:
            logger.info("Freezing all BERT encoder layers")
            for param in self.bert.encoder.parameters():
                param.requires_grad = False
        elif config.num_layers_to_freeze > 0:
            logger.info("Freezing bottom %d BERT encoder layers", config.num_layers_to_freeze)
            for layer_idx in range(config.num_layers_to_freeze):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False

        # Output projection: predict letter probabilities (26 classes)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        logger.info("HangmanBERT initialized with hybrid pretrained approach")

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass for Hangman BERT using Hugging Face transformers.

        Args:
            inputs: Token indices [batch_size, seq_len] - values 0-27
            lengths: Actual length of each sequence [batch_size]

        Returns:
            logits: Per-position letter predictions [batch_size, seq_len, 26]
        """
        logger.debug(
            "BERT forward called with inputs shape=%s, lengths shape=%s",
            tuple(inputs.shape),
            tuple(lengths.shape),
        )

        seq_len = inputs.shape[1]

        # Create attention mask for BERT (1 for real tokens, 0 for padding)
        # HuggingFace BERT uses opposite 
        # convention: 1 = attend, 0 = ignore
        attention_mask \
            = torch.arange(seq_len, \
            device=inputs.device).unsqueeze(0) < lengths.unsqueeze(1)
        attention_mask = attention_mask.long()

        # Pass through BERT encoder
        bert_output = self.bert(
            input_ids=inputs,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get the last hidden state
        encoded = bert_output.last_hidden_state

        # Project to vocabulary size for letter prediction
        logits = self.output(self.dropout(encoded))

        logger.debug("BERT logits shape=%s", tuple(logits.shape))
        return logits
