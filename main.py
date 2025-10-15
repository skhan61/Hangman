"""Central entry point for preparing Hangman datasets.

This script checks whether a parquet dataset already exists. If it does, the
path is surfaced so downstream components (e.g. data modules) can reuse it. If
not, it triggers dataset generation using the simulation utilities.
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
import warnings
from pathlib import Path
from typing import Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

# Suppress PyTorch nested tensor prototype warning
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*")

from dataset.data_module import HangmanDataModule, HangmanDataModuleConfig
from models import (
    HangmanBERT,
    HangmanBERTConfig,
    HangmanBiLSTM,
    HangmanBiLSTMConfig,
    HangmanBiLSTMAttention,
    HangmanBiLSTMAttentionConfig,
    HangmanBiLSTMMultiHead,
    HangmanBiLSTMMultiHeadConfig,
    HangmanCharRNN,
    HangmanCharRNNConfig,
    HangmanGRU,
    HangmanGRUConfig,
    HangmanMLP,
    HangmanMLPConfig,
    HangmanTransformer,
    HangmanTransformerConfig,
    HangmanLightningModule,
    TrainingModuleConfig,
)

# from simulation.data_generation import (
#     generate_dataset_and_save_parquet,
#     read_words_list,
# )
from hangman_callback.callback import CustomHangmanEvalCallback
from hangman_callback.telegram_notification import TelegramNotificationCallback

logger = logging.getLogger(__name__)


def parse_eval_words(value: str) -> int | None:
    """Parse evaluation word limit allowing integers or 'all'."""
    lowered = value.strip().lower()
    if lowered == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "eval-words must be a positive integer or 'all'"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "eval-words must be a positive integer when specified"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare hangman dataset")
    parser.add_argument(
        "--words-file-path",
        "--words-file_path",
        dest="words_file_path",
        type=Path,
        default=Path("data/train_words.txt"),
        help="Text file containing newline-separated candidate words.",
    )
    parser.add_argument(
        "--test-words-file_path",
        dest="test_words_file_path",
        type=Path,
        default=Path("data/test_words.txt"),
        help="Text file containing words used for hangman game testing.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for the Lightning data module.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Max epochs for the Trainer when --train is set.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer when --train is set.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer when --train is set.",
    )
    parser.add_argument(
        "--progress-refresh",
        type=int,
        default=20,
        help="TQDM progress bar refresh rate in steps (smaller = more frequent updates).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("logs/checkpoints"),
        help="Directory to store model checkpoints when training.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="Number of best checkpoints to keep (<=0 disables saving).",
    )
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="val_acc",
        help="Metric name to monitor for checkpoint selection.",
    )
    parser.add_argument(
        "--monitor-mode",
        type=str,
        choices=["min", "max"],
        default="max",
        help="Whether the monitored metric should be minimized or maximized.",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping based on hangman win rate.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Number of testing evaluations with no improvement before stopping (requires --early-stopping).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.01,
        help="Minimum change in hangman win rate to qualify as improvement (requires --early-stopping).",
    )
    parser.add_argument(
        "--test-word-limit",
        type=int,
        default=-1,
        help="Number of testing words to sample (<=0 uses all available).",
    )
    parser.add_argument(
        "--test-eval-frequency",
        type=int,
        default=5,
        help="Run the hangman testing callback every N training epochs.",
    )
    parser.add_argument(
        "--eval-words",
        type=parse_eval_words,
        default=None,
        help="Number of words to use for hangman evaluation (use 'all' for the full list).",
    )
    parser.add_argument(
        "--model-arch",
        "--model_arch",
        dest="model_arch",
        type=str,
        choices=["bilstm", "bilstm_attention", "bilstm_multihead", "transformer", "bert", "charrnn", "gru", "mlp"],
        default="bilstm",
        help="Model architecture to use for Hangman training.",
    )
    parser.add_argument(
        "--freeze-bert",
        action="store_true",
        help="Freeze all BERT encoder layers (only train embeddings and output head).",
    )
    parser.add_argument(
        "--freeze-bert-layers",
        type=int,
        default=0,
        help="Number of bottom BERT layers to freeze (0 = freeze none, 6 = freeze bottom half).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="letter_based",
        help=(
            "Comma-separated list of masking strategies to use for data generation. "
            "Options: letter_based, left_to_right, right_to_left, random_position, "
            "vowels_first, frequency_based, center_outward, edges_first, alternating, "
            "rare_letters_first, consonants_first, word_patterns, random_percentage. "
            "Use 'all' to enable all 13 strategies. Default: letter_based"
        ),
    )
    parser.add_argument(
        "--row-group-cache-size",
        type=int,
        default=50,
        help="Number of parquet row groups to cache in memory (higher = faster for large batches).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Number of batches to prefetch per worker (higher = faster but more memory).",
    )
    parser.add_argument(
        "--use-contrastive",
        action="store_true",
        help="Enable contrastive self-supervised learning with dual forward passes.",
    )
    parser.add_argument(
        "--lambda-contrast",
        type=float,
        default=0.1,
        help="Weight for contrastive loss component (default: 0.1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature parameter for NTXent contrastive loss (default: 0.07).",
    )
    parser.add_argument(
        "--embedding-regularizer",
        type=str,
        default=None,
        choices=['lp', 'center_invariant', 'zero_mean'],
        help="Embedding regularizer for contrastive loss. Options: lp (L2), center_invariant, zero_mean.",
    )
    parser.add_argument(
        "--regularizer-weight",
        type=float,
        default=1.0,
        help="Weight for embedding regularizer (default: 1.0).",
    )
    parser.add_argument(
        "--num-embedding-layers",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Number of top LSTM layers to use for contrastive embeddings (1-4, default: 1).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.log_level.upper() == "NONE":
        logging.disable(logging.CRITICAL)
    else:
        level = (
            logging.DEBUG
            if args.debug
            else getattr(logging, args.log_level.upper(), logging.INFO)
        )
        logging.basicConfig(
            level=level,
            format="%(levelname)s | %(name)s | %(message)s",
            force=True,
        )
        logger.setLevel(level)

    # if args.train:
    #     try:
    # # sys.stdout.reconfigure(line_buffering=True)
    #     except AttributeError:
    #         pass

    # Parse strategies from command line argument
    # if args.strategies.lower() == 'all':
    strategies = [
        "letter_based",
        "left_to_right",
        "right_to_left",
        "random_position",
        "vowels_first",
        "frequency_based",
        "center_outward",
        "edges_first",
        "alternating",
        "rare_letters_first",
        "consonants_first",
        "word_patterns",
        "random_percentage",
    ]
    # else:
    #     strategies = [s.strip() for s in args.strategies.split(',')]

    logger.info("Using masking strategies: %s", strategies)

    logger.info(
        f"Preparing data module with batch size {args.batch_size}, num_workers {args.num_workers}"
    )

    datamodule_config = HangmanDataModuleConfig(
        words_path=args.words_file_path,
        # eval_words_path=args.test_words_file_path,
        strategies=strategies,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        row_group_cache_size=args.row_group_cache_size,
        prefetch_factor=args.prefetch_factor,
        # train_val_split=args.train_val_split,
    )

    datamodule = HangmanDataModule(datamodule_config)
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()

    logger.debug("len(train_loader) = %s batches", (len(train_loader)))

    # # eval_loader = datamodule.val_dataloader()
    # logger.debug("len(train_loader) = %s batches", len(train_loader))
    # # logger.debug("len(eval_loader) = %s batches", len(eval_loader))

    batch = next(iter(train_loader))
    inputs_shape = tuple(batch["inputs"].shape)
    labels_shape = tuple(batch["labels"].shape)
    mask_shape = tuple(batch["label_mask"].shape)
    lengths_shape = tuple(batch["lengths"].shape)
    logger.debug(
        f"First batch tensor shapes — inputs: {inputs_shape}, labels: {labels_shape}, label_mask: {mask_shape}, lengths: {lengths_shape}"
    )
    logger.debug(
        (
            "First batch tensor shapes — inputs: %s, labels: %s, label_mask: %s, "
            "lengths: %s"
        ),
        tuple(batch["inputs"].shape),
        tuple(batch["labels"].shape),
        tuple(batch["label_mask"].shape),
        tuple(batch["lengths"].shape),
    )

    vocab_size = len(batch["labels"][0][0])
    logger.debug(f"Vocab size: {vocab_size}")

    # Get mask_idx and pad_idx from the dataset's encoder
    from dataset.encoder_utils import DEFAULT_ALPHABET

    mask_idx = len(DEFAULT_ALPHABET)
    pad_idx = len(DEFAULT_ALPHABET) + 1

    if args.model_arch == "transformer":
        model_config = HangmanTransformerConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            max_word_length=45,  # Use hardcoded max length for all sequences
        )
        model = HangmanTransformer(model_config)
        logger.info("Initialized HangmanTransformer with config: %s", model_config)
    elif args.model_arch == "bert":
        model_config = HangmanBERTConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            max_word_length=45,  # Use hardcoded max length for all sequences
            freeze_bert_layers=args.freeze_bert,
            num_layers_to_freeze=args.freeze_bert_layers,
        )
        model = HangmanBERT(model_config)
        logger.info("Initialized HangmanBERT with config: %s", model_config)
    elif args.model_arch == "charrnn":
        model_config = HangmanCharRNNConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        model = HangmanCharRNN(model_config)
        logger.info("Initialized HangmanCharRNN with config: %s", model_config)
    elif args.model_arch == "gru":
        model_config = HangmanGRUConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        model = HangmanGRU(model_config)
        logger.info("Initialized HangmanGRU with config: %s", model_config)
    elif args.model_arch == "mlp":
        model_config = HangmanMLPConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            max_word_length=45,  # MLP needs max_word_length for positional embeddings
        )
        model = HangmanMLP(model_config)
        logger.info("Initialized HangmanMLP with config: %s", model_config)
    elif args.model_arch == "bilstm_attention":
        model_config = HangmanBiLSTMAttentionConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        model = HangmanBiLSTMAttention(model_config)
        logger.info("Initialized HangmanBiLSTM-Attention with config: %s", model_config)
    elif args.model_arch == "bilstm_multihead":
        model_config = HangmanBiLSTMMultiHeadConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        model = HangmanBiLSTMMultiHead(model_config)
        logger.info("Initialized HangmanBiLSTM-MultiHead with config: %s", model_config)
    elif args.model_arch == "bilstm":
        model_config = HangmanBiLSTMConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        model = HangmanBiLSTM(model_config)
        logger.info("Initialized HangmanBiLSTM with config: %s", model_config)
    else:
        raise ValueError(
            f"Unsupported model architecture: '{args.model_arch}'. "
            f"Supported architectures: bilstm, bilstm_attention, bilstm_multihead, "
            f"transformer, bert, charrnn, gru, mlp"
        )

    logits = model(batch["inputs"], batch["lengths"])
    logger.debug("Model output logits shape: %s", tuple(logits.shape))

    lightning_module = HangmanLightningModule(
        model,
        TrainingModuleConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_contrastive=args.use_contrastive,
            lambda_contrast=args.lambda_contrast,
            temperature=args.temperature,
            embedding_regularizer=args.embedding_regularizer,
            regularizer_weight=args.regularizer_weight,
            num_embedding_layers=args.num_embedding_layers,
        ),
    )

    eval_words_display = "all" if args.eval_words is None else args.eval_words
    logger.info("Hangman evaluation word limit: %s", eval_words_display)

    evaluation_callback = CustomHangmanEvalCallback(
        val_words_path=str(args.test_words_file_path),
        dictionary_path=str(args.words_file_path),
        max_words=args.eval_words,
        verbose=args.debug,
        parallel=not args.debug,
        patience=args.patience if args.early_stopping else 0,
        min_delta=args.min_delta,
        mode=args.monitor_mode,
        frequency=1, # args.test_eval_frequency,
    )

    # Setup model checkpoint callback to save BEST model based on hangman win rate
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best-hangman-{epoch:02d}-{hangman_win_rate:.4f}",
        monitor="hangman_win_rate",
        mode="max",
        save_top_k=1,  # Save only the best model
        save_last=False,  # Don't save last checkpoint
        verbose=True,
        every_n_epochs=1,  # Check every epoch (will only save when metric exists)
    )

    # Enable Tensor Cores for better performance on CUDA devices
    torch.set_float32_matmul_precision('medium')
    logger.info("Set float32 matmul precision to 'medium' for Tensor Cores")

    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "logger": False,
        "enable_checkpointing": True,  # Enable checkpointing
        "log_every_n_steps": 1,
        "enable_progress_bar": True,
        "accelerator": "auto",
        "num_sanity_val_steps": 0,  # Skip validation sanity checks
    }

    # Add Telegram notification callback
    telegram_callback = TelegramNotificationCallback(
        send_on_epoch_end=True,
        send_on_train_end=True,
    )

    callbacks = [evaluation_callback, checkpoint_callback, telegram_callback]
    logger.info("Best model checkpoint will be saved to: %s", args.checkpoint_dir)

    trainer = Trainer(**trainer_kwargs, callbacks=callbacks)
    # trainer = Trainer(**trainer_kwargs) # , callbacks=[evaluation_callback])

    logger.info("Starting training for %s epochs", args.max_epochs)
    try:
        trainer.fit(lightning_module, train_dataloaders=train_loader)
        logger.info("Training complete")
    except Exception as e:
        logger.error("Training failed with error: %s", e, exc_info=True)
        raise

    # logger.info("Running initial hangman evaluation after training...")
    # eval_summary = evaluation_callback._run_evaluation(
    #     lightning_module.model,
    #     )
    
    # logger.info("Win rate: %.2f", eval_summary["win_rate"])
    # logger.info("Average tries remaining: %.2f", \
    #             eval_summary["average_tries_remaining"])

if __name__ == "__main__":
    main()
