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
from pathlib import Path
from typing import Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from dataset.data_module import HangmanDataModule, HangmanDataModuleConfig
from models import (
    HangmanBiLSTM,
    HangmanBiLSTMConfig,
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

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare hangman dataset")
    parser.add_argument(
        "--words-file-path",
        "--words-file_path",
        dest="words_file_path",
        type=Path,
        default=Path("data/words_250000_train.txt"),
        help="Text file containing newline-separated candidate words.",
    )
    parser.add_argument(
        "--test-words-file-path",
        "--test-words-file_path",
        dest="test_words_file_path",
        type=Path,
        default=Path("data/test_unique.txt"),
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
        default=3,
        help="Number of testing evaluations with no improvement before stopping (requires --early-stopping).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
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
        "--model-arch",
        "--model_arch",
        dest="model_arch",
        type=str,
        choices=["bilstm", "transformer"],
        default="bilstm",
        help="Model architecture to use for Hangman training.",
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
        # "left_to_right",
        # "right_to_left",
        # "random_position",
        # "vowels_first",
        # "frequency_based",
        # "center_outward",
        # "edges_first",
        # "alternating",
        # "rare_letters_first",
        # "consonants_first",
        # "word_patterns",
        # "random_percentage",
    ]
    # else:
    #     strategies = [s.strip() for s in args.strategies.split(',')]

    logger.info("Using masking strategies: %s", strategies)

    # dataset_path = ensure_dataset(
    #     words_file=args.words_file,
    #     dataset_path=args.dataset_path,
    #     # max_words=args.max_words,
    #     strategies=strategies,
    #     force=args.force,
    #     show_summary=args.debug,
    # )

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

    # if args.model_arch == "transformer":
    #     model_config = HangmanTransformerConfig(
    #         vocab_size=vocab_size,
    #         mask_idx=mask_idx,
    #         pad_idx=pad_idx,
    #         max_word_length=45,  # Use hardcoded max length for all sequences
    #     )
    #     model = HangmanTransformer(model_config)
    #     logger.info("Initialized HangmanTransformer with config: %s", model_config)
    # else:
    model_config = HangmanBiLSTMConfig(
        vocab_size=vocab_size,
        mask_idx=mask_idx,
        pad_idx=pad_idx,
    )
    model = HangmanBiLSTM(model_config)
    logger.info("Initialized model with config: %s", model_config)

    logits = model(batch["inputs"], batch["lengths"])
    logger.debug("Model output logits shape: %s", tuple(logits.shape))

    # from tqdm import tqdm
    # for batch in tqdm(train_loader):
    #     input = batch["inputs"].to('cuda')
    #     lengths = batch["lengths"].to('cuda')
    #     model = model.to('cuda')
    #     logits = model(input, lengths)
    #     logger.debug("Model output logits shape: %s", tuple(logits.shape))
    #     # break  # Just process one batch for demonstration

    # model = model.to('cuda')

    lightning_module = HangmanLightningModule(
        model,
        TrainingModuleConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )


    evaluation_callback = CustomHangmanEvalCallback(
        val_words_path=str(args.test_words_file_path),
        dictionary_path=str(args.words_file_path),
        max_words=1000,
        verbose=args.debug,
        parallel=not args.debug,
        patience=args.patience if args.early_stopping else 0,
        min_delta=args.min_delta,
        mode=args.monitor_mode,
        frequency=args.test_eval_frequency,
    )

    # logger.info("Running initial hangman evaluation before training...")
    # eval_summary = evaluation_callback._run_evaluation(
    #     lightning_module.model,
    #     )
    
    # logger.info("Win rate: %.2f", eval_summary["win_rate"])
    # logger.info("Average tries remaining: %.2f", \
    #             eval_summary["average_tries_remaining"])

    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "logger": False,
        "enable_checkpointing": False,  # Disable checkpointing to avoid issues
        "log_every_n_steps": 1,
        "enable_progress_bar": True,
        "accelerator": "auto",
        "num_sanity_val_steps": 0,  # Skip validation sanity checks
    }

    trainer = Trainer(**trainer_kwargs, callbacks=[evaluation_callback])
    # trainer = Trainer(**trainer_kwargs) # , callbacks=[evaluation_callback])

    logger.info("Starting training for %s epochs", args.max_epochs)
    try:
        trainer.fit(lightning_module, train_dataloaders=train_loader)
        logger.info("Training complete")
    except Exception as e:
        logger.error("Training failed with error: %s", e, exc_info=True)
        raise

    logger.info("Running initial hangman evaluation after training...")
    eval_summary = evaluation_callback._run_evaluation(
        lightning_module.model,
        )
    
    logger.info("Win rate: %.2f", eval_summary["win_rate"])
    logger.info("Average tries remaining: %.2f", \
                eval_summary["average_tries_remaining"])




if __name__ == "__main__":
    main()
