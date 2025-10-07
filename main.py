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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from dataset.data_module import HangmanDataModule, HangmanDataModuleConfig
from models import (
    HangmanBiLSTM,
    HangmanBiLSTMConfig,
    HangmanLightningModule,
    TrainingConfig,
)
from simulation.data_generation import (
    generate_dataset_parquet,
    read_words_list,
)

logger = logging.getLogger(__name__)


def ensure_dataset(
    *,
    words_file: Path,
    dataset_path: Optional[Path],
    max_words: Optional[int],
    force: bool,
    show_summary: bool,
) -> Path:
    """Guarantee a parquet dataset exists and return its path."""

    words = read_words_list(str(words_file))
    if max_words:
        words = words[:max_words]

    # Default location mirrors simulation script naming: dataset_<N>words.parquet
    if dataset_path is None:
        suffix = f"{len(words)}words"
        dataset_path = words_file.parent / f"dataset_{suffix}.parquet"

    dataset_path = dataset_path.resolve()

    if dataset_path.exists() and not force:
        logger.info("Dataset found at %s. Skipping regeneration.", dataset_path)
        return dataset_path

    logger.info("Creating dataset at %s...", dataset_path)
    generate_dataset_parquet(
        str(words_file),
        dataset_path,
        words=words,
        force=True,
        show_summary=show_summary,
    )
    return dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare hangman dataset")
    parser.add_argument(
        "--words-file",
        type=Path,
        default=Path("data/words_250000_train.txt"),
        help="Text file containing newline-separated candidate words.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Destination parquet path. Defaults to data/dataset_<N>words.parquet.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional cap on number of words to include.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the parquet file already exists.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run a Lightning Trainer fit loop after dataset prep.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for the Lightning data module.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.9,
        help="Fraction of dataset to use for training.",
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
        help="Enable early stopping on the monitored validation metric.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of validation checks with no improvement before stopping (requires --early-stopping).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum change in monitored metric to qualify as improvement (requires --early-stopping).",
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

    if args.train:
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except AttributeError:
            pass

    dataset_path = ensure_dataset(
        words_file=args.words_file,
        dataset_path=args.dataset_path,
        max_words=args.max_words,
        force=args.force,
        show_summary=args.debug,
    )

    datamodule_config = HangmanDataModuleConfig(
        parquet_path=dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_val_split=args.train_val_split,
    )

    datamodule = HangmanDataModule(datamodule_config)
    datamodule.setup()

    def tensor_repr(t: torch.Tensor):
        if t.ndim <= 2:
            return t.detach().cpu().numpy()
        return "<omitted>"

    first_sample = datamodule.train_dataset[0]
    logger.debug("Sample from dataset before batching:")
    for key, tensor in first_sample.items():
        logger.debug(
            "  %s: shape=%s, dtype=%s, repr=%s",
            key,
            tuple(tensor.shape),
            tensor.dtype,
            tensor_repr(tensor),
        )

    train_loader = datamodule.train_dataloader()
    eval_loader = datamodule.val_dataloader()
    logger.debug("len(train_loader) = %s batches", len(train_loader))
    logger.debug("len(eval_loader) = %s batches", len(eval_loader))

    batch = next(iter(train_loader))
    logger.debug("Dataset ready at %s.", dataset_path)
    logger.debug(
        (
            "First batch tensor shapes — inputs: %s, labels: %s, label_mask: %s, "
            "miss_chars: %s, lengths: %s"
        ),
        tuple(batch["inputs"].shape),
        tuple(batch["labels"].shape),
        tuple(batch["label_mask"].shape),
        tuple(batch["miss_chars"].shape),
        tuple(batch["lengths"].shape),
    )

    model_config = HangmanBiLSTMConfig(
        vocab_size=len(batch["miss_chars"][0]),
        mask_idx=datamodule.dataset._mask_idx,
        pad_idx=datamodule.dataset._pad_idx,
    )
    model = HangmanBiLSTM(model_config)

    logits = model(batch["inputs"], batch["lengths"])
    logger.debug("Model output logits shape: %s", tuple(logits.shape))

    if args.train:
        lightning_module = HangmanLightningModule(
            model,
            TrainingConfig(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            ),
        )

        trainer_kwargs = {
            "max_epochs": args.max_epochs,
            "logger": False,
            "enable_checkpointing": args.save_top_k != 0,
            "log_every_n_steps": 1,
            "enable_progress_bar": True,
        }

        refresh_rate = args.progress_refresh

        if args.debug:
            trainer_kwargs.update(
                {
                    "limit_train_batches": 1,
                    "limit_val_batches": 1,
                    "num_sanity_val_steps": 0,
                }
            )
            refresh_rate = 1

        callbacks = [TQDMProgressBar(refresh_rate=refresh_rate)]

        if args.save_top_k != 0:
            checkpoint_dir = args.checkpoint_dir.resolve()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            monitor_metric = args.monitor_metric
            callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="epoch{epoch:03d}-{metric:.4f}".replace(
                        "metric", monitor_metric
                    ),
                    monitor=monitor_metric,
                    mode=args.monitor_mode,
                    save_top_k=args.save_top_k,
                    save_last=True,
                )
            )

        if args.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=args.monitor_metric,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    mode=args.monitor_mode,
                    verbose=True,
                )
            )

        logger.debug("Trainer configuration: %s", trainer_kwargs)

        trainer = Trainer(callbacks=callbacks, **trainer_kwargs)
        if datamodule.val_dataset is None or len(datamodule.val_dataset) == 0:
            logger.warning(
                "Validation dataset is empty; skipping pre-training validation."
            )
        else:
            logger.info("Running pre-training validation...")
            trainer.validate(lightning_module, datamodule=datamodule)

        logger.info("Starting training for %s epochs", args.max_epochs)
        trainer.fit(lightning_module, datamodule=datamodule)
        logger.info("Training complete")


if __name__ == "__main__":
    main()
