"""PyTorch Lightning DataModule for Hangman dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import logging

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split

from .hangman_dataset import (
    HangmanDataset,
    HangmanDatasetConfig,
)
from .encoder_utils import DEFAULT_ALPHABET
from .data_generation import read_words_list

import os

logger = logging.getLogger(__name__)


DEFAULT_WORDS_FILE = Path("data/words_250000_train.txt")

# DEFAULT_STRATEGIES = [
#     "letter_based",
#     "left_to_right",
#     "right_to_left",
#     "random_position",
#     "vowels_first",
#     "frequency_based",
#     "center_outward",
#     "edges_first",
#     "alternating",
#     "rare_letters_first",
#     "consonants_first",
#     "word_patterns",
#     "random_percentage",
# ]
DEFAULT_PARQUET_PATH = Path("data/dataset_227300words.parquet")
from .data_generation import generate_full_dataset


@dataclass
class HangmanDataModuleConfig:
    words_path: Path = DEFAULT_WORDS_FILE
    # eval_words_path: Optional[Path] = (None,)
    strategies: Sequence[str] = None,
    batch_size: int = 1024
    num_workers: int = os.cpu_count()
    # train_val_split: Optional[float] = None
    shuffle: bool = True
    lazy_load: bool = True
    row_group_cache_size: int = 200  # Cache row groups for faster batching
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 8  # Prefetch batches per worker


class HangmanDataModule(LightningDataModule):
    def __init__(self, config: HangmanDataModuleConfig):
        super().__init__()
        # print("Initializing HangmanDataModule with config:", config)
        self.config = config
        self.dataset_path = None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.dataset_path = self._resolve_parquet_path()

        if self.dataset_path.exists():
            logger.info("Dataset already present at %s", self.dataset_path)
            self.config.parquet_path = self.dataset_path
            return

        logger.info("Dataset missing at %s; generating...", self.dataset_path)

        # Read all words
        words = read_words_list(str(self.config.words_path))

        # Define schema for raw trajectory data (no encoding/padding)
        schema = pa.schema(
            [
                ("word", pa.string()),
                ("state", pa.list_(pa.string())),
                ("targets", pa.map_(pa.int32(), pa.string())),  # {position: letter}
                ("length", pa.int64()),
            ]
        )

        # Ensure parent directory exists
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        # Write parquet file in batches to avoid memory issues
        batch_size = 10_000  # Process 10k words at a time
        total_samples_written = 0

        writer = None
        try:
            for batch_start in range(0, len(words), batch_size):
                batch_end = min(batch_start + batch_size, len(words))
                word_batch = words[batch_start:batch_end]

                logger.info(
                    "Processing words %d-%d / %d", batch_start, batch_end, len(words)
                )

                # Generate samples for this batch
                batch_samples = generate_full_dataset(
                    words=word_batch,
                    strategies=list(self.config.strategies),
                    parallel=True,
                    num_workers=self.config.num_workers,
                )

                # Convert to PyArrow table
                records = [vars(sample) for sample in batch_samples]
                batch_table = pa.Table.from_pylist(records, schema=schema)

                # Write to parquet (create writer on first batch)
                if writer is None:
                    writer = pq.ParquetWriter(self.dataset_path, schema)

                writer.write_table(batch_table)
                total_samples_written += len(batch_samples)

                logger.info(
                    "Wrote %d samples (total: %d)",
                    len(batch_samples),
                    total_samples_written,
                )

                # Free memory
                del batch_samples, records, batch_table

        finally:
            if writer is not None:
                writer.close()

    def setup(self, stage: Optional[str] = None) -> None:
        # Use dataset_path if already set, otherwise resolve it
        if self.dataset_path is None:
            self.dataset_path = self._resolve_parquet_path()

        dataset_config = HangmanDatasetConfig(
            parquet_path=self.dataset_path,
            row_group_cache_size=self.config.row_group_cache_size,
        )
        self.dataset = HangmanDataset(dataset_config)
        self.train_dataset = self.dataset
        self.val_dataset = None

    def _resolve_parquet_path(self) -> Path:
        # Check if default parquet path exists
        if DEFAULT_PARQUET_PATH.exists():
            logger.info("Using existing default parquet at %s", DEFAULT_PARQUET_PATH)
            return DEFAULT_PARQUET_PATH

        # Otherwise use default parquet path as target
        return DEFAULT_PARQUET_PATH

    def train_dataloader(self) -> DataLoader:
        num_workers = self.config.num_workers
        # Persistent workers requires num_workers > 0
        persistent_workers = self.config.persistent_workers and num_workers > 0
        # prefetch_factor only valid when num_workers > 0
        prefetch_factor = self.config.prefetch_factor if num_workers > 0 else None

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=num_workers,
            collate_fn=HangmanDataset.collate_fn,
            persistent_workers=persistent_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=prefetch_factor,
            multiprocessing_context='fork' if num_workers > 0 else None,
        )

    # # def val_dataloader(self):
    # #     if self.val_dataset is None:
    # #         return []
    # #     return DataLoader(
    # #         self.val_dataset,
    # #         batch_size=self.config.batch_size,
    # #         shuffle=False,
    # #         num_workers=self.config.num_workers,
    # #         collate_fn=HangmanDataset.collate_fn,
    # #     )
