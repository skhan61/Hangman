"""PyTorch Lightning DataModule for Hangman dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import logging

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split

from .hangman_dataset import (
    DEFAULT_ALPHABET,
    HangmanDataset,
    HangmanDatasetConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class HangmanDataModuleConfig:
    parquet_path: Path
    batch_size: int = 256
    num_workers: int = 4
    train_val_split: float = 0.9
    shuffle: bool = True


class HangmanDataModule(LightningDataModule):
    def __init__(self, config: HangmanDataModuleConfig):
        super().__init__()
        self.config = config
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.pad_idx = len(DEFAULT_ALPHABET) + 1

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            dataset_config = HangmanDatasetConfig(parquet_path=self.config.parquet_path)
            self.dataset = HangmanDataset(dataset_config)
            self.pad_idx = self.dataset._pad_idx  # internal constant

            total_len = len(self.dataset)
            train_len = int(total_len * self.config.train_val_split)
            val_len = total_len - train_len

            if val_len == 0:
                train_len = max(total_len - 1, 1)
                val_len = total_len - train_len

            self.train_dataset, self.val_dataset = random_split(
                self.dataset, lengths=[train_len, val_len]
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate,
        )

    def _collate(self, batch):
        inputs = [item["inputs"] for item in batch]
        labels = [item["labels"] for item in batch]
        label_mask = [item["label_mask"] for item in batch]
        miss_chars = [item["miss_chars"] for item in batch]
        lengths = [item["length"] for item in batch]

        inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=self.pad_idx
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=0.0
        )
        label_mask = torch.nn.utils.rnn.pad_sequence(
            label_mask, batch_first=True, padding_value=0.0
        )

        miss_chars = torch.stack(miss_chars)
        lengths = torch.stack(lengths)

        # logger.debug(
        #     "Collated batch shapes — inputs: %s, labels: %s, label_mask: %s, miss_chars: %s, lengths: %s",
        #     tuple(inputs.shape),
        #     tuple(labels.shape),
        #     tuple(label_mask.shape),
        #     tuple(miss_chars.shape),
        #     tuple(lengths.shape),
        # )

        return {
            "inputs": inputs,
            "labels": labels,
            "label_mask": label_mask,
            "miss_chars": miss_chars,
            "lengths": lengths,
        }
