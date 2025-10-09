"""Torch dataset for Hangman parquet states."""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

try:
    from .encoder_utils import CharacterEncoder, DEFAULT_ALPHABET
except ImportError:  # pragma: no cover - script execution helper
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from dataset.encoder_utils import CharacterEncoder, DEFAULT_ALPHABET

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pq = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class HangmanDatasetConfig:
    parquet_path: Path
    row_group_cache_size: int = 200  # Default for large batches

    def __post_init__(self) -> None:
        self.parquet_path = Path(self.parquet_path)


class HangmanDataset(Dataset):
    """Lazy parquet reader with small row-group cache."""

    def __init__(self, config: HangmanDatasetConfig):
        if pq is None:
            raise ImportError("pyarrow is required for HangmanDataset")

        self.config = config

        if not self.config.parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet file not found at {self.config.parquet_path}"
            )

        self.encoder = CharacterEncoder(
            alphabet=DEFAULT_ALPHABET,
            mask_token="_",
            pad_token="<PAD>",
        )

        self._pq = None
        self._num_rows: int = 0
        self._row_group_boundaries: List[int] = []
        self._cache_limit = max(1, int(self.config.row_group_cache_size))
        self._cache: Dict[int, object] = {}
        self._cache_order: Deque[int] = deque()

        metadata = self._open_parquet()

        logger.debug(
            "HangmanDataset initialised — rows=%d, row_groups=%d, cache_size=%d",
            self._num_rows,
            metadata.num_row_groups,
            self._cache_limit,
        )

    @staticmethod
    def _build_row_group_boundaries(metadata) -> List[int]:
        boundaries: List[int] = []
        total = 0
        for rg_idx in range(metadata.num_row_groups):
            total += metadata.row_group(rg_idx).num_rows
            boundaries.append(total)
        return boundaries

    def _log_metadata_preview(self, metadata) -> None:
        """Log summary stats and first-row preview to aid debugging."""
        try:
            total_bytes = sum(
                metadata.row_group(i).total_byte_size
                for i in range(metadata.num_row_groups)
            )
        except Exception:  # pragma: no cover - defensive logging helper
            total_bytes = 0

        logger.info(
            "Parquet file: %d rows, %d row groups, %.2f MB",
            metadata.num_rows,
            metadata.num_row_groups,
            total_bytes / (1024 * 1024),
        )

        if metadata.num_row_groups == 0 or self._pq is None:
            logger.debug("Parquet file contains no row groups to preview.")
            return

        try:
            first_row_group = self._pq.read_row_group(0)
            first_row = first_row_group.slice(0, 1).to_pydict()
            logger.debug("First row from parquet: %s", first_row)
        except Exception as exc:  # pragma: no cover - optional preview
            logger.debug("Unable to preview first row group: %s", exc)

    def _open_parquet(self):
        pq_file = pq.ParquetFile(self.config.parquet_path)
        metadata = pq_file.metadata

        if metadata is None:
            raise ValueError(
                f"Parquet file {self.config.parquet_path} is missing metadata"
            )

        self._pq = pq_file
        self._num_rows = metadata.num_rows
        self._row_group_boundaries = self._build_row_group_boundaries(metadata)
        self.reset_cache()
        self._log_metadata_preview(metadata)
        return metadata

    def __len__(self) -> int:
        return self._num_rows

    def _locate_row_group(self, idx: int) -> tuple[int, int]:
        if self._num_rows == 0:
            raise IndexError("Cannot index into empty HangmanDataset")

        if idx < 0:
            idx += self._num_rows

        if not 0 <= idx < self._num_rows:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self._num_rows}"
            )

        rg_idx = bisect_right(self._row_group_boundaries, idx)
        start = 0 if rg_idx == 0 else self._row_group_boundaries[rg_idx - 1]
        local_idx = idx - start
        return rg_idx, local_idx

    def _touch_cache(self, rg_idx: int) -> None:
        if self._cache_order and self._cache_order[-1] == rg_idx:
            return
        try:
            self._cache_order.remove(rg_idx)
        except ValueError:
            return
        self._cache_order.append(rg_idx)

    def _add_to_cache(self, rg_idx: int, rows: Sequence[Dict[str, object]]) -> None:
        self._cache[rg_idx] = rows
        self._cache_order.append(rg_idx)
        while len(self._cache_order) > self._cache_limit:
            oldest = self._cache_order.popleft()
            self._cache.pop(oldest, None)

    def _load_row_group(self, rg_idx: int):
        cached = self._cache.get(rg_idx)
        if cached is not None:
            self._touch_cache(rg_idx)
            return cached

        if self._pq is None:
            raise RuntimeError("Parquet file handle is not initialised.")

        table = self._pq.read_row_group(rg_idx)
        self._add_to_cache(rg_idx, table)
        return table

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rg_idx, local_idx = self._locate_row_group(idx)
        row_group = self._load_row_group(rg_idx)

        if not 0 <= local_idx < row_group.num_rows:
            raise IndexError(
                f"Local index {local_idx} out of range for row group {rg_idx}"
            )

        row = row_group.slice(local_idx, 1).to_pylist()[0]

        # Get raw data from parquet
        word = row["word"]
        state_sequence = list(row.get("state") or [])
        targets_list = row.get("targets", [])  # PyArrow map returns list of tuples
        length = int(row.get("length", 0))

        # Convert PyArrow map format [(key, val), ...] to dict
        targets_dict = dict(targets_list) if targets_list else {}

        # Encode state to tensor
        encoded_state, enc_len = self.encoder.encode_state(state_sequence)
        inputs_tensor = torch.as_tensor(encoded_state, dtype=torch.long)

        # Encode targets to labels and mask tensors
        labels_arr, mask_arr = self.encoder.encode_targets(targets_dict, length)
        labels_tensor = torch.as_tensor(labels_arr, dtype=torch.float32)
        mask_tensor = torch.as_tensor(mask_arr, dtype=torch.float32)

        return {
            "word": word,
            "state": state_sequence,
            "inputs": inputs_tensor,
            "labels": labels_tensor,
            "label_mask": mask_tensor,
            "length": torch.tensor(length, dtype=torch.long),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_pq", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = {}
        self._cache_order = deque()
        metadata = self._open_parquet()
        logger.debug(
            "HangmanDataset parquet reopened — rows=%d, row_groups=%d, cache_size=%d",
            self._num_rows,
            metadata.num_row_groups,
            self._cache_limit,
        )

    def reset_cache(self) -> None:
        """Clear cached row groups."""
        self._cache.clear()
        self._cache_order.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    dataset_path = Path("/home/sayem/Desktop/Hangman/data/dataset_227300words.parquet")

    try:
        dataset = HangmanDataset(HangmanDatasetConfig(parquet_path=dataset_path))
    except Exception as exc:  # pragma: no cover - manual execution helper
        logger.error("Failed to load dataset from %s: %s", dataset_path, exc)
        raise SystemExit(1) from exc

    logger.info("Loaded HangmanDataset from %s with %d rows", dataset_path, len(dataset))

    try:
        sample = dataset[1_000]
        print(sample)
    except Exception as exc:  # pragma: no cover - manual execution helper
        logger.error("Failed to read first sample: %s", exc)
        raise SystemExit(1) from exc

    labels_shape = tuple(sample["labels"].shape)
    mask_nonzero = int(sample["label_mask"].sum().item())
    logger.info(
        "Sample[0] — word=%s, inputs=%s, labels=%s, mask_nonzero=%d, length=%d",
        sample["word"],
        tuple(sample["inputs"].shape),
        labels_shape,
        mask_nonzero,
        int(sample["length"].item()),
    )
