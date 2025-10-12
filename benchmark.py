"""Benchmark script to compare multiple Hangman model architectures.

This script trains multiple models and records their performance metrics
for comparison. Results are saved to CSV and displayed in a table.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset.data_module import HangmanDataModule, HangmanDataModuleConfig
from dataset.encoder_utils import DEFAULT_ALPHABET
from hangman_callback.callback import CustomHangmanEvalCallback
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


logger = logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters in a model.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"


def format_number(num: int) -> str:
    """Format large numbers with M/K suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def parse_eval_words(value: str) -> int | None:
    """Allow positive integers or 'all' for evaluation word count."""
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


def create_model(model_arch: str, vocab_size: int, mask_idx: int, pad_idx: int) -> torch.nn.Module:
    """Create a model instance based on architecture name.

    Args:
        model_arch: Name of model architecture
        vocab_size: Size of vocabulary
        mask_idx: Index for mask token
        pad_idx: Index for padding token

    Returns:
        Instantiated model
    """
    if model_arch == "bilstm":
        config = HangmanBiLSTMConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        return HangmanBiLSTM(config)

    elif model_arch == "bilstm_attention":
        config = HangmanBiLSTMAttentionConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        return HangmanBiLSTMAttention(config)

    elif model_arch == "bilstm_multihead":
        config = HangmanBiLSTMMultiHeadConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        return HangmanBiLSTMMultiHead(config)

    elif model_arch == "gru":
        config = HangmanGRUConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        return HangmanGRU(config)

    elif model_arch == "charrnn":
        config = HangmanCharRNNConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
        )
        return HangmanCharRNN(config)

    elif model_arch == "mlp":
        config = HangmanMLPConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            max_word_length=45,
        )
        return HangmanMLP(config)

    elif model_arch == "transformer":
        config = HangmanTransformerConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            max_word_length=45,
        )
        return HangmanTransformer(config)

    elif model_arch == "bert":
        config = HangmanBERTConfig(
            vocab_size=vocab_size,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            max_word_length=45,
            freeze_bert_layers=False,
            num_layers_to_freeze=6,
        )
        return HangmanBERT(config)

    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")


def train_model(
    model_arch: str,
    datamodule: HangmanDataModule,
    args: argparse.Namespace,
    vocab_size: int,
    mask_idx: int,
    pad_idx: int,
) -> dict[str, Any]:
    """Train a single model and return its metrics.

    Args:
        model_arch: Name of model architecture
        datamodule: Data module for training
        args: Command line arguments
        vocab_size: Size of vocabulary
        mask_idx: Index for mask token
        pad_idx: Index for padding token

    Returns:
        Dictionary of metrics (win_rate, time, params, etc)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_arch}...")
    logger.info(f"{'='*60}")

    # Create model
    model = create_model(model_arch, vocab_size, mask_idx, pad_idx)

    # Count parameters
    trainable_params, total_params = count_parameters(model)
    logger.info(f"  Parameters: {format_number(trainable_params)} trainable, {format_number(total_params)} total")

    # Create lightning module
    lightning_module = HangmanLightningModule(
        model,
        TrainingModuleConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )

    # Setup checkpoint callback
    checkpoint_dir = Path(args.checkpoint_dir) / model_arch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"best-{model_arch}-{{epoch:02d}}-{{hangman_win_rate:.4f}}",
        monitor="hangman_win_rate",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=False,
    )

    # Setup evaluation callback
    evaluation_callback = CustomHangmanEvalCallback(
        val_words_path=str(args.test_words_file_path),
        dictionary_path=str(args.words_file_path),
        max_words=args.eval_words,
        verbose=False,
        parallel=not args.debug,
        patience=0,  # No early stopping in benchmark
        min_delta=0.0,
        mode="max",
        frequency=1,
    )

    # Setup trainer
    torch.set_float32_matmul_precision('medium')

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=False,
        enable_checkpointing=True,
        log_every_n_steps=50,
        enable_progress_bar=not args.no_progress,
        accelerator="auto",
        num_sanity_val_steps=0,
        callbacks=[evaluation_callback, checkpoint_callback],
    )

    # Train
    start_time = time.time()
    try:
        trainer.fit(lightning_module, train_dataloaders=datamodule.train_dataloader())
        training_time = time.time() - start_time
        success = True
    except Exception as e:
        logger.error(f"Training failed for {model_arch}: {e}")
        training_time = time.time() - start_time
        success = False

    # Get best metrics from callback
    if success and hasattr(evaluation_callback, 'best_win_rate'):
        best_win_rate = evaluation_callback.best_win_rate
        best_epoch = evaluation_callback.best_epoch
    else:
        best_win_rate = 0.0
        best_epoch = 0

    # Get checkpoint path
    if checkpoint_callback.best_model_path:
        checkpoint_path = checkpoint_callback.best_model_path
    else:
        checkpoint_path = "N/A"

    logger.info(f"  Best Win Rate: {best_win_rate*100:.1f}% (epoch {best_epoch})")
    logger.info(f"  Training Time: {format_time(training_time)}")
    logger.info(f"  Checkpoint: {checkpoint_path}")

    return {
        "model": model_arch,
        "win_rate": best_win_rate,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "training_time_sec": training_time,
        "epochs": args.max_epochs,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "success": success,
    }


def save_results_csv(results: list[dict[str, Any]], output_file: Path) -> None:
    """Save benchmark results to CSV file."""
    if not results:
        return

    fieldnames = [
        "model",
        "win_rate",
        "trainable_params",
        "total_params",
        "training_time_sec",
        "epochs",
        "best_epoch",
        "checkpoint_path",
        "success",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\nResults saved to: {output_file}")


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table of results."""
    if not results:
        logger.info("No results to display")
        return

    # Sort by win rate (descending)
    sorted_results = sorted(results, key=lambda x: x["win_rate"], reverse=True)

    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK RESULTS")
    logger.info(f"{'='*80}")

    # Print header
    header = f"{'Model':<20} | {'Win Rate':>10} | {'Params':>10} | {'Time':>10} | {'Epochs':>7}"
    logger.info(header)
    logger.info("-" * 80)

    # Print rows
    best_win_rate = sorted_results[0]["win_rate"]
    for result in sorted_results:
        if not result["success"]:
            continue

        model = result["model"]
        win_rate = result["win_rate"] * 100
        params = format_number(result["trainable_params"])
        time_str = format_time(result["training_time_sec"])
        epoch = f"{result['best_epoch']}/{result['epochs']}"

        # Mark best model with star
        star = " ⭐" if result["win_rate"] == best_win_rate else ""

        row = f"{model:<20} | {win_rate:>9.1f}%{star} | {params:>10} | {time_str:>10} | {epoch:>7}"
        logger.info(row)

    logger.info("-" * 80)
    logger.info(f"\nBest Model: {sorted_results[0]['model']} ({sorted_results[0]['win_rate']*100:.1f}%)")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark multiple Hangman model architectures"
    )

    parser.add_argument(
        "--words-file-path",
        type=Path,
        default=Path("data/train_words.txt"),
        help="Training words file path",
    )
    parser.add_argument(
        "--test-words-file-path",
        type=Path,
        default=Path("data/test_words.txt"),
        help="Test words file path",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bilstm", "bilstm_attention", "bilstm_multihead", "gru", "charrnn", "mlp"],
        help="List of models to benchmark (default: all except bert/transformer)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Number of epochs to train each model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay",
    )
    parser.add_argument(
        "--eval-words",
        type=parse_eval_words,
        default=1000,
        help="Number of words to use for evaluation (use 'all' to evaluate the full list)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("logs/benchmarks"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results CSV",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main benchmark function."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    logger.info("="*60)
    logger.info("HANGMAN MODEL BENCHMARK")
    logger.info("="*60)
    logger.info(f"Models to benchmark: {', '.join(args.models)}")
    logger.info(f"Epochs per model: {args.max_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    eval_words_display = "all" if args.eval_words is None else args.eval_words
    logger.info(f"Evaluation words: {eval_words_display}")

    # Create results directory
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup data module (shared across all models)
    logger.info("\nPreparing dataset...")
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

    datamodule_config = HangmanDataModuleConfig(
        words_path=args.words_file_path,
        strategies=strategies,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        row_group_cache_size=50,
        prefetch_factor=4,
    )

    datamodule = HangmanDataModule(datamodule_config)
    datamodule.prepare_data()
    datamodule.setup()

    # Get vocab info
    batch = next(iter(datamodule.train_dataloader()))
    vocab_size = len(batch["labels"][0][0])
    mask_idx = len(DEFAULT_ALPHABET)
    pad_idx = len(DEFAULT_ALPHABET) + 1

    logger.info(f"Dataset prepared: vocab_size={vocab_size}")

    # Run benchmark for each model
    results = []
    total_models = len(args.models)

    for idx, model_arch in enumerate(args.models, 1):
        logger.info(f"\n[{idx}/{total_models}] Starting {model_arch}...")

        try:
            result = train_model(
                model_arch=model_arch,
                datamodule=datamodule,
                args=args,
                vocab_size=vocab_size,
                mask_idx=mask_idx,
                pad_idx=pad_idx,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {model_arch}: {e}")
            results.append({
                "model": model_arch,
                "win_rate": 0.0,
                "trainable_params": 0,
                "total_params": 0,
                "training_time_sec": 0.0,
                "epochs": args.max_epochs,
                "best_epoch": 0,
                "checkpoint_path": "N/A",
                "success": False,
            })

    # Save and display results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_file = args.results_dir / f"benchmark_{timestamp}.csv"
    save_results_csv(results, output_file)
    print_comparison_table(results)

    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
