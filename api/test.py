"""Quick smoke tests for the offline Hangman API."""

from __future__ import annotations

import argparse
import logging
import os
import random
import string
from functools import partial
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from api.offline_api import HangmanOfflineAPI
from api.guess_strategies import (
    frequency_guess_strategy,
    bert_style_guess_strategy,
    neural_guess_strategy,
)

log = logging.getLogger(__name__)


def load_test_words(
    path: Path,
    limit: int = -1,
    *,
    seed: int | None = None,
    shuffle: bool = True,
) -> List[str]:
    """Load words from ``path`` optionally shuffling and truncating."""
    words: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            word = line.strip()
            if not word:
                continue
            words.append(word.lower())
            if limit >= 0 and len(words) >= limit:
                break

    if shuffle and words:
        rng = np.random.default_rng(seed)
        rng.shuffle(words)

    return words


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the offline Hangman API")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging (default is INFO).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Number of words to test (negative means all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible shuffling (overrides HANGMAN_TEST_SEED).",
    )
    parser.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        default=True,
        help="Run simulations in parallel (default).",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum worker processes when running in parallel mode.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    repo_root = Path(__file__).resolve().parent.parent
    words_path = repo_root / "data" / "test_unique.txt"

    env_seed = os.getenv("HANGMAN_TEST_SEED") if args.seed is None else None
    seed_value = (
        args.seed
        if args.seed is not None
        else (int(env_seed) if env_seed is not None and env_seed.isdigit() else None)
    )

    if seed_value is not None:
        np.random.seed(seed_value)
        random.seed(seed_value)
        log.debug("Using deterministic seed=%d", seed_value)
    else:
        log.debug("Using non-deterministic shuffle seed")

    words = load_test_words(words_path, limit=args.limit, seed=seed_value, shuffle=True)
    log.info("Loaded %d words", len(words))

    if not words:
        log.warning("No words available to test")
        return

    log.info("Using guess strategy: %s", frequency_guess_strategy.__name__)
    api = HangmanOfflineAPI(strategy=frequency_guess_strategy)  # bert_style_guess_strategy

    sample_word = words[0]
    win, attempts_remaining, progress \
        = api.play_a_game_with_a_word(sample_word)
    log.info(
        "Single word test -> word='%s', win=%s, attempts_remaining=%s",
        sample_word,
        win,
        attempts_remaining,
    )
    log.debug("Game progress: %s", progress)

    summary = api.simulate_games_for_word_list(
        words, 
        parallel=args.parallel, 
        max_workers=args.max_workers
    )
    log.info("Aggregate results: overall=%s", summary["overall"])
    for length, stats in sorted(summary["results_by_length"].items()):
        log.debug("length=%s stats=%s", length, stats)

    # Test BERT-style strategy
    log.info("\n" + "="*60)
    log.info("Testing BERT-style Strategy")
    log.info("="*60)
    log.info("Using guess strategy: %s", bert_style_guess_strategy.__name__)
    api_bert = HangmanOfflineAPI(strategy=bert_style_guess_strategy)

    log.info("Testing with BERT-style strategy...")
    summary_bert = api_bert.simulate_games_for_word_list(
        words,
        parallel=args.parallel,
        max_workers=args.max_workers
    )
    log.info("BERT-style strategy results: overall=%s", summary_bert["overall"])

    # Test neural strategy with checkpoint
    log.info("\n" + "="*60)
    log.info("Testing Neural Strategy with trained model")
    log.info("="*60)

    checkpoint_dir = repo_root / "logs" / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("best-hangman-*.ckpt"))

    if checkpoint_files:
        # Load the best checkpoint (sorted by win rate)
        best_checkpoint = max(checkpoint_files, key=lambda p: float(p.stem.split("=")[-1]))
        log.info("Loading checkpoint: %s", best_checkpoint.name)

        # Load checkpoint
        checkpoint = torch.load(best_checkpoint, map_location='cpu')

        # Reconstruct model from checkpoint hyperparameters
        from models import HangmanBiLSTM, HangmanTransformer, HangmanBiLSTMConfig, HangmanTransformerConfig
        from dataset.encoder_utils import DEFAULT_ALPHABET

        # Determine model type from checkpoint
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            model_type = hparams.get('model', {})
            if hasattr(model_type, '__class__'):
                model_class_name = model_type.__class__.__name__
            else:
                model_class_name = 'HangmanBiLSTM'  # Default
        else:
            model_class_name = 'HangmanBiLSTM'  # Default

        # Create model config
        vocab_size = 26  # alphabet size
        mask_idx = len(DEFAULT_ALPHABET)
        pad_idx = len(DEFAULT_ALPHABET) + 1

        if 'Transformer' in model_class_name:
            config = HangmanTransformerConfig(vocab_size=vocab_size, mask_idx=mask_idx, pad_idx=pad_idx, max_word_length=45)
            model = HangmanTransformer(config)
        else:
            config = HangmanBiLSTMConfig(vocab_size=vocab_size, mask_idx=mask_idx, pad_idx=pad_idx)
            model = HangmanBiLSTM(config)

        # Load state dict
        from models import HangmanLightningModule, TrainingModuleConfig
        lightning_module = HangmanLightningModule(model, TrainingModuleConfig())
        lightning_module.load_state_dict(checkpoint['state_dict'])
        lightning_module.eval()

        log.info("Model loaded successfully")

        # Create neural strategy
        strategy = partial(neural_guess_strategy, model=lightning_module.model)
        api_neural = HangmanOfflineAPI(strategy=strategy)

        log.info("Testing with neural strategy...")
        summary_neural = api_neural.simulate_games_for_word_list(
            words,
            parallel=False,  # Neural strategy doesn't support parallel
            max_workers=None
        )
        log.info("Neural strategy results: overall=%s", summary_neural["overall"])

        # Compare all results
        log.info("\n" + "="*60)
        log.info("COMPARISON - ALL STRATEGIES")
        log.info("="*60)
        log.info("Frequency Strategy - Win Rate: %.2f%%, Avg Tries: %.2f",
                 summary["overall"]["win_rate"] * 100,
                 summary["overall"]["average_tries_remaining"])
        log.info("BERT-style Strategy - Win Rate: %.2f%%, Avg Tries: %.2f",
                 summary_bert["overall"]["win_rate"] * 100,
                 summary_bert["overall"]["average_tries_remaining"])
        log.info("Neural Strategy     - Win Rate: %.2f%%, Avg Tries: %.2f",
                 summary_neural["overall"]["win_rate"] * 100,
                 summary_neural["overall"]["average_tries_remaining"])
        log.info("="*60)
    else:
        log.warning("No checkpoint found in %s. Skipping neural strategy test.", checkpoint_dir)

        # Still show comparison of frequency and BERT
        log.info("\n" + "="*60)
        log.info("COMPARISON - Frequency vs BERT-style")
        log.info("="*60)
        log.info("Frequency Strategy  - Win Rate: %.2f%%, Avg Tries: %.2f",
                 summary["overall"]["win_rate"] * 100,
                 summary["overall"]["average_tries_remaining"])
        log.info("BERT-style Strategy - Win Rate: %.2f%%, Avg Tries: %.2f",
                 summary_bert["overall"]["win_rate"] * 100,
                 summary_bert["overall"]["average_tries_remaining"])
        log.info("="*60)

    log.info("Done")


if __name__ == "__main__":
    main()
