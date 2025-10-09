"""Quick smoke tests for the offline Hangman API."""

from __future__ import annotations

import argparse
import logging
import os
import random
import string
from pathlib import Path
from typing import Iterable, List

import numpy as np

from api.offline_api import HangmanOfflineAPI
from api.guess_strategies \
    import frequency_guess_strategy, bert_style_guess_strategy

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

    log.info("Done")

    log.info("Using guess strategy: %s", bert_style_guess_strategy.__name__)
    api = HangmanOfflineAPI(strategy=bert_style_guess_strategy)
    summary = api.simulate_games_for_word_list(
        words, 
        parallel=args.parallel, 
        max_workers=args.max_workers
    )
    log.info("Aggregate results: overall=%s", summary["overall"])
    for length, stats in sorted(summary["results_by_length"].items()):
        log.debug("length=%s stats=%s", length, stats)

    log.info("Done")

if __name__ == "__main__":
    main()
