"""Offline Hangman API that reuses pluggable guess strategies."""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - tqdm optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from api.guess_strategies import GuessStrategy

logger = logging.getLogger(__name__)


class HangmanOfflineAPI:
    """Frequency-based hangman helper for local simulations."""

    def __init__(
        self,
        dictionary_file_location: str = "/home/sayem/Desktop/Hangman/data/words_250000_train.txt",
        *,
        strategy: GuessStrategy | None = None,
    ) -> None:
        self.dictionary_file_location = dictionary_file_location
        self.full_dictionary = self.build_dictionary(dictionary_file_location)
        self.full_dictionary_common_letter_sorted = Counter(
            "".join(self.full_dictionary)
        ).most_common()
        self.strategy: GuessStrategy = strategy 

        logger.debug(
            "Strategy name: %s",
            getattr(self.strategy, "__name__", repr(self.strategy)),
        )
        self.start_new_game()

    def build_dictionary(self, dictionary_file_location: str) -> List[str]:
        with open(dictionary_file_location, "r", encoding="utf-8") as fp:
            return [line.strip().lower() for line in fp if line.strip()]

    def start_new_game(self) -> None:
        self.guessed_letters: List[str] = []
        self.incorrect_letters: set[str] = set()
        self.current_dictionary: List[str] = list(self.full_dictionary)

    def guess(self, masked_state: str) -> str:
        strategy_name = getattr(self.strategy, "__name__", repr(self.strategy))
        logger.debug("Using strategy '%s' for state '%s'", strategy_name, masked_state)
        guess_letter = self.strategy(masked_state, self)
        if guess_letter not in self.guessed_letters:
            self.guessed_letters.append(guess_letter)
        return guess_letter

    def set_strategy(self, strategy: GuessStrategy) -> None:
        """Replace the current guess strategy."""
        self.strategy = strategy

    @staticmethod
    def update_word_state(
        word: str, masked_word: str, guessed_char: str
    ) -> Tuple[str, bool]:
        updated = list(masked_word)
        correct = False
        for idx, letter in enumerate(word):
            if letter == guessed_char:
                updated[idx] = guessed_char
                correct = True
        return "".join(updated), correct

    def play_a_game_with_a_word(
        self,
        word: str,
        *,
        max_attempts: int = 6,
        initial_masked_word: str | None = None,
    ) -> Tuple[bool, int, List[Tuple[str, str, bool]]]:
        target_word = word.lower()
        masked_word = (
            initial_masked_word.lower()
            if initial_masked_word is not None
            else "_" * len(target_word)
        )

        if len(masked_word) != len(target_word):
            raise ValueError("Masked word length must match the target word length.")

        self.start_new_game()

        attempts_remaining = max_attempts
        game_progress: List[Tuple[str, str, bool]] = []

        while "_" in masked_word and attempts_remaining > 0:
            guess_letter = self.guess(masked_word)
            new_masked_word, guess_correct = self.update_word_state(
                target_word, masked_word, guess_letter
            )

            if not guess_correct:
                attempts_remaining -= 1
                self.incorrect_letters.add(guess_letter)
                self.current_dictionary = [
                    candidate
                    for candidate in self.current_dictionary
                    if guess_letter not in candidate
                ]

            masked_word = new_masked_word
            game_progress.append((guess_letter, masked_word, guess_correct))

        win = masked_word == target_word
        return win, attempts_remaining, game_progress

    def simulate_games_for_word_list(
        self,
        word_list: Sequence[str],
        *,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> dict:
        if not word_list:
            return {
                "results_by_length": {},
                "overall": {
                    "total_games": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0,
                    "average_tries_remaining": 0,
                },
            }

        if parallel:
            dictionary_path = self.dictionary_file_location
            strategy = self.strategy
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                tasks = [
                    executor.submit(_simulate_single_word, (word, dictionary_path, strategy))
                    for word in word_list
                ]

                progress_bar = (
                    tqdm(total=len(word_list), desc="Simulating games", unit="word")
                    if tqdm
                    else None
                )

                word_results = []
                for future in tasks:
                    word_results.append(future.result())
                    if progress_bar is not None:
                        progress_bar.update(1)

                if progress_bar is not None:
                    progress_bar.close()
        else:
            word_results = []
            iterator: Iterable[str]
            iterator = (
                tqdm(word_list, desc="Simulating games", unit="word")
                if tqdm
                else word_list
            )

            for word in iterator:
                win, tries_remaining, game_progress = self.play_a_game_with_a_word(word)
                word_results.append(
                    {
                        "word": word,
                        "win": win,
                        "tries_remaining": tries_remaining,
                        "progress": game_progress,
                    }
                )

            if tqdm and hasattr(iterator, "close"):
                iterator.close()

        return _aggregate_results(word_results)


def _simulate_single_word(args: Tuple[str, str, GuessStrategy]) -> Dict:
    word, dictionary_path, strategy = args
    api = HangmanOfflineAPI(dictionary_file_location=dictionary_path, strategy=strategy)
    win, attempts_remaining, progress = api.play_a_game_with_a_word(word)
    return {
        "word": word,
        "win": win,
        "tries_remaining": attempts_remaining,
        "progress": progress,
    }


def _aggregate_results(word_results: Sequence[Dict]) -> Dict:
    results: Dict[int, Dict[str, object]] = {}
    total_wins = 0
    total_losses = 0
    total_tries_remaining: List[int] = []

    for entry in word_results:
        word = entry["word"]
        tries_remaining = entry["tries_remaining"]
        win = entry["win"]

        word_length = len(word)
        bucket = results.setdefault(
            word_length,
            {
                "wins": 0,
                "losses": 0,
                "total_tries_remaining": [],
                "games": [],
            },
        )

        bucket["games"].append(entry)
        bucket["total_tries_remaining"].append(tries_remaining)

        if win:
            bucket["wins"] += 1
            total_wins += 1
        else:
            bucket["losses"] += 1
            total_losses += 1

        total_tries_remaining.append(tries_remaining)

    overall_win_rate = total_wins / len(word_results) if word_results else 0
    average_tries_remaining = (
        sum(total_tries_remaining) / len(total_tries_remaining)
        if total_tries_remaining
        else 0
    )

    aggregated_results: Dict[int, Dict[str, object]] = {}
    for length, data in results.items():
        avg_tries_remaining = (
            sum(data["total_tries_remaining"]) / len(data["total_tries_remaining"])
            if data["total_tries_remaining"]
            else 0
        )
        total_games = data["wins"] + data["losses"]
        win_rate = data["wins"] / total_games if total_games > 0 else 0
        aggregated_results[length] = {
            "average_tries_remaining": avg_tries_remaining,
            "win_rate": win_rate,
            "total_games": total_games,
            "games": data["games"],
        }

    return {
        "results_by_length": aggregated_results,
        "overall": {
            "total_games": len(word_results),
            "wins": total_wins,
            "losses": total_losses,
            "win_rate": overall_win_rate,
            "average_tries_remaining": average_tries_remaining,
        },
        "games": word_results,
    }


__all__ = [
    "HangmanOfflineAPI",
    "GuessStrategy",
    "frequency_guess_strategy",
    "bert_style_guess_strategy",
]
