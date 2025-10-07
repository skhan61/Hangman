"""Pluggable guess strategies shared across Hangman implementations."""

from __future__ import annotations

import collections
import logging
import re
import string
from typing import Callable, List, Protocol, Tuple

logger = logging.getLogger(__name__)


class HangmanStrategyContext(Protocol):
    """Minimal protocol a guessing strategy expects from a context object."""

    current_dictionary: List[str]
    full_dictionary: List[str]
    incorrect_letters: set[str]
    guessed_letters: List[str]
    full_dictionary_common_letter_sorted: List[Tuple[str, int]]


GuessStrategy = Callable[[str, HangmanStrategyContext], str]


def _collect_candidates(
    masked_state: str, context: HangmanStrategyContext
) -> Tuple[str, List[str]]:
    masked_word = masked_state.replace(" ", "").lower()
    if not masked_word:
        raise ValueError("Masked word must be non-empty.")

    base_candidates = context.current_dictionary or context.full_dictionary
    pattern = re.compile("^" + masked_word.replace("_", ".") + "$")
    candidates = [
        candidate
        for candidate in base_candidates
        if len(candidate) == len(masked_word)
        and pattern.match(candidate)
        and context.incorrect_letters.isdisjoint(candidate)
    ]

    if not candidates:
        candidates = [
            word
            for word in context.full_dictionary
            if len(word) == len(masked_word)
            and context.incorrect_letters.isdisjoint(word)
        ]

    context.current_dictionary = candidates
    return masked_word, candidates


def _choose_letter(
    letter_counts: collections.Counter,
    context: HangmanStrategyContext,
    *,
    masked_state: str,
) -> str:
    guessed_letter_set = set(context.guessed_letters)

    for letter, _ in letter_counts.most_common():
        if letter not in guessed_letter_set:
            return letter

    for letter, _ in context.full_dictionary_common_letter_sorted:
        if letter not in guessed_letter_set:
            return letter

    for letter in string.ascii_lowercase:
        if letter not in guessed_letter_set:
            return letter

    raise RuntimeError(
        f"Unable to determine next guess letter for state '{masked_state}'."
    )


def _template_strategy(
    masked_state: str,
    context: HangmanStrategyContext,
    *,
    counter_builder: Callable[[str, List[str]], collections.Counter],
    log_suffix: str,
) -> str:
    masked_word, candidates = _collect_candidates(masked_state, context)
    letter_counts = counter_builder(masked_word, candidates)
    logger.debug(
        "%s strategy: %d candidates for state '%s'",
        log_suffix,
        len(candidates),
        masked_state,
    )
    return _choose_letter(letter_counts, context, masked_state=masked_state)


def _frequency_counter(_: str, candidates: List[str]) -> collections.Counter:
    return collections.Counter("".join(candidates))


def frequency_guess_strategy(masked_state: str, context: HangmanStrategyContext) -> str:
    """Default frequency-based guess strategy."""

    return _template_strategy(
        masked_state,
        context,
        counter_builder=_frequency_counter,
        log_suffix="Frequency",
    )


def _bert_counter(masked_word: str, candidates: List[str]) -> collections.Counter:
    position_counts = collections.defaultdict(collections.Counter)
    masked_positions = [idx for idx, char in enumerate(masked_word) if char == "_"]

    for word in candidates:
        for pos in masked_positions:
            if pos < len(word):
                position_counts[pos][word[pos]] += 1

    aggregated_counts = collections.Counter()
    for pos in masked_positions:
        aggregated_counts.update(position_counts[pos])
    logger.debug(
        "BERT counter: masked positions=%s, aggregated=%s",
        masked_positions,
        aggregated_counts,
    )
    return aggregated_counts


def bert_style_guess_strategy(
    masked_state: str, context: HangmanStrategyContext
) -> str:
    """BERT-style per-position frequency-based guess strategy."""

    return _template_strategy(
        masked_state,
        context,
        counter_builder=_bert_counter,
        log_suffix="BERT",
    )


__all__ = [
    "GuessStrategy",
    "HangmanStrategyContext",
    "frequency_guess_strategy",
    "bert_style_guess_strategy",
]
