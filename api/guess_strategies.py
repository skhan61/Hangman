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


def positional_frequency_strategy(
    masked_state: str, context: HangmanStrategyContext
) -> str:
    """Position-aware frequency strategy.

    Counts letter frequencies only at masked positions (not whole word).
    More accurate than simple frequency counting.
    """

    return _template_strategy(
        masked_state,
        context,
        counter_builder=_bert_counter,
        log_suffix="Positional",
    )


# Keep old name for backward compatibility
def bert_style_guess_strategy(
    masked_state: str, context: HangmanStrategyContext
) -> str:
    """Deprecated: Use positional_frequency_strategy instead."""
    return positional_frequency_strategy(masked_state, context)


def _ngram_counter(masked_word: str, candidates: List[str]) -> collections.Counter:
    """Count letters using n-gram context (bigrams and trigrams).

    Uses surrounding revealed letters to predict masked positions.
    For example, if we see 'Q_', we know next letter is likely 'U'.
    """
    letter_scores = collections.Counter()
    masked_positions = [idx for idx, char in enumerate(masked_word) if char == "_"]

    if not masked_positions or not candidates:
        return letter_scores

    # For each masked position, look at context (before and after)
    for pos in masked_positions:
        position_scores = collections.Counter()

        # Get context: 1 letter before and 1 letter after
        before = masked_word[pos - 1] if pos > 0 else None
        after = masked_word[pos + 1] if pos < len(masked_word) - 1 else None

        # Count letters that appear in candidates at this position with this context
        for word in candidates:
            if pos >= len(word):
                continue

            letter = word[pos]

            # Check if context matches (bigram scoring)
            context_match = True

            if before is not None and before != "_":
                # Before context must match
                if pos == 0 or word[pos - 1] != before:
                    context_match = False

            if after is not None and after != "_":
                # After context must match
                if pos >= len(word) - 1 or word[pos + 1] != after:
                    context_match = False

            if context_match:
                # Weight by how specific the context is
                # More context = higher confidence
                specificity = 1.0
                if before is not None and before != "_":
                    specificity += 1.0  # Bigram bonus
                if after is not None and after != "_":
                    specificity += 1.0  # Bigram bonus

                position_scores[letter] += specificity

        # Aggregate scores from this position
        letter_scores.update(position_scores)

    return letter_scores


def ngram_guess_strategy(masked_state: str, context: HangmanStrategyContext) -> str:
    """N-gram based guess strategy using bigram/trigram patterns.

    Uses context of revealed letters around masked positions to make better predictions.
    For example: 'Q_' strongly suggests 'U', '_H_' often suggests 'T' or 'E'.
    """
    return _template_strategy(
        masked_state,
        context,
        counter_builder=_ngram_counter,
        log_suffix="N-gram",
    )


def _pattern_matching_counter(masked_word: str, candidates: List[str]) -> collections.Counter:
    """Pattern matching: only count letters from words matching exact pattern.

    Most precise frequency counting - only uses words that match the current pattern.
    """
    # _collect_candidates already filters by pattern, so just count letters
    return collections.Counter("".join(candidates))


def pattern_matching_strategy(masked_state: str, context: HangmanStrategyContext) -> str:
    """Pattern matching strategy - most precise dictionary filtering.

    Matches exact pattern with dictionary and counts letters only from matching words.
    Example: '_A_E' matches 'CAKE', 'SAFE', 'WAVE' → count letters from only those.
    """
    return _template_strategy(
        masked_state,
        context,
        counter_builder=_pattern_matching_counter,
        log_suffix="Pattern",
    )


def entropy_strategy(masked_state: str, context: HangmanStrategyContext) -> str:
    """Entropy-based strategy - maximize information gain per guess.

    Picks letter that splits candidate set closest to 50/50, giving maximum information.
    Uses Shannon entropy to measure uncertainty reduction.
    """
    import math

    masked_word, candidates = _collect_candidates(masked_state, context)

    if not candidates:
        # Fallback to frequency if no candidates
        return frequency_guess_strategy(masked_state, context)

    guessed_set = set(context.guessed_letters)
    best_letter = None
    best_entropy = -1.0

    # Calculate entropy for each unguessed letter
    for letter in string.ascii_lowercase:
        if letter in guessed_set:
            continue

        # Count how many candidates have this letter
        with_letter = sum(1 for word in candidates if letter in word)
        without_letter = len(candidates) - with_letter
        total = len(candidates)

        if with_letter == 0 or without_letter == 0:
            # No information gain if all or none have this letter
            entropy = 0.0
        else:
            # Calculate Shannon entropy
            p_with = with_letter / total
            p_without = without_letter / total
            entropy = -(p_with * math.log2(p_with) + p_without * math.log2(p_without))

        if entropy > best_entropy:
            best_entropy = entropy
            best_letter = letter

    if best_letter is None:
        # Fallback to frequency
        return frequency_guess_strategy(masked_state, context)

    logger.debug("Entropy strategy: picked '%s' with entropy %.3f", best_letter, best_entropy)
    return best_letter


def vowel_consonant_strategy(masked_state: str, context: HangmanStrategyContext) -> str:
    """Vowel-consonant strategy - guess vowels first, then common consonants.

    Vowels (A, E, I, O, U) provide maximum pattern information in English.
    After vowels, guess consonants by frequency.
    """
    masked_word, candidates = _collect_candidates(masked_state, context)
    guessed_set = set(context.guessed_letters)

    # Vowels in frequency order
    vowels = ['e', 'a', 'i', 'o', 'u']

    # Try vowels first
    for vowel in vowels:
        if vowel not in guessed_set:
            # Check if any candidates have this vowel
            if any(vowel in word for word in candidates):
                logger.debug("Vowel-consonant strategy: guessing vowel '%s'", vowel)
                return vowel

    # All vowels guessed, use frequency for consonants
    logger.debug("Vowel-consonant strategy: all vowels guessed, using frequency")
    return frequency_guess_strategy(masked_state, context)


def _length_aware_counter(masked_word: str, candidates: List[str]) -> collections.Counter:
    """Length-aware counting with different strategies per word length."""
    word_length = len(masked_word)

    # Use position-aware counting but weight by word length patterns
    letter_counts = collections.Counter("".join(candidates))

    # Boost certain letters based on length
    if word_length <= 4:
        # Short words: boost common short-word letters
        for letter in ['a', 'e', 'i', 'o', 't', 'h', 's']:
            if letter in letter_counts:
                letter_counts[letter] = int(letter_counts[letter] * 1.2)
    elif word_length >= 8:
        # Long words: boost letters common in longer words
        for letter in ['i', 'n', 'g', 't', 'r', 'l', 'c']:
            if letter in letter_counts:
                letter_counts[letter] = int(letter_counts[letter] * 1.2)

    return letter_counts


def length_aware_strategy(masked_state: str, context: HangmanStrategyContext) -> str:
    """Length-aware strategy - adapt guessing based on word length.

    Short words (≤4): Favor common short-word letters (A, E, I, O, T, H, S)
    Long words (≥8): Favor letters common in longer words (I, N, G, T, R, L, C)
    """
    return _template_strategy(
        masked_state,
        context,
        counter_builder=_length_aware_counter,
        log_suffix="LengthAware",
    )


def suffix_prefix_strategy(masked_state: str, context: HangmanStrategyContext) -> str:
    """Suffix/prefix strategy - detect common word endings and beginnings.

    Looks for patterns like:
    - Endings: _ING, _TION, _LY, _ED, _ER, _EST
    - Beginnings: UN_, RE_, IN_, DIS_
    """
    masked_word, candidates = _collect_candidates(masked_state, context)
    guessed_set = set(context.guessed_letters)
    masked_word_lower = masked_word.lower()

    # Check for common suffix patterns
    suffix_hints = {
        '_ng': ['i'],  # likely ING
        '__ng': ['i'],  # likely _ING
        '_ion': ['t', 'a'],  # likely TION, SION
        '__ion': ['t', 's', 'a'],  # likely TION, SION
        '_ly': ['l', 'i'],  # likely LLY, ILY
        '_ed': ['t', 'r', 'n'],  # likely TED, RED, NED
        '_er': ['t', 'n', 'k'],  # likely TER, NER, KER
    }

    # Check for common prefix patterns
    prefix_hints = {
        'un_': ['d', 'i', 't'],  # likely UND, UNI, UNT
        're_': ['a', 'd', 'p'],  # likely REA, RED, REP
        'in_': ['t', 'g', 'f'],  # likely INT, ING, INF
        'dis_': ['a', 't', 'c'],  # likely DISA, DIST, DISC
    }

    # Check suffix patterns
    for pattern, hint_letters in suffix_hints.items():
        if masked_word_lower.endswith(pattern):
            for letter in hint_letters:
                if letter not in guessed_set:
                    logger.debug("Suffix-prefix strategy: detected suffix '%s', guessing '%s'", pattern, letter)
                    return letter

    # Check prefix patterns
    for pattern, hint_letters in prefix_hints.items():
        if masked_word_lower.startswith(pattern):
            for letter in hint_letters:
                if letter not in guessed_set:
                    logger.debug("Suffix-prefix strategy: detected prefix '%s', guessing '%s'", pattern, letter)
                    return letter

    # No pattern detected, use frequency
    logger.debug("Suffix-prefix strategy: no pattern detected, using frequency")
    return frequency_guess_strategy(masked_state, context)


def ensemble_strategy(masked_state: str, context: HangmanStrategyContext) -> str:
    """Ensemble strategy - combines multiple heuristics with voting.

    Combines:
    - Frequency (25%)
    - Positional frequency (25%)
    - N-gram (25%)
    - Entropy (25%)

    Each strategy votes, and letter with highest combined score wins.
    """
    guessed_set = set(context.guessed_letters)

    # Collect scores from each strategy
    strategy_scores = collections.defaultdict(float)
    weights = {
        'frequency': 0.25,
        'positional': 0.25,
        'ngram': 0.25,
        'entropy': 0.25,
    }

    # Get candidates for scoring
    masked_word, candidates = _collect_candidates(masked_state, context)

    # 1. Frequency strategy scores
    freq_counts = _frequency_counter(masked_word, candidates)
    max_freq = max(freq_counts.values()) if freq_counts else 1
    for letter, count in freq_counts.items():
        if letter not in guessed_set:
            strategy_scores[letter] += weights['frequency'] * (count / max_freq)

    # 2. Positional frequency scores
    pos_counts = _bert_counter(masked_word, candidates)
    max_pos = max(pos_counts.values()) if pos_counts else 1
    for letter, count in pos_counts.items():
        if letter not in guessed_set:
            strategy_scores[letter] += weights['positional'] * (count / max_pos)

    # 3. N-gram scores
    ngram_counts = _ngram_counter(masked_word, candidates)
    max_ngram = max(ngram_counts.values()) if ngram_counts else 1
    for letter, count in ngram_counts.items():
        if letter not in guessed_set:
            strategy_scores[letter] += weights['ngram'] * (count / max_ngram)

    # 4. Entropy scores
    import math
    for letter in string.ascii_lowercase:
        if letter in guessed_set:
            continue

        with_letter = sum(1 for word in candidates if letter in word)
        without_letter = len(candidates) - with_letter
        total = len(candidates) if candidates else 1

        if with_letter > 0 and without_letter > 0:
            p_with = with_letter / total
            p_without = without_letter / total
            entropy = -(p_with * math.log2(p_with) + p_without * math.log2(p_without))
            strategy_scores[letter] += weights['entropy'] * entropy

    # Pick letter with highest combined score
    if not strategy_scores:
        return frequency_guess_strategy(masked_state, context)

    best_letter = max(strategy_scores.items(), key=lambda x: x[1])[0]
    logger.debug("Ensemble strategy: picked '%s' with score %.3f", best_letter, strategy_scores[best_letter])
    return best_letter


def _calculate_information_gain(
    letter: str, dictionary: List[str], masked_word: str
) -> float:
    """Calculate information gain for guessing a letter.

    Information gain measures how much a guess reduces uncertainty about the word.
    Uses Shannon entropy to measure the information gained from a binary outcome.

    Args:
        letter: Letter to evaluate ('a' to 'z')
        dictionary: Current candidate words
        masked_word: Current masked state (e.g., "_pp_e")

    Returns:
        Information gain score (0 to 1, where 1 is maximum information)
    """
    import math

    if not dictionary:
        return 0.0

    # Count candidates WITH this letter vs WITHOUT
    with_letter = 0
    without_letter = 0

    for word in dictionary:
        if len(word) != len(masked_word):
            continue
        if letter in word:
            with_letter += 1
        else:
            without_letter += 1

    total = with_letter + without_letter
    if total == 0:
        return 0.0

    # Binary entropy: measures information from binary outcome
    p_with = with_letter / total
    p_without = without_letter / total

    if p_with == 0 or p_without == 0:
        return 0.0

    # Shannon entropy: H = -p*log2(p) - (1-p)*log2(1-p)
    # Maximum (1.0) when p = 0.5 (50/50 split gives most information)
    entropy = -(p_with * math.log2(p_with) + p_without * math.log2(p_without))

    return entropy


def neural_guess_strategy(
    masked_state: str, context: HangmanStrategyContext, model=None
) -> str:
    """Neural network-based guess strategy using trained model (pure neural, no info gain).

    Args:
        masked_state: Current word state, e.g., "_ p p _ e "
        context: Strategy context with dictionary and guessed letters
        model: Trained PyTorch model (required)

    Raises:
        ValueError: If model is None
    """
    import torch
    from dataset.observation_builder import build_model_inputs

    if model is None:
        raise ValueError("Neural strategy requires a trained model")

    # Build model inputs from masked state
    masked_word = masked_state.replace(" ", "")

    state_tensor, length_tensor = build_model_inputs(masked_word)
    device = next(model.parameters()).device
    state_tensor = state_tensor.to(device)
    length_tensor = length_tensor.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(state_tensor, length_tensor)

    # Find masked positions
    masked_positions = [i for i, c in enumerate(masked_word) if c == "_"]

    # Aggregate logits across masked positions: [26]
    aggregated_logits = logits[0, masked_positions, :].sum(dim=0)

    # Get sorted letter indices by score
    sorted_indices = torch.argsort(aggregated_logits, descending=True)

    # Find first unguessed letter
    guessed_set = set(context.guessed_letters)
    for idx in sorted_indices:
        letter = chr(ord("a") + idx.item())
        if letter not in guessed_set:
            logger.debug(
                "Neural strategy: predicted '%s' for state '%s' (score=%.3f)",
                letter,
                masked_state,
                aggregated_logits[idx].item(),
            )
            return letter

    raise RuntimeError(f"All letters guessed for state '{masked_state}'")


def neural_info_gain_strategy(
    masked_state: str, context: HangmanStrategyContext, model=None, info_gain_weight=2.0
) -> str:
    """Neural network + information gain boost strategy.

    Combines neural model predictions with dictionary-based information gain
    to make more strategic guesses.

    Args:
        masked_state: Current word state, e.g., "_ p p _ e "
        context: Strategy context with dictionary and guessed letters
        model: Trained PyTorch model (required)
        info_gain_weight: Weight for information gain term (default: 2.0)

    Raises:
        ValueError: If model is None
    """
    import torch
    from dataset.observation_builder import build_model_inputs

    if model is None:
        raise ValueError("Neural strategy requires a trained model")

    # Build model inputs from masked state
    masked_word = masked_state.replace(" ", "").lower()

    state_tensor, length_tensor = build_model_inputs(masked_word)
    device = next(model.parameters()).device
    state_tensor = state_tensor.to(device)
    length_tensor = length_tensor.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(state_tensor, length_tensor)

    # Find masked positions
    masked_positions = [i for i, c in enumerate(masked_word) if c == "_"]

    # Aggregate logits across masked positions: [26]
    aggregated_logits = logits[0, masked_positions, :].sum(dim=0)

    # ADD INFORMATION GAIN BOOST
    # For each letter, calculate how much it reduces dictionary uncertainty
    for letter_idx in range(26):
        letter = chr(ord("a") + letter_idx)

        # Skip already guessed letters
        if letter in context.guessed_letters:
            continue

        # Calculate information gain from dictionary
        info_gain = _calculate_information_gain(
            letter, context.current_dictionary, masked_word
        )

        # Boost neural logit by information gain
        aggregated_logits[letter_idx] += info_gain_weight * info_gain

    # Get sorted letter indices by combined score
    sorted_indices = torch.argsort(aggregated_logits, descending=True)

    # Find first unguessed letter
    guessed_set = set(context.guessed_letters)
    for idx in sorted_indices:
        letter = chr(ord("a") + idx.item())
        if letter not in guessed_set:
            logger.debug(
                "Neural+InfoGain strategy: predicted '%s' for state '%s' (score=%.3f)",
                letter,
                masked_state,
                aggregated_logits[idx].item(),
            )
            return letter

    raise RuntimeError(f"All letters guessed for state '{masked_state}'")


__all__ = [
    "GuessStrategy",
    "HangmanStrategyContext",
    # Heuristic strategies
    "frequency_guess_strategy",
    "positional_frequency_strategy",
    "bert_style_guess_strategy",  # Deprecated alias
    "ngram_guess_strategy",
    "pattern_matching_strategy",
    "entropy_strategy",
    "vowel_consonant_strategy",
    "length_aware_strategy",
    "suffix_prefix_strategy",
    "ensemble_strategy",
    # Neural strategies
    "neural_guess_strategy",
    "neural_info_gain_strategy",
]
