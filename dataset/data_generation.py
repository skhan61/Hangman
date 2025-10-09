import random
import sys
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence
from multiprocessing import Pool, cpu_count

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional at runtime
    tqdm = None
from dataset.encoder_utils import DEFAULT_ALPHABET, CharacterEncoder

# Set NumExpr to use all cores
os.environ.setdefault('NUMEXPR_MAX_THREADS', str(cpu_count()))


ENCODER = CharacterEncoder(
    alphabet=DEFAULT_ALPHABET,
    mask_token="_",
    pad_token="<PAD>",
)

def read_words_list(file_path):
    """Read words from a text file and return as a list."""
    with open(file_path, "r") as f:
        words = [line.strip().upper() for line in f if line.strip()]
    return words


@dataclass
class TrajectoryStep:
    """One reveal step containing the visible state and outstanding targets."""

    state: List[Optional[str]]
    targets: Dict[int, str]
    guessed_letters: List[str]


def _make_step(
    state: List[Optional[str]],
    targets: Dict[int, str],
    guessed: Optional[Sequence[str]] = None,
) -> TrajectoryStep:
    """Create a trajectory step with defensive copies."""

    guessed_list = list(guessed) if guessed is not None else []
    return TrajectoryStep(
        state=list(state), targets=dict(targets), guessed_letters=guessed_list
    )


def _collect_targets(word: str, state: List[Optional[str]]) -> Dict[int, str]:
    """Return outstanding targets for the current masked positions."""

    return {idx: word[idx] for idx, value in enumerate(state) if value is None}

@dataclass
class Sample:
    """Sample ready for parquet serialization - NO padding, NO encoding.

    Stores raw trajectory data:
    - word: the original word
    - state: list of revealed/masked positions [None, 'A', None, ...]
    - targets: dict of {position: letter} for unrevealed positions
    - length: word length
    """
    word: str
    state: List[Optional[str]]
    targets: Dict[int, str]
    length: int


def reveal_word_step_by_step(word: str) -> List[TrajectoryStep]:
    """Reveal the word letter by letter in classic hangman fashion."""
    word = word.upper()
    word_length = len(word)

    # Get unique letters and shuffle for random reveal order
    unique_letters = list(set(word))
    random.shuffle(unique_letters)

    trajectory: List[TrajectoryStep] = []
    guessed = []
    state = [None] * word_length  # None = masked position

    # Reveal one letter at a time
    for letter in unique_letters:
        # Capture current state BEFORE revealing this letter
        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Now reveal this letter
        guessed.append(letter)
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

        # Stop if word is fully revealed
        if None not in state:
            break

    return trajectory


def reveal_word_position_left_to_right(word: str) -> List[TrajectoryStep]:
    """Reveal the word from left to right, one unique letter at a time."""
    word = word.upper()
    word_length = len(word)

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # Reveal positions from left to right
    for pos in range(word_length):
        # Skip if this position is already revealed
        if state[pos] is not None:
            continue

        # Capture current state BEFORE revealing this position
        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this position AND all other positions with the same letter
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_position_right_to_left(word: str) -> List[TrajectoryStep]:
    """Reveal the word from right to left, mirroring the left-to-right strategy."""
    word = word.upper()
    word_length = len(word)

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # Reveal positions from right to left
    for pos in range(word_length - 1, -1, -1):
        # Skip if this position is already revealed
        if state[pos] is not None:
            continue

        # Capture current state BEFORE revealing this position
        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this position AND all other positions with the same letter
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_position_random(word: str) -> List[TrajectoryStep]:
    """Reveal positions in a random order until the word is uncovered."""
    word = word.upper()
    word_length = len(word)

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # Create random position order
    positions = list(range(word_length))
    random.shuffle(positions)

    # Reveal positions in random order
    for pos in positions:
        # Skip if this position is already revealed
        if state[pos] is not None:
            continue

        # Capture current state BEFORE revealing this position
        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this position AND all other positions with the same letter
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_vowels_first(word: str) -> List[TrajectoryStep]:
    """Reveal vowels first, then consonants from left to right."""
    word = word.upper()
    word_length = len(word)
    vowels = set("AEIOU")

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # First pass: reveal vowels from left to right
    for pos in range(word_length):
        if state[pos] is not None:
            continue
        if word[pos] not in vowels:
            continue

        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this vowel at all positions
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    # Second pass: reveal consonants from left to right
    for pos in range(word_length):
        if state[pos] is not None:
            continue

        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this consonant at all positions
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_frequency_based(word: str) -> List[TrajectoryStep]:
    """Reveal letters following English letter frequency (most common first)."""
    word = word.upper()
    word_length = len(word)

    # English letter frequency order (most common first)
    frequency_order = "ETAOINSHRDLCUMWFGYPBVKJXQZ"

    # Get unique letters in the word, sorted by frequency
    unique_letters = list(set(word))
    unique_letters.sort(
        key=lambda x: frequency_order.index(x) if x in frequency_order else 99
    )

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    for letter in unique_letters:
        # Capture current state BEFORE revealing this letter
        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this letter at all positions
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_center_outward(word: str) -> List[TrajectoryStep]:
    """Reveal positions starting from the center and expanding outward."""
    word = word.upper()
    word_length = len(word)

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # Create center-outward position order
    center = word_length // 2
    positions = []

    # Start from center
    positions.append(center)

    # Alternate between left and right
    for offset in range(1, word_length):
        # Try right side first
        if center + offset < word_length:
            positions.append(center + offset)
        # Then left side
        if center - offset >= 0:
            positions.append(center - offset)

    # Reveal positions in center-outward order
    for pos in positions:
        if state[pos] is not None:
            continue

        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this position AND all other positions with the same letter
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_edges_first(word: str) -> List[TrajectoryStep]:
    """Reveal positions from the edges inward (outside-in)."""
    word = word.upper()
    word_length = len(word)

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # Create edges-first position order
    positions = []
    left = 0
    right = word_length - 1

    while left <= right:
        if left == right:
            positions.append(left)
        else:
            positions.append(left)
            positions.append(right)
        left += 1
        right -= 1

    # Reveal positions in edges-first order
    for pos in positions:
        if state[pos] is not None:
            continue

        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this position AND all other positions with the same letter
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_alternating(word: str) -> List[TrajectoryStep]:
    """Reveal alternating positions (even indices first, then odd)."""
    word = word.upper()
    word_length = len(word)

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # Create alternating position order: even positions first, then odd
    positions = list(range(0, word_length, 2)) + list(range(1, word_length, 2))

    # Reveal positions in alternating order
    for pos in positions:
        if state[pos] is not None:
            continue

        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this position AND all other positions with the same letter
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_rare_letters_first(word: str) -> List[TrajectoryStep]:
    """Reveal letters from rarest to most common in English."""
    word = word.upper()
    word_length = len(word)

    # English letter rarity order (rarest first)
    rarity_order = "ZQXJKVBPGWFYMUCLDRHSNITOATE"

    # Get unique letters in the word, sorted by rarity
    unique_letters = list(set(word))
    unique_letters.sort(
        key=lambda x: rarity_order.index(x) if x in rarity_order else 99
    )

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    for letter in unique_letters:
        # Capture current state BEFORE revealing this letter
        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this letter at all positions
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_consonants_first(word: str) -> List[TrajectoryStep]:
    """Reveal consonants first, then vowels."""
    word = word.upper()
    word_length = len(word)
    vowels = set("AEIOU")

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # First pass: reveal consonants from left to right
    for pos in range(word_length):
        if state[pos] is not None:
            continue
        if word[pos] in vowels:
            continue

        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this consonant at all positions
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    # Second pass: reveal vowels from left to right
    for pos in range(word_length):
        if state[pos] is not None:
            continue

        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this vowel at all positions
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_patterns(word: str) -> List[TrajectoryStep]:
    """Reveal positions guided by common English word patterns."""
    word = word.upper()
    word_length = len(word)

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # Define priority positions: endings first, then beginnings, then rest
    priority_positions = []

    # Add ending positions (last 3 letters)
    if word_length >= 3:
        priority_positions.extend([word_length - 3, word_length - 2, word_length - 1])
    elif word_length == 2:
        priority_positions.extend([word_length - 2, word_length - 1])
    elif word_length == 1:
        priority_positions.append(0)

    # Add beginning positions (first 2 letters if not already added)
    if word_length > 3:
        priority_positions.extend([0, 1])

    # Add remaining middle positions
    for i in range(word_length):
        if i not in priority_positions:
            priority_positions.append(i)

    # Reveal positions following pattern priority
    for pos in priority_positions:
        if state[pos] is not None:
            continue

        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this position AND all other positions with the same letter
        letter = word[pos]
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def reveal_word_random_percentage(word: str) -> List[TrajectoryStep]:
    """Reveal letters in a random order, mimicking percentage-based masking."""
    word = word.upper()
    word_length = len(word)

    trajectory: List[TrajectoryStep] = []
    state = [None] * word_length

    # Get unique letters and shuffle for random reveal
    unique_letters = list(set(word))
    random.shuffle(unique_letters)

    # Calculate how many letters to reveal at each step (roughly)
    # We'll reveal letters one at a time but create roughly percentage-based distribution
    total_unique = len(unique_letters)

    # Reveal letters one by one (each reveal reduces masked percentage)
    for letter in unique_letters:
        # Capture current state BEFORE revealing this letter
        targets = _collect_targets(word, state)
        trajectory.append(_make_step(state, targets))

        # Reveal this letter at all positions
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

    return trajectory


def _get_strategy_function(strategy_name: str):
    """Map strategy name to function (module-level for pickling)."""
    strategy_map = {
        "letter_based": reveal_word_step_by_step,
        "left_to_right": reveal_word_position_left_to_right,
        "right_to_left": reveal_word_position_right_to_left,
        "random_position": reveal_word_position_random,
        "vowels_first": reveal_word_vowels_first,
        "frequency_based": reveal_word_frequency_based,
        "center_outward": reveal_word_center_outward,
        "edges_first": reveal_word_edges_first,
        "alternating": reveal_word_alternating,
        "rare_letters_first": reveal_word_rare_letters_first,
        "consonants_first": reveal_word_consonants_first,
        "word_patterns": reveal_word_patterns,
        "random_percentage": reveal_word_random_percentage,
    }
    return strategy_map[strategy_name]


def _process_word_with_strategies(args):
    """Helper function to process a single word with all strategies (for multiprocessing)."""
    word, strategies = args
    word_samples = []

    for strategy_name in strategies:
        strategy_func = _get_strategy_function(strategy_name)
        trajectory_steps = strategy_func(word)

        for step in trajectory_steps:
            state_list = list(step.state)
            targets_dict = dict(step.targets)

            sample = Sample(
                word=word,
                state=state_list,
                targets=targets_dict,
                length=len(state_list),
            )

            word_samples.append(sample)

    return word_samples


def generate_full_dataset(
    words: Sequence[str],
    strategies: Optional[Sequence[str]] = None,
    parallel: bool = True,
    num_workers: Optional[int] = None,
) -> List[Sample]:
    """Generate trajectory samples and encoded fields for each strategy."""

    if strategies is None:
        strategies = ["letter_based"]
    elif not isinstance(strategies, (list, tuple)):
        strategies = list(strategies)

    # Map strategy names to functions
    strategy_functions = {
        "letter_based": reveal_word_step_by_step,
        "left_to_right": reveal_word_position_left_to_right,
        "right_to_left": reveal_word_position_right_to_left,
        "random_position": reveal_word_position_random,
        "vowels_first": reveal_word_vowels_first,
        "frequency_based": reveal_word_frequency_based,
        "center_outward": reveal_word_center_outward,
        "edges_first": reveal_word_edges_first,
        "alternating": reveal_word_alternating,
        "rare_letters_first": reveal_word_rare_letters_first,
        "consonants_first": reveal_word_consonants_first,
        "word_patterns": reveal_word_patterns,
        "random_percentage": reveal_word_random_percentage,
    }

    # Validate strategies
    for strategy in strategies:
        if strategy not in strategy_functions:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid options: {list(strategy_functions.keys())}"
            )
        
    # print(type(Sample))

    records: List[Sample] = []

    # print(type(records))

    iterator = (
        tqdm(
            words,
            desc="Generating game states",
            unit="word",
            total=len(words),
            file=sys.stdout,
        )
        if tqdm
        else words
    )

    if parallel and len(words) > 100:  # Only parallelize for large datasets
        # Prepare arguments for parallel processing
        if num_workers is None:
            num_workers = cpu_count()

        args_list = [(word, strategies) for word in words]

        # Use multiprocessing pool with progress bar
        with Pool(processes=num_workers) as pool:
            if tqdm:
                # Process with progress bar
                word_results = list(
                    tqdm(
                        pool.imap(_process_word_with_strategies, args_list),
                        desc="Generating game states (parallel)",
                        unit="word",
                        total=len(words),
                        file=sys.stdout,
                    )
                )
            else:
                word_results = pool.map(_process_word_with_strategies, args_list)

        # Flatten results
        for word_samples in word_results:
            records.extend(word_samples)

    else:
        # Sequential processing (for small datasets or when parallel=False)
        for word in iterator:
            # Apply each strategy to this word
            for strategy_name in strategies:
                strategy_func = strategy_functions[strategy_name]
                trajectory_steps = strategy_func(word)

                # Add word and strategy metadata to each state
                for step in trajectory_steps:
                    state_list = list(step.state)
                    targets_dict = dict(step.targets)

                    sample = Sample(
                        word=word,
                        state=state_list,
                        targets=targets_dict,
                        length=len(state_list),
                    )

                    records.append(sample)

        if hasattr(iterator, "close"):
            iterator.close()

    return records

