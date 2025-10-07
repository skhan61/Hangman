import random
import sys

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional at runtime
    tqdm = None


def read_words_list(file_path):
    """Read words from a text file and return as a list."""
    with open(file_path, 'r') as f:
        words = [line.strip().upper() for line in f if line.strip()]
    return words


def reveal_word_step_by_step(word):
    """
    Take a word and reveal it letter by letter.
    At each step, capture the current state and targets for each masked position.

    Args:
        word: string, the target word (will be uppercased)

    Returns:
        List of dicts, each containing:
            - state: list of revealed letters (None for masked)
            - targets: dict mapping position -> target letter for masked positions
            - guessed: list of letters guessed so far
    """
    word = word.upper()
    word_length = len(word)

    # Get unique letters and shuffle for random reveal order
    unique_letters = list(set(word))
    random.shuffle(unique_letters)

    trajectory = []
    guessed = []
    state = [None] * word_length  # None = masked position

    # Reveal one letter at a time
    for letter in unique_letters:
        # Capture current state BEFORE revealing this letter
        targets = {}
        for i in range(word_length):
            if state[i] is None:  # If position is masked
                targets[i] = word[i]  # Target is the actual letter

        turn_data = {
            'state': state.copy(),
            'targets': targets,
            'guessed': guessed.copy()
        }
        trajectory.append(turn_data)

        # Now reveal this letter
        guessed.append(letter)
        for i in range(word_length):
            if word[i] == letter:
                state[i] = letter

        # Stop if word is fully revealed
        if None not in state:
            break

    return trajectory


def generate_full_dataset(words):
    """
    Generate dataset from all words.
    Each word produces multiple states (one per unique letter).

    Returns:
        List of dicts, where each dict is one game state
    """
    all_states = []

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

    for word in iterator:
        # Get trajectory for this word
        trajectory = reveal_word_step_by_step(word)

        # Add word to each state and append to dataset
        for turn in trajectory:
            turn['word'] = word
            all_states.append(turn)

    if hasattr(iterator, "close"):
        iterator.close()

    return all_states


def generate_dataset_parquet(
    words_file,
    output_path,
    *,
    words=None,
    max_words=None,
    force=False,
    show_summary=True,
):
    """Generate hangman game states and write them to a parquet file.

    Args:
        words_file: Source text file containing candidate words.
        output_path: Destination parquet file path.
        words: Optional pre-loaded list of words. If provided, ``words_file`` is
            only used in log messages.
        max_words: Optional upper bound on number of words to use.
        force: If True, regenerate even when the parquet already exists.
        show_summary: Print dataset statistics when True.

    Returns:
        pandas.DataFrame containing the generated states.
    """

    import pandas as pd
    from pathlib import Path

    output_path = Path(output_path)
    if words is None:
        words = read_words_list(words_file)

    if max_words:
        words = words[:max_words]

    if output_path.exists() and not force:
        if show_summary:
            print(
                f"Using existing dataset at {output_path} (skip generation)",
                flush=True,
            )
        return pd.read_parquet(output_path)

    if show_summary:
        print(
            f"Generating dataset from {len(words)} words using {words_file}...",
            flush=True,
        )

    all_states = generate_full_dataset(words)

    if show_summary:
        print(
            f"Generated {len(all_states)} total game states", flush=True
        )

    df = pd.DataFrame(all_states)
    df['targets'] = df['targets'].apply(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    if show_summary:
        print(f"Saved dataset to {output_path}", flush=True)

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # Test mode - show examples
        print("TEST MODE: Showing examples\n")
        test_words = ["CAT", "APPLE", "MISSISSIPPI"]

        for test_word in test_words:
            print(f"{'='*60}")
            print(f"Testing with word: {test_word}")
            print(f"Word length: {len(test_word)}, Unique letters: {len(set(test_word))}")
            print(f"{'='*60}\n")

            trajectory = reveal_word_step_by_step(test_word)

            print(f"Generated {len(trajectory)} turns:\n")
            for i, turn in enumerate(trajectory):
                print(f"Turn {i+1}:")
                print(f"  State:   {turn['state']}")
                print(f"  Targets: {turn['targets']}")
                print(f"  Guessed: {turn['guessed']}")
                print()
            print()

        print("\nTo generate full dataset, run:")
        print("  python simulation/data_generation.py generate <num_words>")
        print("Example:")
        print("  python simulation/data_generation.py generate 1000")

    elif sys.argv[1] == "generate":
        # Generate mode
        import pandas as pd

        # Get number of words to process
        max_words = int(sys.argv[2]) if len(sys.argv) > 2 else None

        # Load words
        words_file = "data/words_250000_train.txt"
        print(f"Loading words from {words_file}...", flush=True)
        words = read_words_list(words_file)

        if max_words:
            words = words[:max_words]

        print(f"Loaded {len(words)} words\n", flush=True)

        output_file = f"data/dataset_{len(words)}words.parquet"

        df = generate_dataset_parquet(
            words_file,
            output_file,
            words=words,
            force=True,
            show_summary=True,
        )

        print(f"DataFrame shape: {df.shape}", flush=True)
        print(f"Columns: {list(df.columns)}", flush=True)

        print("\nFirst few rows:", flush=True)
        print(df.head().to_string(index=False), flush=True)

        # Read the parquet we just wrote to confirm contents
        print("\nReading back from parquet...", flush=True)
        reloaded_df = pd.read_parquet(output_file)
        print("\nFirst few rows reloaded from parquet:", flush=True)
        print(reloaded_df.head().to_string(index=False), flush=True)
