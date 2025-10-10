"""Extract unique words from 20k.txt that are NOT in words_250000_train.txt"""

from pathlib import Path

# File paths
train_file = Path("data/words_250000_train.txt")
test_file = Path("data/20k.txt")
output_file = Path("data/20k_unique_not_in_train.txt")

# Read training words
print(f"Reading {train_file}...")
with open(train_file, "r", encoding="utf-8") as f:
    train_words = set(line.strip().lower() for line in f if line.strip())

# Read 20k test words
print(f"Reading {test_file}...")
with open(test_file, "r", encoding="utf-8") as f:
    test_words = set(line.strip().lower() for line in f if line.strip())

print(f"\nTotal words in {train_file.name}: {len(train_words)}")
print(f"Total words in {test_file.name}: {len(test_words)}")

# Find words in 20k that are NOT in training set
unique_test_words = test_words - train_words

print(f"\nWords in {test_file.name} that are NOT in training: {len(unique_test_words)}")
print(f"Words in both files: {len(test_words & train_words)}")

# Save unique words
print(f"\nSaving unique words to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    for word in sorted(unique_test_words):
        f.write(word + "\n")

print(f"\n✓ Done! Saved {len(unique_test_words)} unique words to: {output_file}")
