"""Find unique words between two word lists and save them separately."""

from pathlib import Path

# File paths
file1_path = Path("data/words_250000_train.txt")
file2_path = Path("data/words_alpha.txt")

# Read words from both files
print(f"Reading {file1_path}...")
with open(file1_path, "r", encoding="utf-8") as f:
    words1 = set(line.strip().lower() for line in f if line.strip())

print(f"Reading {file2_path}...")
with open(file2_path, "r", encoding="utf-8") as f:
    words2 = set(line.strip().lower() for line in f if line.strip())

print(f"Total words in {file1_path.name}: {len(words1)}")
print(f"Total words in {file2_path.name}: {len(words2)}")

# Find unique words in each file
unique_to_file1 = words1 - words2  # Words only in file1
unique_to_file2 = words2 - words1  # Words only in file2
common_words = words1 & words2     # Words in both files

print(f"\nUnique to {file1_path.name}: {len(unique_to_file1)}")
print(f"Unique to {file2_path.name}: {len(unique_to_file2)}")
print(f"Common words: {len(common_words)}")

# Save unique words to separate files
output1 = Path("data/unique_to_250000_train.txt")
output2 = Path("data/unique_to_alpha.txt")

print(f"\nSaving unique words to {output1}...")
with open(output1, "w", encoding="utf-8") as f:
    for word in sorted(unique_to_file1):
        f.write(word + "\n")

print(f"Saving unique words to {output2}...")
with open(output2, "w", encoding="utf-8") as f:
    for word in sorted(unique_to_file2):
        f.write(word + "\n")

print(f"\nDone!")
print(f"Unique words from {file1_path.name} saved to: {output1}")
print(f"Unique words from {file2_path.name} saved to: {output2}")
