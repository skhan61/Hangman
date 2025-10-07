# Hangman Solver - Position-wise BERT-Style Approach

## Current Status

This project implements a **position-wise character prediction model** for solving Hangman, inspired by BERT's masked language modeling. The model is currently in development with the core architecture and data pipeline completed.

---

## My Approach: Why Position-wise Prediction?

### The Problem with Traditional Approaches

Most Hangman solvers use **frequency-based heuristics**:
- Count letter frequencies in corpus
- Filter dictionary by word length and pattern
- Guess most frequent letter

**Limitation**: Picks ONE letter for the entire word, ignoring position-specific context.

### My Solution: BERT-Style Position-wise Prediction

Instead of predicting a single letter, my model predicts **what letter belongs in each masked position**:

```
Traditional:  "_pp_e" → guess 'a' (most frequent overall)

My approach:  "_pp_e" →
              Position 0: P(a)=0.95, P(o)=0.02, ...
              Position 3: P(l)=0.92, P(k)=0.05, ...
              → Aggregate → guess 'a' (highest across all positions)
```

**Model Output**: `[batch_size, word_length, 26]`
- For each position, probability distribution over 26 letters
- Like BERT predicting masked tokens in a sentence

---

## Architecture

### BiLSTM for Bidirectional Context

```
Input: Masked word [batch, word_len]
       e.g., [MASK, p, p, MASK, e]  (encoded as integers)
       ↓
Character Embedding [batch, word_len, 256]
       ↓
BiLSTM [batch, word_len, 512]  (bidirectional → 256*2)
       ↓
Linear Classifier [batch, word_len, 26]
       ↓
For each position: probability distribution over a-z
```

**Why BiLSTM?**
- Captures **left context** (letters before position)
- Captures **right context** (letters after position)
- Similar to BERT's bidirectional attention, but using LSTM

---

## Data Pipeline - Critical Encoding Consistency

### The Encoding Problem I Solved

**Initial mistake**: Had different encoding in training vs inference
- Training: `'a'=0, 'b'=1, ..., 'z'=25`
- Inference: `'a'=1, 'b'=2, ..., 'z'=26` (WRONG!)

**Solution**: Shared encoder for both

```python
# dataset/encoder_utils.py - SINGLE SOURCE OF TRUTH
DEFAULT_ALPHABET = ('a', 'b', ..., 'z')  # lowercase

class CharacterEncoder:
    'a' or 'A' → 0
    'b' or 'B' → 1
    ...
    'z' or 'Z' → 25
    '_' (mask) → 26
    '<PAD>'    → 27
```

Both training (`dataset/`) and inference (`dataset/observation_builder.py`) import this same encoder.

### Dataset Structure

```python
{
    'state': [26, 15, 15, 26, 4],      # Encoded: [_, p, p, _, e]
    'guessed': [0,0,0,0,1,...,0,1,0],  # Binary vector: guessed 'e' and 'p'
    'targets': {0: 0, 3: 11},          # Ground truth: pos 0='a'(0), pos 3='l'(11)
    'length': 5
}
```

---

## Project Structure (Current)

```
Hangman/
├── dataset/                  # Data encoding & loading
│   ├── encoder_utils.py     ✅ Shared encoder (training + inference)
│   ├── hangman_dataset.py   ✅ PyTorch dataset
│   ├── data_module.py       ✅ Lightning data module
│   └── observation_builder.py ✅ Inference encoding
│
├── models/                   # Model architectures
│   ├── architectures/       ✅ BiLSTM
│   ├── lightning_module.py  ✅ Training wrapper
│   └── metrics.py           ✅ Metrics
│
├── api/                      # API & testing
│   ├── hangman_api.py       ✅ Online API
│   ├── offline_api.py       🚧 Local testing
│   └── guess_strategies.py  ✅ Strategies
│   └── test.py              🚧 Test scripts
│
├── simulation/
│   └── data_generation.py   ✅ Generate training data
│
├── hangman_callback/
│   └── callback.py           ✅ Training callbacks
│
├── data/
│   ├── words_250000_train.txt ✅ Corpus
│   └── dataset_250000words.parquet ✅ Preprocessed
│
└── train.py                  ✅ Training script
```

✅ = Complete
🚧 = In progress
❌ = Not started

---

## What's Working

1. **Data encoding pipeline**: Shared encoder ensures training/inference consistency
2. **BiLSTM architecture**: Position-wise prediction `[batch, word_len, 26]`
3. **Frequency baseline**: `FreqGuesser` class for comparison
4. **Game engine**: Can simulate games word-by-word
5. **Modular testing**: Same game engine works with freq baseline OR neural model

---

## What's Next

1. **Train the BiLSTM model** on full dataset
2. **Implement inference logic**:
   - Get position-wise logits from model
   - Aggregate across masked positions
   - Select best letter to guess
3. **Test against baseline**:
   - Frequency baseline: ~65% win rate
   - BiLSTM model: TBD
4. **Offline API testing**: Complete `api/offline_api.py` for local testing

---

## Key Innovation vs Baseline

| Aspect | Frequency Baseline | My BiLSTM Approach |
|--------|-------------------|-------------------|
| Prediction | One letter for whole word | Letter for each position |
| Context | Word length + pattern | Full bidirectional context |
| Output | Single letter | `[word_len, 26]` distribution |
| Approach | Hand-crafted heuristics | Learned from data |

**Example**:
```
Word: "_pp_e"

Frequency: Filters dict → counts freq → guesses 'a' or 'l'

My model:
  Position 0 logits: [0.95(a), 0.02(o), ...]
  Position 3 logits: [0.05(a), 0.92(l), ...]
  Aggregates → sees 'a' needed at pos 0, 'l' at pos 3
  → Smarter decision based on position-specific context
```

---

## How It's Like BERT

| BERT | My Hangman Model |
|------|-----------------|
| Task: Masked Language Modeling | Task: Masked Character Prediction |
| Input: "The cat [MASK] on mat" | Input: "_ p p _ e" |
| Output: P(word \| context) for MASK | Output: P(letter \| context) for each _ |
| Architecture: Transformer encoder | Architecture: BiLSTM encoder |
| Bidirectional context | Bidirectional context |
| Position-wise classification | Position-wise classification |

---

## Running the Code

### Train Model
```bash
python train.py
```

### Test Frequency Baseline
```bash
python api/test.py
```

### Current Test Output
```
Testing word: 'apple'
...
Result: WIN
Tries remaining: 4/6
```

---

## Technical Learnings

1. **Encoding consistency is critical**: Train and inference must use identical encoding
2. **Position-wise > word-wise**: More information by predicting per position
3. **BiLSTM captures context**: Sees both left and right neighbors
4. **Modular testing framework**: Same `game_engine` for all models

---
