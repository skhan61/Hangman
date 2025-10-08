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
│   ├── offline_api.py       ✅ Local game simulation
│   ├── guess_strategies.py  ✅ Pluggable strategies (frequency, BERT, neural)
│   ├── test.py              ✅ Test scripts
│   └── hangman_api_user.ipynb ✅ API usage examples
│
├── simulation/
│   └── data_generation.py   ✅ Generate training data
│
├── hangman_callback/
│   ├── __init__.py           ✅ Module init
│   └── callback.py           ✅ CustomHangmanEvalCallback for training evaluation
│
├── data/
│   ├── words_250000_train.txt ✅ Corpus
│   └── dataset_250000words.parquet ✅ Preprocessed
│
├── train.py                  ✅ Training script (standalone)
├── main.py                   ✅ Dataset preparation & eval testing
├── 3-game_testing.ipynb     ✅ Game testing notebook
└── checkpoints/              📁 Model checkpoints (local only, gitignored)
```

✅ = Complete
🚧 = In progress
❌ = Not started

---

## What's Working

1. **Data encoding pipeline**: Shared encoder ensures training/inference consistency ✅
2. **BiLSTM architecture**: Position-wise prediction `[batch, word_len, 26]` ✅
3. **Frequency baseline**: Strategy-based approach in `api/guess_strategies.py` ✅
4. **Neural guess strategy**: `neural_guess_strategy()` uses trained model for predictions ✅
5. **Offline API**: `HangmanOfflineAPI` for local game simulation ✅
6. **Evaluation callback**: `CustomHangmanEvalCallback` for training-time evaluation ✅
7. **Modular testing**: Pluggable strategies (frequency, BERT-style, neural) ✅

---

## What's Next

1. **Train the BiLSTM model** on full dataset (ready to train)
2. **Compare strategies**:
   - Frequency baseline: TBD
   - BERT-style baseline: TBD
   - Neural BiLSTM: TBD (after training)
3. **Optimize hyperparameters**: Learning rate, hidden dims, dropout, etc.
4. **Deploy**: Create online API endpoint for trained model

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

### Prepare Dataset
```bash
python main.py
```

### Train Model
```bash
python train.py --train --max-epochs 10
```

### Test Evaluation Callback (Debug Mode)
```bash
python main.py --debug --test-eval-only --eval-callback
```

This will:
- Use the real BiLSTM model (untrained)
- Test on 3 validation words
- Show detailed game progress for each word
- Display win rate and average tries remaining

### Example Output
```
INFO | hangman_callback.callback |
Word: 'apple'
INFO | hangman_callback.callback | Result: WIN
INFO | hangman_callback.callback | Tries remaining: 3/6
INFO | hangman_callback.callback | Game progress:
INFO | hangman_callback.callback |   Guess 'e' ✓ -> ____e
INFO | hangman_callback.callback |   Guess 'a' ✓ -> a___e
INFO | hangman_callback.callback |   Guess 'p' ✓ -> app_e
INFO | hangman_callback.callback |   Guess 'l' ✓ -> apple

Win Rate: 100.00%
Average Tries Remaining: 3.00
```

---

## Technical Learnings

1. **Encoding consistency is critical**: Train and inference must use identical encoding
2. **Position-wise > word-wise**: More information by predicting per position
3. **BiLSTM captures context**: Sees both left and right neighbors
4. **Modular testing framework**: Same `game_engine` for all models

---
