# Hangman Solver - Position-wise BERT-Style Approach

## 🎯 Project Status: **Training & Evaluation Phase**

A neural network-based Hangman solver using **position-wise character prediction**, inspired by BERT's masked language modeling approach. The model achieves **56-68% win rate** vs. 10-20% for frequency-based baselines.

---

## 🚀 Recent Updates

### Performance Achievements
- ✅ **Neural Strategy**: 63.3% win rate, 2.1 avg tries remaining (1000-word run)
- ✅ **Frequency Baseline**: 15.1% win rate, 0.3 avg tries remaining
- ✅ **Contextual Baseline**: 17.0% win rate, 0.4 avg tries remaining
- ✅ **Model Checkpointing**: Best models saved based on Hangman win rate
- ✅ **Fast Data Loading**: Optimized for batch size 1024-4096 with prefetching

### Latest Evaluation (1000-word sample)
- Compared three strategies on 1000 shuffled words from `data/test_unique.txt`
- Neural model leads with a 63.3% win rate and 2.1 tries left on average
- Frequency and pattern-based baselines stay under 20% win rate, underscoring the gap
- Run executed via `python -m api.test --limit 1000` inside the `orchestra` environment

### Architecture Enhancements
- ✅ Added a `HangmanBERT` model alongside BiLSTM and Transformer options.
- ✅ New CLI flags `--freeze-bert` and `--freeze-bert-layers` provide fine-tuning control.
- ✅ Validation callback now caps evaluation at 1000 words for faster feedback loops.

### Infrastructure Improvements
- ✅ Lightning callbacks for epoch-based Hangman evaluation
- ✅ Automatic best model checkpointing
- ✅ Tensor Cores enabled for RTX GPUs
- ✅ Optimized parquet-based dataset with row group caching
- ✅ Persistent workers and pin_memory for DataLoader
- ✅ API testing script comparing all strategies

---

## 📊 Current Results

| Strategy | Win Rate | Avg Tries Remaining | Description |
|----------|----------|---------------------|-------------|
| **Neural** | **63.3%** | **2.1** | Trained position-wise neural model |
| Frequency | 15.1% | 0.3 | Letter frequency baseline |
| BERT-style | 17.0% | 0.4 | Pattern-aware heuristic |

*Evaluated on 1000 unseen words from `data/test_unique.txt`*

---

## 💡 My Approach: Why Position-wise Prediction?

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

## 🏗️ Architecture

### BiLSTM for Bidirectional Context

```
Input: Masked word [batch, word_len]
       e.g., [MASK, p, p, MASK, e]  (encoded as integers)
       ↓
Character Embedding [batch, word_len, 256]
       ↓
BiLSTM [batch, word_len, 512]  (bidirectional → 256*2)
       ↓
Linear Projection [batch, word_len, 26]
       ↓
Softmax per position → Letter probabilities
```

### Transformer Alternative

```
Input + Positional Embedding
       ↓
Multi-head Self-Attention (4 layers)
       ↓
Feed-forward Network
       ↓
Linear Projection [batch, word_len, 26]
```

---

## 📁 Project Structure

```
Hangman/
├── api/                          # Hangman game API and strategies
│   ├── __init__.py
│   ├── guess_strategies.py      # Frequency, contextual, neural strategies
│   ├── hangman_api.py           # Core API wiring for interactive agents
│   ├── hangman_api_user.ipynb   # Notebook demo for manual play
│   ├── offline_api.py           # Offline game simulation helpers
│   └── test.py                  # Smoke tests and strategy comparison
├── dataset/                      # Data loading and generation
│   ├── __init__.py
│   ├── data_generation.py       # Trajectory generation utilities
│   ├── data_module.py           # Lightning DataModule with optimizations
│   ├── encoder_utils.py         # Character encoding utilities
│   ├── hangman_dataset.py       # Parquet-backed dataset
│   └── observation_builder.py   # Converts guesses into model observations
├── models/                       # Model architectures
│   ├── __init__.py
│   ├── lightning_module.py      # Training orchestration + metrics
│   ├── metrics.py               # Custom evaluation metrics
│   └── architectures/
│       ├── __init__.py
│       ├── base.py              # Shared blocks for architectures
│       ├── bilstm.py            # BiLSTM implementation
│       └── transformer.py       # Transformer implementation
├── hangman_callback/            # Training callbacks
│   ├── __init__.py
│   └── callback.py              # Periodic Hangman evaluation callback
├── data/                        # Word lists and evaluation corpora
│   ├── 20k.txt
│   ├── test_unique.txt
│   ├── words_250000_train.txt
│   ├── dataset_227300words.parquet
│   └── check_overlap.py
├── logs/
│   └── checkpoints/             # Trained model checkpoints
├── extract_unique_20k.py        # Utility scripts
├── split_unique_words.py
├── main.py                      # Training entry point
├── Makefile                     # Development commands
├── environment.yaml
├── pyproject.toml
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🎯 Data Generation Strategy

### Masking Strategies (13 types)

To teach the model diverse patterns, I use **13 masking strategies**:

1. **letter_based**: Mask all occurrences of random letters
2. **left_to_right**: Sequential masking from left
3. **right_to_left**: Sequential masking from right
4. **random_position**: Random position masking
5. **vowels_first**: Mask vowels first
6. **frequency_based**: Mask by letter frequency
7. **center_outward**: Mask from center outward
8. **edges_first**: Mask edges first
9. **alternating**: Alternating pattern masking
10. **rare_letters_first**: Mask rare letters first
11. **consonants_first**: Mask consonants first
12. **word_patterns**: Pattern-based masking
13. **random_percentage**: Random percentage masking

### Training Data

- **227,300 words** from English dictionary
- **~1.7M training samples** (multiple trajectories per word)
- Stored in **Parquet format** for efficient lazy loading
- Row group caching for fast batch loading (batch size up to 4096)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Hangman

# Install dependencies
conda env create -f environment.yml
conda activate orchestra
```

### Training

```bash
# Train BiLSTM model (default)
python main.py --max-epochs 10 --batch-size 1024

# Train Transformer model
python main.py --max-epochs 10 --batch-size 1024 --model-arch transformer

# With custom data loading settings for large batches
python main.py --max-epochs 10 --batch-size 4096 \
  --row-group-cache-size 300 --prefetch-factor 10
```

### Testing

```bash
# Test all strategies (Frequency, BERT-style, Neural)
python api/test.py --limit 100

# Quick test
python api/test.py --limit 30
```

### Using Makefile

```bash
# Push changes (no pre-commit hooks)
make push-no-verify m="Your commit message"

# Push changes (with CI skip)
make push-no-ci m="Your commit message"

# Clean generated files
make clean

# Run tests
make test
```

---

## 📊 Training Features

### DataLoader Optimizations

- **Persistent Workers**: Workers stay alive between epochs
- **Pin Memory**: Faster GPU transfers
- **Prefetching**: Workers prefetch N batches ahead (configurable)
- **Row Group Caching**: Cache parquet row groups for faster access
- **Optimized Collation**: Pre-allocate tensors instead of list concatenation

```bash
# Tune for your system
python main.py \
  --batch-size 2048 \
  --num-workers 24 \
  --row-group-cache-size 200 \
  --prefetch-factor 8
```

### Evaluation Callback

- Runs at start (untrained model baseline)
- Runs every N epochs during training
- Evaluates on 1000 test words
- Logs win rate and average tries remaining
- Triggers early stopping if no improvement

### Model Checkpointing

- Monitors `hangman_win_rate` metric
- Saves only the best model
- Filename: `best-hangman-epoch{N}-{win_rate}.ckpt`
- Directory: `logs/checkpoints/`

---

## 🎮 API Usage

### Offline API

```python
from api.offline_api import HangmanOfflineAPI
from api.guess_strategies import frequency_guess_strategy, neural_guess_strategy

# Frequency-based strategy
api = HangmanOfflineAPI(strategy=frequency_guess_strategy)
win, tries, progress = api.play_a_game_with_a_word("apple")

# Neural strategy (requires trained model)
from functools import partial
model = load_trained_model()  # Your model loading code
strategy = partial(neural_guess_strategy, model=model)
api = HangmanOfflineAPI(strategy=strategy)
```

### Strategy Comparison

Run `api/test.py` to compare all strategies:
- Loads best checkpoint automatically
- Tests on custom word list
- Outputs comparison table

---

## 🔬 Technical Details

### Loss Function

```python
# Per-position cross-entropy with masking
per_position_loss = -(labels * log_probs).sum(dim=-1)
masked_loss = per_position_loss * mask
loss = masked_loss.sum() / mask.sum()
```

### Model Configurations

**BiLSTM:**
- Embedding: 256
- Hidden: 256 (bidirectional → 512)
- Layers: 4
- Dropout: 0.3

**Transformer:**
- Embedding: 256
- Heads: 8
- Layers: 4
- Dropout: 0.1
- Max Length: 45

### Device Optimizations

- **Tensor Cores**: Enabled with `torch.set_float32_matmul_precision('medium')`
- **Mixed Precision**: Available via Lightning's `precision='16-mixed'`
- **Gradient Accumulation**: Configurable via Lightning

---
