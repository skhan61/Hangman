# Hangman AI Solver - Position-wise Neural Approach
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17336653.svg)](https://doi.org/10.5281/zenodo.17336653)

> **Trexquant Investment LP Interview Project**

A neural network-based Hangman solver using **position-wise character prediction**, inspired by BERT's masked language modeling approach. The system achieves **67.2% win rate** on Trexquant's official test set, representing a **3.7× improvement** over the 18% frequency-based baseline.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Problem Statement](#-problem-statement)
- [Our Approach](#-our-approach)
- [Technical Architecture](#-technical-architecture)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Testing & Evaluation](#-testing--evaluation)
- [Data Generation](#-data-generation)
- [Guessing Strategies](#-guessing-strategies)
- [Technical Report](#-technical-report)
- [Development](#-development)
- [Makefile Commands](#-makefile-commands)
- [Citation](#-citation)

---

## 🎯 Project Overview

This project implements a state-of-the-art Hangman solver for the Trexquant Investment LP technical interview challenge. The challenge required developing an algorithm that:

- Plays Hangman through Trexquant's REST API
- Significantly outperforms the 18% baseline win rate
- Trains only on a provided 250,000-word dictionary
- Tests on a completely disjoint set of 250,000 unseen words
- Handles a maximum of 6 incorrect guesses per game

**Status:** ✅ **Completed & Submitted**

**Citation:** The full project package is archived on Zenodo — DOI [10.5281/zenodo.17336653](https://doi.org/10.5281/zenodo.17336653).

---

## 🏆 Key Results

### Official Test Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Official Win Rate** | **67.2%** | 672 wins out of 1,000 games |
| **Improvement vs Baseline** | **3.7×** | From 18% to 67.2% |
| **Percentage Point Gain** | **+49.2 pp** | Absolute improvement |
| Practice Win Rate | 63.07% | 1,752 wins out of 2,778 games |
| Avg Tries Remaining | 2.1 | On successful games |
| Model Used | `epoch=18` | `win_rate=0.6380` checkpoint |

### Strategy Comparison

Evaluated on 1,000 unseen test words from `data/test_unique.txt`:

| Strategy | Win Rate | Avg Tries Left | Description |
|----------|----------|----------------|-------------|
| **Neural (Ours)** | **63.3%** | **2.1** | Position-wise neural prediction |
| Positional Frequency | 17.0% | 0.4 | Pattern-aware heuristic |
| Frequency Baseline | 15.1% | 0.3 | Simple letter frequency |
| Trexquant Baseline | ~18% | N/A | Provided baseline |

**Key Insight:** Neural approach achieves **4.2× better win rate** than heuristic baselines while maintaining robust performance with 2.1 average tries remaining on wins.

---

## 📝 Problem Statement

### The Hangman Game Challenge

**From Trexquant Investment LP:**

> For this coding test, your mission is to write an algorithm that plays the game of Hangman through our API server. When a user plays Hangman, the server first selects a secret word at random from a list. The server then returns a row of underscores (space separated)—one for each letter in the secret word—and asks the user to guess a letter.

**Rules:**
1. If the guessed letter is in the word: reveal all instances in correct positions
2. If the guessed letter is not in the word: charge an incorrect guess
3. Win condition: correctly guess all letters
4. Lose condition: make 6 incorrect guesses

**Constraints:**
- Training dictionary: 250,000 words (provided)
- Test dictionary: 250,000 words (disjoint, unseen)
- No external dictionaries allowed
- Must significantly outperform 18% baseline

**Baseline Strategy:**
Trexquant provided a frequency-based baseline (~18% win rate) that:
1. Filters dictionary by word length and revealed pattern
2. Counts letter frequencies in matching words
3. Guesses the most frequent unguessed letter
4. Falls back to global frequency when no matches exist

---

## 💡 Our Approach

### Why Position-wise Prediction?

#### ❌ Traditional Approach Limitation

Most Hangman solvers use **frequency-based heuristics** that treat it as a single-letter classification problem:

```python
# Example: "_pp_e"
# Problem: Picks ONE letter for entire word
candidates = filter_dictionary(pattern="_pp_e")  # ["apple", "ample", ...]
letter_freq = Counter("".join(candidates))        # {a: 45, o: 12, ...}
guess = letter_freq.most_common(1)[0][0]          # 'a'
```

**Limitation:** Ignores position-specific context. The letter at position 0 and position 3 may have very different distributions.

#### ✅ Our Position-wise Solution

We frame Hangman as a **position-wise multi-label prediction problem**, inspired by BERT's masked language modeling:

```python
# Example: "_pp_e"
# Solution: Predict letter at EACH masked position

state = encode("_pp_e")     # [MASK, p, p, MASK, e]
logits = model(state)       # [batch, length, 26]

# Position 0: P(a|pos=0)=0.95, P(o|pos=0)=0.02, ...
# Position 3: P(l|pos=3)=0.92, P(k|pos=3)=0.05, ...

# Aggregate predictions across masked positions
aggregated = logits[0, [0, 3], :].sum(dim=0)  # [26]
guess = pick_highest_unguessed(aggregated)     # 'a'
```

**Model Output Format:** `[batch_size, word_length, 26]`
- For each position: probability distribution over 26 letters
- Like BERT predicting masked tokens in a sentence

### Key Advantages

1. **Context-Aware**: Each position considers surrounding revealed letters
2. **Bidirectional**: BiLSTM/Transformer captures both left and right context
3. **Learned Patterns**: Neural model learns linguistic patterns from data
4. **Robust**: Handles rare patterns better than frequency heuristics
5. **Scalable**: Transfer learning potential from larger language models

---

## 🏗️ Technical Architecture

### Supported Model Architectures

#### 1. BiLSTM (Best Performance ⭐)

```
Input: Masked word [batch, word_len]
       e.g., [MASK, p, p, MASK, e]  (encoded as integers)
       ↓
Character Embedding [batch, word_len, 256]
       ↓
Dropout (0.3)
       ↓
BiLSTM [batch, word_len, 512]  (bidirectional → 256*2)
  - 4 layers
  - Packed sequences for efficiency
       ↓
Dropout (0.3)
       ↓
Linear Projection [batch, word_len, 26]
       ↓
Per-position Softmax → Letter probabilities
```

**Configuration:**
- Embedding dim: 256
- Hidden dim: 256 (bidirectional → 512 total)
- Layers: 4
- Dropout: 0.3
- Parameters: ~5.2M

#### 2. Transformer Alternative

```
Input + Positional Embedding
       ↓
Multi-head Self-Attention (8 heads, 4 layers)
       ↓
Feed-forward Network (dim=1024)
       ↓
Linear Projection [batch, word_len, 26]
```

**Configuration:**
- Embedding dim: 256
- Attention heads: 8
- Layers: 4
- FFN dim: 1024
- Dropout: 0.1
- Max length: 45
- Parameters: ~6.8M

#### 3. HangmanBERT (Experimental)

- Pre-trained BERT embeddings with optional freezing
- Layer-wise unfreezing support (`--freeze-bert-layers`)
- Custom head for position-wise prediction
- Parameters: ~110M (full) or ~2M (frozen BERT)

### Loss Function

Position-wise cross-entropy loss with masking:

```python
# Per-position cross-entropy
per_position_loss = -(labels * log_probs).sum(dim=-1)  # [batch, length]

# Apply mask (only compute loss on masked positions)
masked_loss = per_position_loss * mask  # [batch, length]

# Average over masked positions only
loss = masked_loss.sum() / mask.sum()
```

Mathematical formulation:

```
L = -1/|M| Σ_{i∈M} Σ_{c=1}^{26} y_{i,c} log(ŷ_{i,c})
```

where:
- `M` is the set of masked positions
- `y_{i,c}` is the one-hot target for position i, letter c
- `ŷ_{i,c}` is the predicted probability for position i, letter c

---

## 📁 Project Structure

```
Hangman/
├── api/                          # Hangman game API and strategies
│   ├── __init__.py
│   ├── guess_strategies.py      # 11 guessing strategies (frequency, neural, etc.)
│   ├── hangman_api.py           # Trexquant API wrapper
│   ├── hangman_api_user.ipynb   # Demo notebook (original from Trexquant)
│   ├── offline_api.py           # Offline game simulation
│   ├── test.py                  # Strategy comparison script
│   └── 3-game_testing.ipynb     # Final testing & official submission
│
├── dataset/                      # Data loading and generation
│   ├── __init__.py
│   ├── data_generation.py       # Trajectory generation (13 masking strategies)
│   ├── data_module.py           # Lightning DataModule with optimizations
│   ├── encoder_utils.py         # Character encoding utilities
│   ├── hangman_dataset.py       # Parquet-backed dataset
│   └── observation_builder.py   # Converts game states to model inputs
│
├── models/                       # Neural architectures
│   ├── __init__.py
│   ├── lightning_module.py      # Training orchestration + metrics
│   ├── metrics.py               # Custom evaluation metrics
│   └── architectures/
│       ├── __init__.py
│       ├── base.py              # Shared base classes
│       ├── bilstm.py            # BiLSTM implementation ⭐
│       ├── transformer.py       # Transformer implementation
│       ├── bert.py              # HangmanBERT implementation
│       ├── gru.py               # GRU variant
│       ├── charrnn.py           # Character RNN
│       ├── mlp.py               # MLP baseline
│       ├── bilstm_attention.py  # BiLSTM + attention
│       └── bilstm_multihead.py  # BiLSTM + multi-head attention
│
├── hangman_callback/            # Training callbacks
│   ├── __init__.py
│   └── callback.py              # Periodic Hangman evaluation during training
│
├── data/                        # Word lists and datasets
│   ├── words_250000_train.txt   # Training dictionary (227,300 words)
│   ├── test_unique.txt          # Test set for evaluation
│   ├── 20k.txt                  # Small test set
│   ├── dataset_227300words.parquet  # Preprocessed training data (~21M samples)
│   ├── word_length_stats.py    # Dataset statistics
│   └── check_overlap.py         # Verify train/test separation
│
├── logs/
│   └── checkpoints/             # Trained model checkpoints
│       └── best-hangman-epoch=18-hangman_win_rate=0.6380.ckpt
│
├── report/                      # Technical documentation
│   ├── hangman_project_report.tex   # LaTeX source
│   ├── hangman_project_report.pdf   # 16-page technical report
│   ├── compile_report.sh        # PDF compilation script
│   └── README.md                # Report documentation
│
├── main.py                      # Training entry point
├── benchmark.py                 # Benchmarking utilities
├── extract_unique_20k.py        # Data preparation scripts
├── Makefile                     # Development commands
├── environment.yaml             # Conda environment
├── pyproject.toml               # Python project config
├── requirements.txt             # Pip dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training, optional)
- Conda (recommended)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Hangman

# Create and activate conda environment
conda env create -f environment.yaml
conda activate orchestra

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Environment Details

Key dependencies:
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- PyArrow (for Parquet)
- NumPy, Pandas
- Requests (for API)
- tqdm (for progress bars)

---

## 🎮 Quick Start

### 1. Test Pre-trained Model

```bash
# Test neural strategy on 100 words
conda run -n orchestra python -m api.test --limit 100

# Compare all strategies
conda run -n orchestra python -m api.test --limit 30
```

### 2. Train Your Own Model

```bash
# Train BiLSTM model (recommended)
python main.py --max-epochs 20 --batch-size 1024

# Train Transformer model
python main.py --max-epochs 20 --batch-size 1024 --model-arch transformer

# Train with larger batches (requires more GPU memory)
python main.py --max-epochs 20 --batch-size 4096 \
  --row-group-cache-size 300 --prefetch-factor 10
```

### 3. Run Official API Test

```bash
# Open the testing notebook
jupyter notebook api/3-game_testing.ipynb

# Or run practice games programmatically
python -c "
from api.hangman_api import HangmanAPI
api = HangmanAPI(access_token='YOUR_TOKEN', dict_path='data/words_250000_train.txt')
api.start_game(practice=1, verbose=True)
"
```

---

## 🔬 Training

### Basic Training

```bash
# Train BiLSTM with default settings
python main.py --max-epochs 20 --batch-size 1024
```

### Advanced Training Options

```bash
# Full configuration
python main.py \
  --max-epochs 20 \
  --batch-size 2048 \
  --learning-rate 1e-3 \
  --model-arch bilstm \
  --num-workers 24 \
  --row-group-cache-size 200 \
  --prefetch-factor 8 \
  --precision 16-mixed
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-epochs` | 20 | Maximum training epochs |
| `--batch-size` | 1024 | Training batch size (1024-4096) |
| `--learning-rate` | 1e-3 | Adam learning rate |
| `--model-arch` | `bilstm` | Architecture: `bilstm`, `transformer`, `bert` |
| `--num-workers` | 8 | DataLoader worker processes |
| `--row-group-cache-size` | 100 | Parquet row group cache size |
| `--prefetch-factor` | 4 | Prefetch batches per worker |
| `--precision` | `32` | Training precision: `32`, `16-mixed` |

### Training Optimizations

1. **DataLoader Optimizations**
   - Persistent workers (stay alive between epochs)
   - Pin memory (faster CPU→GPU transfer)
   - Prefetching (N batches ahead)
   - Row group caching (Parquet optimization)
   - Optimized collation (pre-allocated tensors)

2. **Model Checkpointing**
   - Monitors `hangman_win_rate` metric
   - Saves only best model
   - Filename: `best-hangman-epoch=N-hangman_win_rate=X.XXXX.ckpt`
   - Location: `logs/checkpoints/`

3. **Evaluation Callback**
   - Runs at epoch 0 (untrained baseline)
   - Runs every N epochs during training
   - Evaluates on 1,000 test words
   - Logs win rate and avg tries remaining
   - Triggers early stopping if no improvement

4. **Device Optimizations**
   - Tensor Cores enabled (RTX GPUs)
   - Mixed precision training (FP16)
   - Gradient clipping (max norm = 1.0)

---

## 📊 Testing & Evaluation

### Strategy Comparison

```bash
# Compare all strategies on 100 words
python api/test.py --limit 100

# Quick test on 30 words
python api/test.py --limit 30

# Full evaluation on 1,000 words
python api/test.py --limit 1000
```

**Output:**
```
Strategy Comparison Results:
====================================
Frequency Strategy:
  Win Rate: 15.1%
  Avg Tries Remaining: 0.3

Positional Frequency Strategy:
  Win Rate: 17.0%
  Avg Tries Remaining: 0.4

Neural Strategy:
  Win Rate: 63.3%
  Avg Tries Remaining: 2.1
```

### Official API Testing

See [api/3-game_testing.ipynb](api/3-game_testing.ipynb) for the complete testing workflow:

1. Load best checkpoint
2. Run practice games (unlimited)
3. Run official recorded games (1,000 max)
4. Get final statistics via `api.my_status()`

**Important:** Official games can only be run once. Use practice mode extensively before final submission.

---

## 🎲 Data Generation

### Training Data Pipeline

**Source:** 227,300 words from English dictionary
**Output:** ~21M training samples
**Format:** Parquet (efficient lazy loading)

### 13 Masking Strategies

To ensure robust training across diverse game scenarios, we employ 13 masking strategies:

| Strategy | Description | Example (`APPLE`) |
|----------|-------------|-------------------|
| `letter_based` | Mask all occurrences of random letters | `APP__` (E, L masked) |
| `left_to_right` | Sequential from left | `A____`, `AP___`, ... |
| `right_to_left` | Sequential from right | `____E`, `___LE`, ... |
| `random_position` | Random positions | `_P_LE`, `A__L_`, ... |
| `vowels_first` | Mask vowels before consonants | `_PP__` (A, E masked) |
| `frequency_based` | Mask by letter frequency | Rare first, common last |
| `center_outward` | Mask center outward | `__P__`, `_PP__`, ... |
| `edges_first` | Mask edges first | `_PPL_`, `APPL_`, ... |
| `alternating` | Alternating pattern | `_P_L_`, `A_P_E`, ... |
| `rare_letters_first` | Prioritize Q, X, Z | N/A for APPLE |
| `consonants_first` | Mask consonants before vowels | `A___E` (P, P, L masked) |
| `word_patterns` | Pattern-based (suffixes) | `____E` (ending patterns) |
| `random_percentage` | Random 20-80% masking | Variable |

### Trajectory Generation

For each word, generate multiple training samples by incrementally revealing letters:

```python
# Word: "APPLE"
# Generate trajectory (5 steps for 5 unique letters):

Step 1: "_____"  → targets: {0:'A', 1:'P', 2:'P', 3:'L', 4:'E'}
Step 2: "A____"  → targets: {1:'P', 2:'P', 3:'L', 4:'E'}
Step 3: "APP__"  → targets: {3:'L', 4:'E'}
Step 4: "APPL_"  → targets: {4:'E'}
Step 5: "APPLE"  → Complete! (no more targets)

# Each step becomes a training sample
```

**Total Training Samples:** ~21M trajectories from 227K words

---

## 🧠 Guessing Strategies

### Implemented Strategies

#### Heuristic Baselines

1. **`frequency_guess_strategy`** - Letter frequency baseline
   - Count letter frequencies in filtered dictionary
   - Guess most frequent unguessed letter
   - ~15% win rate

2. **`positional_frequency_strategy`** - Position-aware frequency
   - Count frequencies only at masked positions
   - More accurate than simple frequency
   - ~17% win rate

3. **`ngram_guess_strategy`** - N-gram models
   - Use bigrams, trigrams, 4-grams with position awareness
   - Adaptive interpolation and distance weighting
   - ~20% win rate

4. **`entropy_strategy`** - Information gain maximization
   - Pick letter that maximizes information gain
   - Uses Shannon entropy to measure uncertainty reduction
   - ~22% win rate

5. **`vowel_consonant_strategy`** - Vowels first
   - Guess vowels (E, A, I, O, U) before consonants
   - Simple but effective for short words
   - ~16% win rate

6. **`pattern_matching_strategy`** - Exact pattern matching
   - Match exact pattern with regex
   - Count letters only from matching words
   - ~15% win rate

7. **`length_aware_strategy`** - Word length adaptation
   - Different strategies for short vs long words
   - Boost common short-word letters for length ≤4
   - ~18% win rate

8. **`suffix_prefix_strategy`** - Common endings/beginnings
   - Detect patterns like ING, TION, LY, UN_, RE_
   - Guess letters that complete these patterns
   - ~17% win rate

9. **`ensemble_strategy`** - Multi-strategy voting
   - Combines frequency, positional, n-gram, entropy
   - Weighted voting (25% each)
   - ~25% win rate

#### Neural Strategies (Ours)

10. **`neural_guess_strategy`** ⭐ - Pure neural prediction
    - Position-wise neural network
    - Aggregates logits across masked positions
    - **63-67% win rate** (best performance)

11. **`neural_info_gain_strategy`** - Neural + information gain
    - Combines neural predictions with dictionary-based information gain
    - Boosts neural logits by information gain score
    - ~65% win rate

### Strategy API

All strategies follow the same interface:

```python
from api.guess_strategies import neural_guess_strategy, frequency_guess_strategy
from api.hangman_api import HangmanAPI

# Use frequency strategy
api = HangmanAPI(
    access_token="YOUR_TOKEN",
    dict_path="data/words_250000_train.txt",
    strategy=frequency_guess_strategy
)

# Use neural strategy (requires model)
from functools import partial
strategy = partial(neural_guess_strategy, model=trained_model)
api = HangmanAPI(
    access_token="YOUR_TOKEN",
    dict_path="data/words_250000_train.txt",
    strategy=strategy
)
```

---

## 📖 Technical Report

A comprehensive 16-page technical report is available in the [report/](report/) directory:

### Report Contents

1. **Introduction** - Problem statement and objectives
2. **Approach Overview** - Position-wise prediction methodology
3. **Technical Architecture** - Model details and configurations
4. **Training Infrastructure** - Data pipeline and optimizations
5. **Guessing Strategies** - Comparison of 11 strategies
6. **Experimental Results** - Official test results and analysis
7. **Implementation Details** - Project structure and technologies
8. **Challenges & Solutions** - Technical obstacles overcome
9. **Future Work** - Potential improvements
10. **Appendices** - Code samples, training logs, complete results

### Accessing the Report

```bash
# View PDF
xdg-open report/hangman_project_report.pdf  # Linux
open report/hangman_project_report.pdf      # macOS

# Recompile from LaTeX source
cd report
./compile_report.sh

# Or manually
pdflatex hangman_project_report.tex
pdflatex hangman_project_report.tex  # Run twice for TOC
```

**Report Features:**
- Professional LaTeX formatting
- Mathematical notation for loss functions
- Code listings with syntax highlighting
- Architecture diagrams (TikZ)
- Tables and figures for results
- Full bibliography and references

---

## 🛠️ Development

### Development Environment

Run all Python commands inside the `orchestra` conda environment:

```bash
conda activate orchestra

# Or use conda run
conda run -n orchestra python main.py
```

### Code Quality

The project follows Python best practices:
- Type hints throughout
- Docstrings for all public functions
- Logging for debugging
- Modular architecture
- Clean separation of concerns

### Testing

```bash
# Run strategy comparison tests
python -m api.test --limit 100

# Run data generation tests
python -m pytest tests/  # If test suite exists

# Check model loading
python -c "
from models import HangmanBiLSTM, HangmanBiLSTMConfig
config = HangmanBiLSTMConfig(vocab_size=26, mask_idx=26, pad_idx=27)
model = HangmanBiLSTM(config)
print(f'Model created: {sum(p.numel() for p in model.parameters())} parameters')
"
```

---

## 🎛️ Makefile Commands

### Git Operations

```bash
# Push without pre-commit hooks
make push-no-verify m="Your commit message"

# Push with CI skip
make push-no-ci m="Your commit message"

# Push with CI skip and no verify
make push-no-verify m="Your commit message"
```

### Cleaning

```bash
# Clean generated files
make clean

# Clean logs
make clean-logs
```

### Testing

```bash
# Run tests
make test
```

---

## 🚀 Recent Updates

### Performance Achievements
- ✅ **Official Win Rate**: 67.2% (672/1,000 games)
- ✅ **Practice Win Rate**: 63.07% (1,752/2,778 games)
- ✅ **Neural Strategy**: 63.3% win rate, 2.1 avg tries remaining (1000-word run)
- ✅ **Model Checkpointing**: Best models saved based on Hangman win rate
- ✅ **Fast Data Loading**: Optimized for batch size 1024-4096 with prefetching
- ✅ **Real API Testing**: 67.2% win rate verified via Trexquant API

### Architecture Enhancements
- ✅ Added `HangmanBERT` model alongside BiLSTM and Transformer
- ✅ CLI flags `--freeze-bert` and `--freeze-bert-layers` for fine-tuning control
- ✅ Validation callback caps evaluation at 1,000 words for faster feedback
- ✅ Multiple architecture variants (BiLSTM, Transformer, GRU, MLP, etc.)

### Infrastructure Improvements
- ✅ Lightning callbacks for epoch-based Hangman evaluation
- ✅ Automatic best model checkpointing
- ✅ Tensor Cores enabled for RTX GPUs
- ✅ Optimized Parquet-based dataset with row group caching
- ✅ Persistent workers and pin_memory for DataLoader
- ✅ API testing script comparing all strategies
- ✅ Comprehensive 16-page technical report (LaTeX/PDF)

---

## 📚 Citation

If you use this code or approach in your work, please cite:

```bibtex
@techreport{hangman2024,
  title={Hangman AI Solver: A Position-Wise Neural Approach},
  author={Sayem Khan},
  year={2024},
  institution={Trexquant Investment LP Interview Project},
  note={Achieves 67.2\% win rate on official test set}
}
```

This work builds upon our previous meta-reinforcement learning approach:

```bibtex
@misc{hangman_rl_meta_2024,
  author       = {Sayem Khan},
  title        = {{Learning to Learn Hangman}},
  month        = sep,
  year         = 2024,
  doi          = {10.5281/zenodo.13737841},
  version      = {v1.0},
  publisher    = {Zenodo},
  url          = {https://doi.org/10.5281/zenodo.13737841}
}
```

---

## 📞 Contact

For questions or feedback about this project, please refer to the project repository or the comprehensive technical report in [report/hangman_project_report.pdf](report/hangman_project_report.pdf).

---

## 📄 License

This repository contains original code and documentation authored by Sayem Khan for the Trexquant interview challenge. The official Trexquant dataset is **not** distributed here, and no third-party proprietary assets are included.

**Copyright:** © 2025 Sayem Khan. All Rights Reserved.
**Redistribution:** Do not redistribute the repository contents without written permission from Sayem Khan. Trexquant holds no copyright over the assets present in this repository.

---

**Last Updated:** October 12, 2025
**Project Status:** ✅ Completed & Submitted
**Final Win Rate:** 67.2% (Official Test)
