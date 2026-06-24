# 🌍 English-to-Arabic Neural Machine Translation

End-to-end NMT pipeline progressing from a simple GRU Seq2Seq baseline to a full Transformer architecture, benchmarked against a production pretrained HuggingFace model.

> **Goal:** Given an English sentence, output its Arabic translation. Four systems of increasing power are built and compared.

---

## 📑 Table of Contents
- [Tech Stack](#️-tech-stack)
- [Models Implemented](#models-implemented)
- [Methodology](#-methodology)
  - [1. Data Loading & Preprocessing](#1-data-loading--preprocessing)
  - [2. Model A — Simple Seq2Seq](#2-model-a--simple-seq2seq-word-level-tokenizer)
  - [3. Model B — Attention GRU](#3-model-b--attention-gru-bert-wordpiece-tokenizer)
  - [4. Model C — Full Transformer](#4-model-c--full-transformer)
  - [5. Summary of Graphs & Outputs](#5-summary-of-all-graphs--outputs)

---

## 🛠️ Tech Stack

**Frameworks & Libraries**
- TensorFlow 2 / Keras
- tensorflow-text
- HuggingFace Transformers

**Task:** Sequence-to-Sequence Neural Machine Translation (English → Arabic)

### Models Implemented

| Model | Tokenizer | Architecture | Strength |
|---|---|---|---|
| A — GRU Seq2Seq | Word-level (Keras) | BiLSTM/GRU Encoder-Decoder + RepeatVector | Fast baseline |
| B — Attention GRU | BERT WordPiece | BiGRU Encoder + Cross-Attention + GRU Decoder | OOV robustness |
| C — Transformer | BERT WordPiece | 4-layer, 8-head Transformer (built from scratch) | SOTA accuracy |
| D — HuggingFace | Pretrained | marefa-mt-en-ar | Production ceiling |

Each model is trained/evaluated under the same pipeline to benchmark progressively more advanced NMT architectures against a production-grade pretrained baseline.

---

## 📊 Methodology

### 1. Data Loading & Preprocessing

#### 1.1 Datasets
Two publicly available parallel corpora are merged to form the training dataset:

- **`ara_eng.txt`** — sentence pairs, tab-separated (English | Arabic)
- **`ara.txt`** — sentence pairs with an additional CC-attribution column (dropped on load)

After merging, duplicate rows are removed and null rows are dropped. The combined dataset typically contains **hundreds of thousands** of sentence pairs.

#### 1.2 Exploratory Data Analysis

| Visualization | Description | Key Takeaway |
|---|---|---|
| **Sentence Length Distribution** (2-subplot Plotly histogram) | Word-count distribution per sentence, EN (left) vs. AR (right) | Right-skewed; most sentences < 15 words. Arabic sentences run slightly longer per token due to morphological richness. Long-tail (>50 words) sentences are rare but present |
| **Unique Vocabulary Size** (grouped bar chart) | Total unique tokens per language | Arabic has a far larger vocabulary due to agglutinative morphology — a single root generates dozens of inflected forms, motivating subword tokenization in Models B & C |
| **Short-Sentence Count** (bar chart) | Entries with < 5 words per language | Very short entries (greetings, single words) add training noise; an optional filter is provided (commented out) |

#### 1.3 Train / Test Split
The dataset is split **80% / 20%** using `sklearn.train_test_split` with a fixed `random_state=42` for reproducibility. Tokenizers are fit **exclusively on the training split** to prevent data leakage.

---

### 2. Model A — Simple Seq2Seq (Word-Level Tokenizer)

#### 2.1 Tokenization
Keras's built-in `Tokenizer` is used with:
- `oov_token="<OOV>"` — maps unseen words to a special index at inference time
- `lower=True` — case-folds English (Arabic is case-insensitive)
- Separate tokenizers fit for English (source) and Arabic (target)

Vocabulary size = `len(word_index) + 1` (index 0 reserved for padding). Both tokenizers are serialized with `pickle` for reuse at inference time.

#### 2.2 Sequence Encoding
`texts_to_sequences` converts raw strings to integer ID lists. `pad_sequences` pads/truncates from the **right** (`'post'`) to a fixed length (`MAX_SEQ_LEN = 10`).

#### 2.3 Architecture

| Layer | Config | Purpose |
|---|---|---|
| Embedding | `en_vocab_size × 256`, `mask_zero=True` | Token IDs → dense vectors; masking ignores padding downstream |
| Bidirectional LSTM | 256 units | Encodes input forward + backward into a context vector |
| RepeatVector | `max_sequence_length` | Replicates the context vector across all decoder timesteps ("the bridge") |
| Bidirectional LSTM | 256 units, `return_sequences=True` | Produces one hidden state per output timestep |
| TimeDistributed Dense | `ar_vocab_size`, softmax | Projects each timestep to a probability distribution over the Arabic vocabulary |

A parallel **GRU Seq2Seq** variant uses the same structure with GRU cells instead of LSTM — fewer parameters, faster training, comparable accuracy on short sequences.

#### 2.4 Training
- **Optimizer:** RMSprop (BiLSTM) / Adam (GRU)
- **Loss:** `sparse_categorical_crossentropy` (works directly on integer target IDs)
- **Metric:** accuracy
- **Callbacks:** `ModelCheckpoint` (best val_loss), `EarlyStopping` (patience=3)
- **Pipeline:** `tf.data.Dataset` with shuffle → batch(64) → `prefetch(AUTOTUNE)`

#### 2.5 Evaluation
Both models are evaluated on the held-out test set and compared side by side:
