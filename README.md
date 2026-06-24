## 🛠️ Tech Stack

**Frameworks & Libraries**
- TensorFlow 2 / Keras
- tensorflow-text
- HuggingFace Transformers

**Task:** Sequence-to-Sequence Neural Machine Translation (English → Arabic)

**Models Implemented**

| Model | Tokenizer | Architecture | Strength |
|---|---|---|---|
| A — GRU Seq2Seq | Word-level (Keras) | BiLSTM/GRU Encoder-Decoder + RepeatVector | Fast baseline |
| B — Attention GRU | BERT WordPiece | BiGRU Encoder + Cross-Attention + GRU Decoder | OOV robustness |
| C — Transformer | BERT WordPiece | 4-layer, 8-head Transformer (built from scratch) | SOTA accuracy |
| D — HuggingFace | Pretrained | marefa-mt-en-ar | Production ceiling |

Each model is trained/evaluated under the same pipeline to benchmark progressively more advanced NMT architectures against a production-grade pretrained baseline.
