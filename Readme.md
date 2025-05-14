#  Human vs AI Text Classification

This project classifies whether a given sentence is written by a **human** or **AI**, using NLP techniques and deep learning with BERT embeddings.

---

##  Dataset Overview

- **Total samples**: 300 sentences  
- **Balanced classes**:
  - 150 Human-written
  - 150 AI-generated

- **Subcategories** (within each class):
  - 50 Business
  - 50 Sports
  - 50 Science

- **Labels**:
  - `0` → Human
  - `1` → AI

---

##  Embedding Model

- **Model**: `bert-base-uncased` (HuggingFace Transformers)
- **Embedding size**: `768` dimensions
- **Embedding type**: `[CLS]` token output (represents whole sentence)
- **OOV Handling**: Automatically handled by BERT via WordPiece tokenizer

---

##  Preprocessing Pipeline

Implemented using `scikit-learn` pipeline with custom transformers:

1. Lowercasing
2. Contraction expansion
3. Punctuation removal
4. Stopword removal (NLTK)
5. Special character removal
6. Lemmatization (WordNet + POS tags)
7. BERT vectorization (`768`-dimensional)

---

##  Data Splitting

- **Train**: 70% → 210 samples  
- **Validation**: 15% → 45 samples  
- **Test**: 15% → 45 samples  

Using `train_test_split()` while maintaining class balance.

---

##  Model Architecture

A simple **Feedforward Neural Network** using BERT embeddings:

```python
model = Sequential([
    Dense(256, activation='relu', input_shape=(768,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
