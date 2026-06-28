# FinNLP: Financial Sentiment Analysis Pipeline

This repository contains a sequence classification pipeline optimized for extracting market, economic, and corporate sentiment from multilingual financial text strings. Utilizing `XLM-ROBERTa-Base`, the pipeline supports high-accuracy cross-lingual transfer across both English and Chinese financial corpora.

---

## Pipeline Overview

The pipeline leverages deep self-supervised language representations to handle mixed semantic contexts, domain-specific vocabularies, and cross-lingual requirements typical of financial natural language processing (FinNLP).

* **Model Core**: `FacebookAI/xlm-roberta-base` hosted on the Hugging Face Model Hub.
* **Architecture**: A bidirectional multi-head self-attention transformer network trained on 2.5 Terabytes of raw web text data.
* **Tokenization**: Employs a specialized SentencePiece tokenizer pre-trained on a 100-language corpus with a vocabulary size of 250,000 individual sub-word IDs, successfully mitigating unknown token dropouts (`[UNK]`).

---

## Dataset & Ingestion Layer

The system executes two isolated pipelines tailored to the language and label structure of the target corpus:

### 1. English Pipeline (3-Class)
* **Source**: Streams raw data natively from the Hugging Face repository hub via `AdityaA19/Sentiment_analysis_finance_dataset`.
* **Classes**: Trinary categorization mapping into Negative (0), Neutral (1), and Positive (2) numerical integer classes.
* **Partitioning**: A sub-corpus slice of 20,000 text instances is extracted, shuffled using a locked tracking seed (`seed=42`), and partitioned via a 90/10 operational train-test split boundary (18,000 samples for training and 2,000 records reserved as an out-of-sample test set).

### 2. Chinese Pipeline (Binary)
* **Source**: Ingested manually through dataframes and converted into native table vectors using datasets derived from Professor Yicheng Wang's research.
* **Classes**: Operates on a binary setting where the target column maps explicit financial risks (1) against non-negative control contexts (0).
* **Partitioning**: Configured with an identical 90/10 proportion and a locked seed of 42, yielding a training set of 4,499 samples and a validation set of 500 samples.

---

## Optimization & Hyperparameters

To combat catastrophic forgetting and stabilize validation performance over smaller data distributions, the classification networks utilize conservative optimization boundaries:

| Metric / Configuration | English Pipeline | Chinese Pipeline |
| --- | --- | --- |
| **Learning Rate** | 1e-5 | 1e-5 |
| **Weight Decay** | 0.02 | 0.05 |
| **Training Epochs** | 5 | 3 |
| **Target Outputs (num_labels)** | 3 | 2 |
| **Evaluation Strategy** | Macro-Averaged F1-Score | Macro-Averaged F1-Score |

> **Note:** Both configurations specify `load_best_model_at_end=True`, forcing the trainer to track evaluation data at every epoch checkpoint and dynamically roll back to the historical version that yielded the peak Macro-Averaged F1-Score.

---

## Core Engineering Optimizations

* **Type-Safety Layer**: Includes an inline type-safety conversion layer (`str(text) if text is not None else ""`) to explicitly clean incoming data vectors, converting elements to string primitives and mapping empty fields to safe blank strings to prevent runtime tokenization errors.
* **Batched Tokenization**: Tokenizer execution uses the parameter `batched=True` to feed elements in multi-sample batches, significantly reducing Python loop overhead and maximizing GPU caching parallelization.
* **Dynamic Padding**: Static padding is omitted during the core tokenizer mapping pass. Instead, raw token sequences are handled by a downstream `DataCollatorWithPadding` layer that dynamically pads elements within individual mini-batches to match only the longest sequence length present inside that specific batch.

---

## Empirical Convergence Results

### English Pipeline Performance
The optimization loop completed 5,625 global iterations, documenting a stable, continuous decline of both training and validation losses:
* **Final Evaluation Accuracy**: 99.95% (achieved at Epoch 5)
* **Macro F1-Score**: 0.9995

### Chinese Pipeline Performance
The Chinese sequence classification network successfully resolved complex semantic structures without showing performance bias toward majority distributions:
* **Final Evaluation Accuracy**: 94.80% (achieved at Epoch 3)
* **Macro F1-Score**: 94.68%

---

## Implementation Quickstart

### Data Ingestion & Preprocessing

```python
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# Initialize Tokenizer
model_checkpoint = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# English Tokenization Pipeline
def preprocess_function_en(examples):
    safe_text = [str(text) if text is not None else "" for text in examples["text"]]
    return tokenizer(safe_text, truncation=True)

# Chinese Tokenization Pipeline (Redirecting target column to 'labels')
def preprocess_function_ch(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=128)
    tokenized["labels"] = examples["negative"]
    return tokenized

# Data Collator for Dynamic Mini-Batch Padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

Authors: Luce Busi, Samuel Cañedo   Release Date: June 28, 2026
