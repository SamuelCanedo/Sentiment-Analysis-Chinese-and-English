# FinNLP: Financial Sentiment Analysis Pipeline

[cite_start]This repository contains a sequence classification pipeline optimized for extracting market, economic, and corporate sentiment from multilingual financial text strings[cite: 257, 258]. [cite_start]Utilizing `XLM-ROBERTa-Base`, the pipeline supports high-accuracy cross-lingual transfer across both English and Chinese financial corpora[cite: 259, 277, 283].

---

## Pipeline Overview

[cite_start]The pipeline leverages deep self-supervised language representations to handle mixed semantic contexts, domain-specific vocabularies, and cross-lingual requirements typical of financial natural language processing[cite: 257, 258].

* [cite_start]**Model Core**: `FacebookAI/xlm-roberta-base` hosted on the Hugging Face Model Hub[cite: 260].
* [cite_start]**Architecture**: A bidirectional multi-head self-attention transformer network pre-trained on 2.5 Terabytes of raw web text data[cite: 262].
* [cite_start]**Tokenization**: Employs a specialized SentencePiece tokenizer pre-trained on a 100-language corpus with a vocabulary size of 250,000 individual sub-word IDs, successfully mitigating unknown token dropouts (`[UNK]`)[cite: 314, 315].

---

## Dataset & Ingestion Layer

The system executes two isolated pipelines tailored to the language and label structure of the target corpus:

### 1. English Pipeline (3-Class)
* [cite_start]**Source**: Streams raw data natively from the Hugging Face repository hub via `AdityaA19/Sentiment_analysis_finance_dataset`[cite: 265, 268].
* [cite_start]**Classes**: Trinary categorization mapping into Negative (0), Neutral (1), and Positive (2) numerical integer classes[cite: 275, 276, 277].
* [cite_start]**Partitioning**: A sub-corpus slice of 20,000 text instances is extracted, shuffled using a locked tracking seed (`seed=42`), and partitioned via a 90/10 operational train-test split boundary[cite: 279, 280, 281]. [cite_start]This dedicates 18,000 samples to training and 2,000 records to an out-of-sample test set[cite: 282].

### 2. Chinese Pipeline (Binary)
* [cite_start]**Source**: Ingested manually through dataframes and converted into native table vectors using datasets derived from Professor Yicheng Wang's research[cite: 283, 285].
* [cite_start]**Classes**: Operates on a binary setting where the target column maps explicit financial risks (1) against non-negative control contexts (0)[cite: 294].
* [cite_start]**Partitioning**: Configured with an identical 90/10 proportion and a locked seed of 42, yielding a training set of 4,499 samples and a validation set of 500 samples[cite: 295].

---

## Optimization & Hyperparameters

[cite_start]To combat catastrophic forgetting and stabilize validation performance over smaller data distributions, the classification networks utilize conservative optimization boundaries[cite: 333, 340]:

| Metric / Configuration | English Pipeline | Chinese Pipeline |
| --- | --- | --- |
| **Learning Rate** | [cite_start]1e-5 [cite: 333] | [cite_start]1e-5 [cite: 360] |
| **Weight Decay** | [cite_start]0.02 [cite: 334] | [cite_start]0.05 [cite: 339, 360] |
| **Training Epochs** | [cite_start]5 [cite: 333] | [cite_start]3 [cite: 339, 360] |
| **Target Outputs (num_labels)** | [cite_start]3 [cite: 277] | [cite_start]2 [cite: 338] |
| **Evaluation Strategy** | [cite_start]Macro-Averaged F1-Score [cite: 335] | [cite_start]Macro-Averaged F1-Score [cite: 335] |

> [cite_start]**Note:** Both configurations specify `load_best_model_at_end=True`, forcing the trainer to track evaluation data at every epoch checkpoint and dynamically roll back to the historical version that yielded the peak Macro-Averaged F1-Score[cite: 335]. [cite_start]Macro metrics are mandatory to give minority trend signals equal mathematical weight[cite: 336].

---

## Core Engineering Optimizations

* **Type-Safety Layer**: Includes an inline type-safety conversion layer (`str(text) if text is not None else ""`) to explicitly clean incoming data vectors, converting elements to string primitives and mapping empty fields to safe blank strings to prevent runtime tokenization errors[cite: 301, 316, 317].
* [cite_start]**Batched Tokenization**: Tokenizer execution uses the parameter `batched=True` to feed elements in multi-sample batches, significantly reducing Python loop overhead and maximizing GPU caching parallelization[cite: 318, 319].
* [cite_start]**Dynamic Padding**: Omit static padding during the core tokenizer mapping pass[cite: 320]. [cite_start]Instead, raw token sequences are handled by a downstream `DataCollatorWithPadding` layer that dynamically pads elements within individual mini-batches to match only the longest sequence length present inside that specific batch, optimizing GPU tensor execution speeds[cite: 321, 322, 323].

---

## Empirical Convergence Results

### English Pipeline Performance
[cite_start]The optimization loop completed 5,625 global iterations, documenting a stable, continuous decline of both training and validation losses[cite: 342, 345]:
* **Final Evaluation Accuracy**: 99.95% (achieved at Epoch 5) [cite: 344, 346]
* [cite_start]**Macro F1-Score**: 0.9995 [cite: 344, 346]

### Chinese Pipeline Performance
[cite_start]The optimized Chinese implementation punishes high weight values to maintain validation stability over the smaller dataset[cite: 340]:
* **Final Evaluation Accuracy**: 94.80% (achieved at Epoch 3) [cite: 363]
* [cite_start]**Macro F1-Score**: 94.68% [cite: 363]

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
´´´

Authors: Luce Busi, Samuel Cañedo   Release Date: June 28, 2026
