# FinNLP: Financial Sentiment Analysis Pipeline

[cite_start]This repository contains a sequence classification pipeline optimized for extracting market, economic, and corporate sentiment from multilingual financial text strings[cite: 131, 132]. [cite_start]Utilizing `XLM-ROBERTa-Base`, the pipeline supports high-accuracy cross-lingual transfer across both English and Chinese financial corpora[cite: 133].

---

## Pipeline Overview

[cite_start]The pipeline leverages deep self-supervised language representations to handle the domain-specific vocabularies and mixed semantic contexts typical of financial journalism[cite: 131, 135, 136].

* [cite_start]**Model Core**: `FacebookAI/xlm-roberta-base` hosted on the Hugging Face Model Hub[cite: 134, 172].
* [cite_start]**Architecture**: Bidirectional multi-head self-attention layers pre-trained on 2.5 TB of raw web text[cite: 136].
* [cite_start]**Tokenization**: SentencePiece sub-word tokenization with a 250,000 token vocabulary space, eliminating unknown token dropouts (`[UNK]`) between English and Chinese text[cite: 188, 189].

---

## Dataset & Ingestion Layer

The system runs two isolated pipelines tailored to the language and label structure of the target corpus:

### 1. English Pipeline (3-Class)
* [cite_start]**Source**: Natively streamed from Hugging Face via `AdityaA19/Sentiment_analysis_finance_dataset`[cite: 141, 142].
* [cite_start]**Classes**: Trinary categorization mapping into Negative (0), Neutral (1), and Positive (2)[cite: 143, 149, 151].
* [cite_start]**Partitioning**: A subset of 20,000 rows is extracted, shuffled with a fixed tracking seed (`seed=42`), and partitioned into a 90/10 split (18,000 train / 2,000 test)[cite: 153, 154, 155, 156].

### 2. Chinese Pipeline (Binary)
* [cite_start]**Source**: Manually ingested from table vectors compiled by Professor Yicheng Wang[cite: 157, 159].
* [cite_start]**Classes**: Binary risk mapping where `1` represents explicit financial risks and `0` indicates non-negative control contexts[cite: 168].
* [cite_start]**Partitioning**: Structured with an identical 90/10 boundary (`seed=42`), yielding 4,499 training samples and 500 validation samples[cite: 169].

---

## Optimization & Hyperparameters

[cite_start]To combat catastrophic forgetting and stabilize validation performance over smaller distributions, the classification heads utilize conservative optimization limits[cite: 207, 214]:

| Metric / Configuration | English Pipeline | Chinese Pipeline |
| :--- | :--- | :--- |
| **Learning Rate ($\eta$)** | [cite_start]$1 \times 10^{-5}$ [cite: 207] | [cite_start]$1 \times 10^{-5}$ [cite: 234] |
| **Weight Decay ($\lambda$)** | [cite_start]0.02 [cite: 208] | [cite_start]0.05 [cite: 213] |
| **Training Epochs** | [cite_start]5 [cite: 207] | [cite_start]3 [cite: 213] |
| **Target Outputs (`num_labels`)** | [cite_start]3 [cite: 151, 203] | [cite_start]2 [cite: 212] |
| **Evaluation Strategy** | [cite_start]Macro-Averaged F1-Score [cite: 209] | [cite_start]Macro-Averaged F1-Score [cite: 209] |

> [cite_start]**Note:** Both configurations utilize `load_best_model_at_end=True` to automatically track evaluation loss at every checkpoint and roll back to the peak Macro F1 historical state[cite: 209].

---

## Core Engineering Optimizations

* [cite_start]**Type-Safety Layer**: The preprocessing function includes an inline type-safety conversion layer (`str(text) if text is not None else ""`) to handle missing fields (`NoneType`) and safeguard against runtime parallelization crashes[cite: 190, 191].
* **Batched Tokenization**: Tokenizer operations utilize `batched=True` to feed elements into the processing engine as multi-sample batches, mitigating Python loop overhead and maximizing GPU caching capabilities[cite: 192, 193].
* [cite_start]**Dynamic Padding**: Static sequence padding is disabled during tokenization[cite: 194]. [cite_start]Instead, `DataCollatorWithPadding` dynamically stretches sequences to match only the longest string present within each individual mini-batch, significantly reducing trailing pad overhead[cite: 181, 195, 196, 197].

---

## Empirical Results

### English Pipeline Performance
[cite_start]The English sequence classification module documented a stable convergence path over 5,625 global iterations[cite: 216]:

* [cite_start]**Final Evaluation Accuracy**: 99.95% [cite: 137, 220]
* **Macro F1-Score**: 0.9995 [cite: 137, 220]

### Chinese Pipeline Performance
The Chinese sequence classification network successfully resolved complex semantic structures without showing performance bias toward majority distributions[cite: 234, 247, 248]:

* [cite_start]**Final Evaluation Accuracy**: 94.80% [cite: 137, 237]
* [cite_start]**Macro F1-Score**: 94.68% [cite: 137, 237]

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

Authors: Luce Busi, Samuel Cañedo   Release Date: June 28, 2026
