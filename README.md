# Grounded Conversation Intelligence RAG

A production‑style Retrieval Augmented Generation (RAG) system designed to answer questions about internal conversations, policies, and operational documentation with **grounded citations and measurable retrieval quality improvements**.

This repository demonstrates how to design, evaluate, and monitor a modern RAG system rather than simply calling an LLM with embedded documents.

The project focuses on:

* Retrieval quality
* Grounded answer generation
* Quantitative evaluation
* Production monitoring

The goal is to show **end‑to‑end AI system design**, not just prompt engineering.

---

# Project Overview

Modern RAG systems fail most often because of poor retrieval rather than weak LLMs.

This project explores how improvements in retrieval architecture improve answer quality.

We implement and compare several system variants:

| Variant           | Description                                      |
| ----------------- | ------------------------------------------------ |
| Baseline          | Lexical retrieval (TF‑IDF / BM25 style)          |
| Dense             | Embedding retrieval using sentence transformers  |
| Hybrid            | Combined lexical + dense retrieval               |
| Hybrid + Reranker | Hybrid retrieval followed by relevance reranking |

Each system is evaluated using retrieval metrics and answer quality checks.

---

# Architecture

```
Documents / transcripts
        ↓
Cleaning + chunking
        ↓
Metadata enrichment
        ↓
Indexing
        ↓
Retrieval
   (lexical / dense / hybrid)
        ↓
Reranking
        ↓
Context assembly
        ↓
LLM generation
        ↓
Evaluation + monitoring
```

The system emphasizes **retrieval correctness and observability**, two components frequently missing from typical RAG tutorials.

---

# Repository Structure

```
grounded_conversation_rag/

README.md

notebooks/
    01_baseline_rag.ipynb
    02_hybrid_retrieval.ipynb
    03_reranking.ipynb
    04_evaluation.ipynb
    05_monitoring_demo.ipynb

src/
    config.py
    ingestion.py
    chunking.py
    retrieval.py
    reranking.py
    generation.py
    evaluation.py
    monitoring.py

data/
    raw/
    processed/
    eval/

tests/

reports/
    benchmark_results.csv
    error_analysis.md
    architecture.png
```

---

# Key Concepts Demonstrated

## Retrieval Evaluation

Retrieval is evaluated separately from generation using:

* Recall@k
* Mean Reciprocal Rank (MRR)
* Retrieval error inspection

Understanding retrieval failure modes is critical for building reliable RAG systems.

---

## Grounded Answer Generation

The generator is constrained to answer **only from retrieved context**.

The prompt template enforces:

* citation of source chunks
* abstention when evidence is weak
* grounded reasoning

---

## Evaluation Framework

The project includes a small gold evaluation dataset containing:

* query
* ideal answer
* relevant documents
* required facts
* forbidden claims

Evaluation measures:

* answer relevance
* groundedness
* citation coverage

---

## Monitoring

Production‑style monitoring signals are simulated:

* latency
* retrieval score distribution
* citation rate
* abstention rate
* token cost

Monitoring helps detect degradation before users notice failures.

---

# Example Results

Example benchmark comparison:

| Variant           | Recall@5 | MRR  | Faithfulness | Avg Latency |
| ----------------- | -------- | ---- | ------------ | ----------- |
| Baseline lexical  | 0.61     | 0.49 | 0.72         | 0.8s        |
| Dense retrieval   | 0.68     | 0.56 | 0.76         | 1.0s        |
| Hybrid retrieval  | 0.78     | 0.66 | 0.84         | 1.2s        |
| Hybrid + reranker | 0.82     | 0.73 | 0.88         | 1.7s        |

*Numbers are illustrative; reproduce by running the notebooks.*

---

# Running the Project

## Install dependencies

```
pip install pandas numpy scikit-learn sentence-transformers matplotlib
```

Optional tools:

```
faiss-cpu
rank-bm25
ragas
langsmith
```

---

## Run the notebooks

Start with the baseline notebook:

```
notebooks/01_baseline_rag.ipynb
```

Then progressively run:

1. hybrid retrieval
2. reranking
3. evaluation framework
4. monitoring experiments

---

# Example Query

```
How do we identify agents likely to churn?
```

Example retrieved evidence:

```
[doc_3_chunk_0]
Important churn indicators include declining logins, policy count decay, reduced quoting activity...
```

Example grounded answer:

```
Agents at risk of churn can be identified by declining CRM logins, decreasing policy counts, and reduced quoting activity [doc_3_chunk_0].
```

---

# Future Improvements

Planned improvements include:

* cross‑encoder reranking
* reciprocal rank fusion
* query rewriting
* parent‑child retrieval
* hallucination detection
* LLM‑as‑judge evaluation
* vector database integration

---

# Why This Project Exists

Many RAG tutorials show how to embed documents and call an LLM.

This project instead demonstrates how to:

* measure retrieval quality
* improve ranking
* evaluate answers
* monitor system behavior

These skills are essential for **production AI systems**.

---

# Author

Liam Geron

Senior Data Scientist working on ML systems, NLP, and applied AI.

This repository is part of an ongoing exploration of **retrieval systems, LLM evaluation, and production AI architecture**.
