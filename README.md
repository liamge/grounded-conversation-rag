# Grounded Conversation RAG

Portfolio‑ready RAG system that ingests internal conversations + policies, runs lexical/dense/hybrid retrieval with optional reranking, and serves grounded answers with citations via a Streamlit demo. The codebase is fully runnable (no placeholder notebooks) and ships with a small synthetic corpus plus an evaluation set.

---

## Quickstart (reproducible)
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -e .` for the minimal stack, or `pip install -e ".[dev,eval,dense]"` to include tests, OpenAI, FAISS, and evaluation extras.
- Set `OPENAI_API_KEY` if you want live LLM answers; otherwise a deterministic fallback generator is used.
- Optional: set `RAG_DISABLE_DENSE=1` to skip downloading sentence-transformers weights on first run.

Data paths, chunking parameters, models, and app settings live in `config/settings.yaml` and can be overridden via env vars (e.g., `RAG_APP__PORT=8080`).

---

## What’s implemented
- **Ingestion (`src/ingestion.py`)**: loads `.txt`, `.md`, `.json`, `.jsonl`; normalizes text; deterministic document IDs.
- **Chunking (`src/chunking.py`)**: sentence-aware splitter with overlap and stable `chunk_id`s.
- **Retrieval (`src/retrieval.py`)**: TF‑IDF, BM25, dense (MiniLM), hybrid fusion, optional FAISS caching; score normalization built for benchmarking.
- **Reranking (`src/reranking.py`)**: lightweight keyword-overlap heuristic toggleable per query.
- **Grounded generation (`src/generation.py`)**: context budget, citation-enforcing prompt, abstention guard, OpenAI provider with deterministic fallback.
- **Pipeline orchestrator (`src/pipeline.py`)**: single entrypoint to load corpus → index → retrieve → (re)rank → generate → emit `PipelineResult` + trace.
- **Evaluation (`src/evaluation.py`)**: Recall@k, Precision@k, MRR, citation coverage, evidence overlap, abstention checks; `data/eval/eval_set.jsonl` ships with gold pairs.
- **Monitoring (`src/monitoring.py`)**: per-query telemetry + aggregates; writes CSV summaries to `reports/`.
- **Streamlit demo (`src/app.py`)**: polished UI with retrieval mode selector, rerank toggle, chunk inspector, live latency metrics, and session benchmarks.
- **CLIs (`scripts/`)**: `smoke_generate.py` (sanity check), `build_embeddings.py` (precompute dense + FAISS), `run_retrieval_benchmark.py` (toy Recall/MRR comparison).

---

## Run the demo
```bash
source .venv/bin/activate
streamlit run src/app.py
```
Open http://localhost:8501. The sidebar shows latency + citation counts; the main panel surfaces grounded answers and retrieved chunks. Set `OPENAI_API_KEY` for model-backed answers or rely on the deterministic fallback.

---

## Run quick checks
- **Smoke test (no external keys needed):**
  ```bash
  python scripts/smoke_generate.py
  ```
- **Toy retrieval benchmark:**
  ```bash
  python scripts/run_retrieval_benchmark.py --no-dense   # skip embeddings download
  ```
- **Precompute dense embeddings + FAISS cache:**
  ```bash
  python scripts/build_embeddings.py --config config/settings.yaml
  ```
- **Unit tests:** `pytest`

---

## Data + evaluation assets
- Corpus lives in `data/raw/` (policies, operations runbooks, support conversations). All loaders accept txt/md/json/jsonl so you can drop in new sources.
- Gold eval set at `data/eval/eval_set.jsonl` with queries, expected answers, and relevant chunk IDs for retrieval + groundedness checks.
- Generated artifacts, telemetry, and reports are written to `artifacts/`, `logs/`, and `reports/` (created automatically).

---

## Architecture at a glance
```
raw docs → ingest → chunk → index (tfidf | bm25 | dense | hybrid)
               ↓                     ↓
         metadata-rich chunks   optional rerank
               ↓                     ↓
        context assembly → grounded prompt → generator → citations/abstention
                                        ↓
                      telemetry (latency, scores, citations) → reports/
```

Design choices:
- Keep retrievers small/testable; hybrids fuse normalized scores for clarity in benchmarks.
- Deterministic chunk/document IDs to make eval + caching repeatable.
- Provider-agnostic generation with a strict citation/abstention contract so demos work even without API keys.
- Config-first: YAML + env overrides; defaults safe for laptops, flags to disable heavy components.

---

## Repository layout (actual)
- `src/` — pipeline, retrieval, reranking, generation, config, schemas, telemetry, Streamlit app
- `scripts/` — embeddings builder, toy benchmark, smoke generation
- `data/raw/` — sample corpus; `data/eval/` — gold set; `data/downloads/` placeholder for new corpora
- `config/settings.yaml` — defaults for paths/models/app
- `tests/` — chunking, ingestion, retrieval, monitoring tests
- `Dockerfile` + `docker-compose.yml` — containerized Streamlit app (`docker compose up app`)

---

## Deployment placeholder
- TODO: add instructions for Streamlit Community Cloud + container registry workflow.
- TODO: include `helm/` or Terraform snippet if deploying behind an API.

---

## Portfolio assets to drop in
- TODO: App screenshot (e.g., `docs/images/app.png`) showing answer + citations + sidebar metrics.
- TODO: Benchmark table or chart generated from `reports/` after running evaluation.
- TODO: Short deployment blurb with live URL once hosted.

---

## Author
Liam Geron — applied ML / retrieval systems. Built to showcase grounded RAG design, not just prompt engineering.
