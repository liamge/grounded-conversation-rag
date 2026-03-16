# syntax=docker/dockerfile:1.7
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    KMP_AFFINITY=disabled \
    PATH="/home/app/.local/bin:${PATH}"

# System deps needed for torch / faiss wheels and streamlit
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install project (uses pyproject.toml)
COPY pyproject.toml README.md ./
COPY src src
COPY config config
COPY data data

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[eval,dense]"

EXPOSE 8501

# Streamlit runs as non-root for better defaults
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
