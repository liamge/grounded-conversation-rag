from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .schemas import Document

INDEX_SUBDIR = "index"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class IndexContext:
    """Shared metadata passed to retrievers for artifact management."""

    artifacts_root: Path
    corpus_fingerprint: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int

    def retriever_dir(self, retriever_name: str) -> Path:
        return Path(self.artifacts_root) / INDEX_SUBDIR / retriever_name


def compute_corpus_fingerprint(documents: Sequence[Document]) -> str:
    """Return a stable fingerprint for the ingested corpus.

    The fingerprint is a SHA-256 hash over (doc_id, source, text) tuples sorted by ``doc_id``.
    Using normalized ``Document`` objects keeps the fingerprint stable across runs while still
    changing whenever document contents or sources change.
    """

    hasher = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d.doc_id):
        hasher.update(doc.doc_id.encode("utf-8"))
        hasher.update(doc.source.encode("utf-8"))
        hasher.update(doc.text.encode("utf-8"))
    return hasher.hexdigest()


def load_index_artifacts(artifact_dir: Path) -> Optional[Dict[str, Any]]:
    """Load a manifest from ``artifact_dir`` if present.

    Returns ``None`` when the manifest is missing or unreadable to allow callers to fall back
    to on-the-fly re-indexing without raising.
    """

    manifest_path = Path(artifact_dir) / MANIFEST_FILE
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_index_artifacts(artifact_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Persist a manifest to ``artifact_dir``.

    The directory is created if needed. A UTC timestamp is injected when missing so callers
    don't need to set it explicitly.
    """

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(manifest)
    payload.setdefault("timestamp", datetime.utcnow().isoformat())

    manifest_path = artifact_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
