import json
from pathlib import Path

import pytest

from src.ingestion import Document, ingest_documents, load_document, normalize_text


def test_markdown_title_extraction(tmp_path: Path) -> None:
    md_file = tmp_path / "note.md"
    md_file.write_text("# My Heading\nSome content beneath the heading.\n", encoding="utf-8")

    docs = load_document(md_file)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "My Heading"
    assert doc.text == "# My Heading Some content beneath the heading."


def test_json_list_payload(tmp_path: Path) -> None:
    payload = [
        {"text": "First entry", "title": "Doc A", "section": "intro"},
        {"text": "Second entry", "source": "custom_source"},
    ]

    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps(payload), encoding="utf-8")

    docs = load_document(json_file)

    assert [d.title for d in docs] == ["Doc A", "data"]
    assert docs[0].section == "intro"
    assert docs[1].source == "custom_source"


def test_json_dict_payload(tmp_path: Path) -> None:
    payload = {"text": "Standalone doc", "title": "Single"}
    json_file = tmp_path / "single.json"
    json_file.write_text(json.dumps(payload), encoding="utf-8")

    docs = load_document(json_file)

    assert len(docs) == 1
    assert docs[0].title == "Single"
    assert docs[0].text == "Standalone doc"


def test_missing_text_field_raises(tmp_path: Path) -> None:
    json_file = tmp_path / "bad.json"
    json_file.write_text(json.dumps([{"title": "no text"}]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(json_file)


def test_deterministic_doc_id(tmp_path: Path) -> None:
    txt_file = tmp_path / "note.txt"
    txt_file.write_text("repeatable text", encoding="utf-8")

    first = load_document(txt_file)[0]
    second = load_document(txt_file)[0]

    assert first.doc_id == second.doc_id


def test_ingest_documents_scans_directories(tmp_path: Path) -> None:
    txt_file = tmp_path / "a.txt"
    md_file = tmp_path / "nested" / "b.md"
    md_file.parent.mkdir(parents=True, exist_ok=True)

    txt_file.write_text("hello", encoding="utf-8")
    md_file.write_text("# Title\nBody", encoding="utf-8")

    docs = ingest_documents([tmp_path])

    assert len(docs) == 2
    titles = sorted([d.title for d in docs])
    assert titles == ["Title", "a"]


def test_normalize_text_collapses_whitespace() -> None:
    raw = "line1  \n\n line2\t\tline3"
    assert normalize_text(raw) == "line1 line2 line3"
