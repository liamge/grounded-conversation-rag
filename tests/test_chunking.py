
from src.chunking import Chunk, chunk_document, chunk_documents
from src.ingestion import Document


def _doc(text: str) -> Document:
    return Document(doc_id="doc_123", text=text, source="source.txt", title="Title")


def test_sentence_boundaries_are_preserved():
    doc = _doc("Alpha one. Beta two. Gamma three.")

    chunks = chunk_document(doc, chunk_size=25, chunk_overlap=0)

    assert len(chunks) == 2
    assert chunks[0].text == "Alpha one. Beta two."
    assert chunks[1].text == "Gamma three."
    # offsets should point to the start of each chunk in normalized text
    assert chunks[0].metadata["offset"] == "0"
    assert chunks[1].metadata["offset"] == "21"


def test_long_sentence_is_split_with_overlap():
    text = "a" * 40  # single long sentence with no punctuation
    doc = _doc(text)

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    # Expect windows starting at 0,8,16,24,32
    expected_offsets = ["0", "8", "16", "24", "32"]
    assert [c.metadata["offset"] for c in chunks] == expected_offsets
    # Overlap of 2 means consecutive chunks share 2 characters
    for first, second in zip(chunks, chunks[1:]):
        assert first.text[-2:] == second.text[:2]


def test_chunk_ids_are_deterministic():
    doc = _doc("One. Two. Three.")

    first = chunk_document(doc, chunk_size=12, chunk_overlap=0)
    second = chunk_document(doc, chunk_size=12, chunk_overlap=0)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_documents_maps_multiple_docs():
    docs = [
        _doc("Doc one sentence."),
        _doc("Doc two sentence."),
    ]

    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=0)

    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert {c.source for c in chunks} == {"source.txt"}
