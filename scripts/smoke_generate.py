# scripts/smoke_generate.py
from src.chunking import chunk_document
from src.config import Settings
from src.generation import assemble_context, generate_answer
from src.ingestion import Document
from src.retrieval import BM25Retriever, TfidfRetriever


def main():
    settings = Settings()

    # Tiny toy corpus
    doc = Document(
        doc_id="d0",
        text="Cats love naps and chasing laser pointers. Dogs enjoy walks and fetching balls.",
        source="toy",
        title="Pets 101",
    )
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    print(f"Chunked corpus: {len(chunks)} chunk(s)")
    for ch in chunks:
        print(f"- {ch.chunk_id}: {ch.text}")

    # Retrieval (BM25 with TF-IDF fallback)
    query = "What do cats like to chase?"
    try:
        retriever = BM25Retriever(top_k=3, min_score=float("-inf"))
        retriever.index(chunks)
        results = retriever.search(query)
    except Exception as exc:  # pragma: no cover - debugging convenience
        print(f"[warn] BM25 unavailable ({exc}); falling back to TF-IDF.")
        retriever = TfidfRetriever(top_k=3)
        retriever.index(chunks)
        results = retriever.search(query)

    if not results:
        print("[info] BM25 returned no hits; retrying with TF-IDF.")
        retriever = TfidfRetriever(top_k=3)
        retriever.index(chunks)
        results = retriever.search(query)

    print("Retrieved:", [(r.chunk.chunk_id, round(r.score, 3)) for r in results])
    print("\nContext block:\n", assemble_context(results))

    # Generation
    answer = generate_answer(query, results, settings)
    print("Query:", query)
    print("Answer:", answer.answer)
    print("Citations:", answer.citations)
    print("Provider:", answer.provider)
    print("Model:", answer.model)

if __name__ == "__main__":
    main()
