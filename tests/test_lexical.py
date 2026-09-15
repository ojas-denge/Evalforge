from app.retrieval.lexical import BM25Retriever


def test_bm25_prefers_the_document_with_matching_terms() -> None:
    results = BM25Retriever().search(
        "latency performance target",
        [
            {"chunk_id": "cost", "document": "Token cost policy."},
            {
                "chunk_id": "latency",
                "document": "The latency performance target is p95.",
            },
        ],
        top_k=2,
    )

    assert [result["chunk_id"] for result in results] == ["latency"]
