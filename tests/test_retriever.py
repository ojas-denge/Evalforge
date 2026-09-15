from app.retrieval.retriever import Retriever


class FakeEmbeddingService:
    def embed_query(self, query: str) -> list[float]:
        return [1.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.requested_top_k: int | None = None

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        self.requested_top_k = top_k
        return [
            {
                "chunk_id": "chunk-a",
                "document": "first candidate",
                "metadata": {"document_id": "a.md"},
                "distance": 0.1,
            },
            {
                "chunk_id": "chunk-b",
                "document": "second candidate",
                "metadata": {"document_id": "b.md"},
                "distance": 0.2,
            },
            {
                "chunk_id": "chunk-c",
                "document": "third candidate",
                "metadata": {"document_id": "c.md"},
                "distance": 0.3,
            },
        ][:top_k]


class FakeReranker:
    def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.2, 0.9, 0.4][: len(documents)]


def test_dense_retrieval_preserves_vector_store_order() -> None:
    vector_store = FakeVectorStore()
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        reranking_enabled=False,
    )

    result = retriever.retrieve("query", top_k=2)

    assert retriever.mode == "dense"
    assert vector_store.requested_top_k == 2
    assert [item.document_id for item in result.results] == ["a.md", "b.md"]


def test_reranking_rescores_expanded_dense_candidates() -> None:
    vector_store = FakeVectorStore()
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        reranker=FakeReranker(),
        reranker_candidate_k=3,
    )

    result = retriever.retrieve("query", top_k=2)

    assert retriever.mode == "dense_reranked"
    assert vector_store.requested_top_k == 3
    assert [item.document_id for item in result.results] == ["b.md", "c.md"]
    assert [item.rank for item in result.results] == [1, 2]
