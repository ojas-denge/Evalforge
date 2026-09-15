from dataclasses import dataclass
from time import perf_counter

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedDocument:
    rank: int
    chunk_id: str
    document_id: str
    text: str
    distance: float


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    results: list[RetrievedDocument]
    latency_ms: float


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.embedding_service = (
            embedding_service or EmbeddingService()
        )
        self.vector_store = (
            vector_store or VectorStore()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        start_time = perf_counter()

        query_embedding = self.embedding_service.embed_query(
            query
        )

        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        results = [
            RetrievedDocument(
                rank=rank,
                chunk_id=result["chunk_id"],
                document_id=result["metadata"]["document_id"],
                text=result["document"],
                distance=result["distance"],
            )
            for rank, result in enumerate(
                raw_results,
                start=1,
            )
        ]

        latency_ms = (perf_counter() - start_time) * 1000

        return RetrievalResult(
            query=query,
            results=results,
            latency_ms=latency_ms,
        )