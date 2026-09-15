from pathlib import Path

import chromadb

from app.core.logging import get_logger
from app.retrieval.chunking import DocumentChunk

logger = get_logger(__name__)


class VectorStore:
    def __init__(self) -> None:
        persist_directory = Path("data/chroma")
        persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name="evalforge_documents",
            metadata={
                "description": "EvalForge baseline document collection"
            },
        )

        logger.info(
            "ChromaDB initialized at %s",
            persist_directory,
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        if not chunks:
            logger.warning("No chunks supplied to vector store")
            return

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
        )

        logger.info(
            "Stored %d chunks in ChromaDB",
            len(chunks),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        count = self.collection.count()

        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        matches = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return matches

    def count(self) -> int:
        return self.collection.count()

    def all_chunks(self) -> list[dict]:
        results = self.collection.get(
            include=["documents", "metadatas"],
        )

        return [
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "distance": None,
            }
            for chunk_id, document, metadata in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
            )
        ]
