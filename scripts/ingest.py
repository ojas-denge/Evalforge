from app.retrieval.embeddings import EmbeddingService
from app.retrieval.ingest import load_documents
from app.retrieval.vector_store import VectorStore


def main() -> None:
    print("Loading documents...")
    chunks = load_documents()

    print(f"Loaded chunks: {len(chunks)}")

    print("Loading embedding model...")
    embedding_service = EmbeddingService()

    print("Generating embeddings...")
    embeddings = embedding_service.embed_documents(
        [chunk.text for chunk in chunks]
    )

    print(f"Generated embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {len(embeddings[0])}")

    print("Writing to ChromaDB...")
    vector_store = VectorStore()

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(f"ChromaDB documents: {vector_store.count()}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
