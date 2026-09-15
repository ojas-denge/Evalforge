from app.retrieval.retriever import Retriever


def main() -> None:
    retriever = Retriever()

    result = retriever.retrieve(
        "How does model routing work?",
        top_k=3,
    )

    print("Retriever OK")
    print(f"Query: {result.query}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    print()

    for document in result.results:
        print(
            f"{document.rank}. "
            f"{document.document_id} | "
            f"distance={document.distance:.4f}"
        )


if __name__ == "__main__":
    main()