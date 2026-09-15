from app.retrieval.retriever import Retriever


def main() -> None:
    retriever = Retriever()

    results = retriever.retrieve(
        "How does model routing work?",
        top_k=3,
    )

    print("Retriever OK")
    print()

    for rank, result in enumerate(results, start=1):
        document_id = result["metadata"]["document_id"]
        distance = result["distance"]

        print(
            f"{rank}. {document_id} | "
            f"distance={distance:.4f}"
        )


if __name__ == "__main__":
    main()