from collections.abc import Sequence


def reciprocal_rank(
    expected_documents: Sequence[str],
    retrieved_documents: Sequence[str],
) -> float:
    if not expected_documents:
        raise ValueError("expected_documents cannot be empty")

    expected = set(expected_documents)

    for rank, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        if document in expected:
            return 1.0 / rank

    return 0.0

def hit_at_k(
    expected_documents: Sequence[str],
    retrieved_documents: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not expected_documents:
        raise ValueError("expected_documents cannot be empty")

    retrieved_top_k = set(retrieved_documents[:k])

    return float(
        any(
            document in retrieved_top_k
            for document in expected_documents
        )
    )


def recall_at_k(
    expected_documents: Sequence[str],
    retrieved_documents: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not expected_documents:
        raise ValueError("expected_documents cannot be empty")

    retrieved_top_k = set(retrieved_documents[:k])

    relevant_retrieved = sum(
        document in retrieved_top_k
        for document in expected_documents
    )

    return relevant_retrieved / len(expected_documents)