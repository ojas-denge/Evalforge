from app.generation.base import GenerationRequest
from app.generation.deterministic import DeterministicGenerator
from app.retrieval.retriever import RetrievedDocument


def make_document(rank: int, text: str) -> RetrievedDocument:
    return RetrievedDocument(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        document_id=f"doc-{rank}",
        text=text,
        distance=0.1 * rank,
    )


def test_deterministic_generator_uses_all_context():
    generator = DeterministicGenerator()

    request = GenerationRequest(
        question="What are the important topics?",
        context=[
            make_document(1, "First evidence."),
            make_document(2, "Second evidence."),
            make_document(3, "Third evidence."),
        ],
    )

    result = generator.generate(request)

    assert result.answer == (
        "First evidence.\n\n"
        "Second evidence.\n\n"
        "Third evidence."
    )


def test_deterministic_generator_handles_empty_context():
    generator = DeterministicGenerator()

    request = GenerationRequest(
        question="What is the answer?",
        context=[],
    )

    result = generator.generate(request)

    assert result.answer == "No relevant context was retrieved."
