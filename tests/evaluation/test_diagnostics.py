import pytest

from app.evaluation.diagnostics import analyze_retrieval


@pytest.mark.parametrize(
    ("expected_documents", "retrieved_documents", "failure_type"),
    [
        (["a.md"], ["a.md", "b.md"], "PASS"),
        (["a.md"], ["b.md", "a.md"], "RANKING_FAILURE"),
        (["a.md", "b.md"], ["a.md", "c.md"], "PARTIAL_RECALL"),
        (["a.md"], ["b.md", "c.md"], "RETRIEVAL_FAILURE"),
        (
            ["a.md", "b.md"],
            ["c.md", "a.md", "b.md"],
            "MULTI_EVIDENCE_RANKING_FAILURE",
        ),
    ],
)
def test_analyze_retrieval_classifies_failures(
    expected_documents: list[str],
    retrieved_documents: list[str],
    failure_type: str,
) -> None:
    diagnostic = analyze_retrieval(
        expected_documents,
        retrieved_documents,
    )

    assert diagnostic.failure_type == failure_type


def test_analyze_retrieval_reports_diagnostic_details() -> None:
    diagnostic = analyze_retrieval(
        ["a.md", "b.md"],
        ["noise.md", "b.md", "noise.md"],
    )

    assert diagnostic.relevant_documents_found is True
    assert diagnostic.first_relevant_rank == 2
    assert diagnostic.missing_documents == ["a.md"]
    assert diagnostic.confounding_documents == ["noise.md", "noise.md"]


def test_analyze_retrieval_requires_expected_documents() -> None:
    with pytest.raises(ValueError, match="expected_documents cannot be empty"):
        analyze_retrieval([], ["a.md"])
