import pytest

from app.evaluation.metrics import topic_coverage


def test_topic_coverage_returns_fraction_of_topics_found():
    score = topic_coverage(
        expected_topics=[
            "model routing",
            "latency",
            "cost",
        ],
        answer=(
            "Model routing should consider latency and cost."
        ),
    )

    assert score == pytest.approx(1.0)


def test_topic_coverage_detects_partial_coverage():
    score = topic_coverage(
        expected_topics=[
            "model routing",
            "latency",
            "cost",
        ],
        answer="Model routing should consider latency.",
    )

    assert score == pytest.approx(2 / 3)


def test_topic_coverage_is_case_insensitive():
    score = topic_coverage(
        expected_topics=["Model Routing"],
        answer="MODEL ROUTING is used here.",
    )

    assert score == 1.0


def test_topic_coverage_rejects_empty_topics():
    with pytest.raises(ValueError):
        topic_coverage(
            expected_topics=[],
            answer="Some answer.",
        )


def test_topic_coverage_rejects_empty_answer():
    with pytest.raises(ValueError):
        topic_coverage(
            expected_topics=["latency"],
            answer="",
        )
