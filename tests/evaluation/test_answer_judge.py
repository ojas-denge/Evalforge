import pytest

from app.evaluation.answer_judge import (
    ANSWER_JUDGE_SCHEMA,
    ANSWER_JUDGE_SYSTEM_PROMPT,
    AnswerJudge,
)
from app.generation.base import (
    GenerationRequest,
    GenerationResult,
    Generator,
)
from app.generation.usage import GenerationUsage
from app.models.evaluation import RetrievedEvidence


class FakeJudgeGenerator(Generator):
    def __init__(self, structured_output):
        self.structured_output = structured_output
        self.last_request = None

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        self.last_request = request

        return GenerationResult(
            answer="judge response",
            model="fake-judge",
            provider="fake",
            usage=GenerationUsage(),
            estimated_cost_usd=0.0,
            latency_ms=1.0,
            structured_output=self.structured_output,
        )


def _evidence() -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence(
            rank=1,
            chunk_id="chunk-1",
            document_id="doc-1",
            distance=0.1,
            text="The target p95 latency is below 2500 milliseconds.",
        )
    ]


def _judge_output() -> dict:
    return {
        "answer_correct": True,
        "answer_grounded": True,
        "topics_covered": ["latency"],
        "topics_missing": [],
        "unsupported_claims": [],
        "reasoning": "The answer is supported by the supplied evidence.",
    }


def test_answer_judge_builds_structured_request():
    generator = FakeJudgeGenerator(_judge_output())
    judge = AnswerJudge(generator)

    result = judge.judge(
        question="What is the latency target?",
        expected_answer="The target is below 2500 milliseconds.",
        expected_topics=["latency"],
        generated_answer="The target is below 2500 milliseconds.",
        retrieved_evidence=_evidence(),
    )

    assert result.answer_correct is True
    assert result.answer_grounded is True

    request = generator.last_request

    assert request.system_prompt == ANSWER_JUDGE_SYSTEM_PROMPT
    assert request.response_schema == ANSWER_JUDGE_SCHEMA
    assert request.temperature == 0.0

    assert "What is the latency target?" in request.question
    assert "below 2500 milliseconds" in request.question
    assert "latency" in request.question
    assert "chunk-1" in request.question
    assert "2500 milliseconds" in request.question


def test_answer_judge_rejects_missing_structured_output():
    generator = FakeJudgeGenerator(None)
    judge = AnswerJudge(generator)

    with pytest.raises(
        ValueError,
        match="requires structured output",
    ):
        judge.judge(
            question="What is the latency target?",
            expected_answer="The target is below 2500 milliseconds.",
            expected_topics=["latency"],
            generated_answer="The target is below 2500 milliseconds.",
            retrieved_evidence=_evidence(),
        )


def test_answer_judge_rejects_invalid_structured_output():
    generator = FakeJudgeGenerator(
        {
            "answer_correct": "yes",
            "answer_grounded": True,
            "topics_covered": ["latency"],
            "topics_missing": [],
            "unsupported_claims": [],
            "reasoning": "Invalid boolean value.",
        }
    )

    judge = AnswerJudge(generator)

    with pytest.raises(ValueError):
        judge.judge(
            question="What is the latency target?",
            expected_answer="The target is below 2500 milliseconds.",
            expected_topics=["latency"],
            generated_answer="The target is below 2500 milliseconds.",
            retrieved_evidence=_evidence(),
        )
