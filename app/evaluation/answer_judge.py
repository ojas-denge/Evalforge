from typing import Any

from app.generation.base import GenerationRequest, Generator
from app.models.evaluation import (
    AnswerJudgeResult,
    RetrievedEvidence,
)


ANSWER_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_correct": {
            "type": "boolean",
        },
        "answer_grounded": {
            "type": "boolean",
        },
        "topics_covered": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "topics_missing": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "unsupported_claims": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "reasoning": {
            "type": "string",
        },
    },
    "required": [
        "answer_correct",
        "answer_grounded",
        "topics_covered",
        "topics_missing",
        "unsupported_claims",
        "reasoning",
    ],
    "additionalProperties": False,
}


ANSWER_JUDGE_SYSTEM_PROMPT = """
You are an evaluation judge for an LLM application.

Evaluate the generated answer against:
1. The user's question.
2. The expected reference answer.
3. The expected topics.
4. The retrieved evidence.

Judge only what is supported by the supplied information.

answer_correct:
True only when the generated answer correctly addresses the question
and does not materially contradict the expected answer.

answer_grounded:
True only when the substantive claims in the generated answer are
supported by the retrieved evidence.

topics_covered:
List the expected topics that are actually addressed by the generated
answer.

topics_missing:
List expected topics that are not adequately addressed.

unsupported_claims:
List substantive claims in the generated answer that are not supported
by the retrieved evidence. Use an empty list when there are none.

reasoning:
Briefly explain the judgment using the supplied reference and evidence.
""".strip()


class AnswerJudge:
    """Provider-independent structured judge for generated answers."""

    def __init__(
        self,
        generator: Generator,
    ) -> None:
        self.generator = generator

    def judge(
        self,
        question: str,
        expected_answer: str,
        expected_topics: list[str],
        generated_answer: str,
        retrieved_evidence: list[RetrievedEvidence],
    ) -> AnswerJudgeResult:

        evidence = "\n\n".join(
            (
                f"[{item.document_id} | "
                f"chunk={item.chunk_id} | "
                f"rank={item.rank}]\n"
                f"{item.text}"
            )
            for item in retrieved_evidence
        )

        request = GenerationRequest(
            question=(
                f"Question:\n{question}\n\n"
                f"Expected answer:\n{expected_answer}\n\n"
                f"Expected topics:\n{expected_topics}\n\n"
                f"Generated answer:\n{generated_answer}\n\n"
                f"Retrieved evidence:\n{evidence}"
            ),
            context=[],
            model=None,
            temperature=0.0,
            system_prompt=ANSWER_JUDGE_SYSTEM_PROMPT,
            response_schema=ANSWER_JUDGE_SCHEMA,
        )

        result = self.generator.generate(request)

        if result.structured_output is None:
            raise ValueError(
                "Answer judge requires structured output"
            )

        return AnswerJudgeResult.model_validate(
            result.structured_output
        )
