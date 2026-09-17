import pytest
from pydantic import BaseModel

from app.generation.structured import StructuredOutputValidator


class AnswerSchema(BaseModel):
    answer: str
    confidence: float


def test_validator_returns_validated_model():
    validator = StructuredOutputValidator()

    result = validator.validate(
        {"answer": "Python is a language.", "confidence": 0.95},
        AnswerSchema,
    )

    assert isinstance(result, AnswerSchema)
    assert result.answer == "Python is a language."
    assert result.confidence == 0.95


def test_validator_rejects_invalid_data():
    validator = StructuredOutputValidator()

    with pytest.raises(ValueError):
        validator.validate(
            {
                "answer": "Python is a language.",
                "confidence": "not-a-number",
            },
            AnswerSchema,
        )
