from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class StructuredOutputValidator:
    """Validate structured generation output against a Pydantic model."""

    def validate(
        self,
        data: dict[str, Any],
        schema: type[T],
    ) -> T:
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise ValueError(
                "Structured output failed schema validation"
            ) from exc
