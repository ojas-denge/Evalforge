import json
from time import perf_counter
from typing import Any

import httpx

from app.generation.base import (
    GenerationRequest,
    GenerationResult,
    Generator,
)
from app.generation.cost import CostCalculator
from app.generation.pricing_registry import PricingRegistry
from app.generation.usage import GenerationUsage
from app.observability.tracing import Tracer


class OpenAICompatibleGenerator(Generator):
    """Generator for APIs implementing the OpenAI-compatible chat interface."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        default_model: str,
        tracer: Tracer | None = None,
        cost_calculator: CostCalculator | None = None,
        pricing_registry: PricingRegistry | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.tracer = tracer or Tracer()
        self.cost_calculator = cost_calculator or CostCalculator()
        self.pricing_registry = pricing_registry or PricingRegistry()
        self.timeout = timeout

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        model = request.model or self.default_model

        messages: list[dict[str, str]] = []

        if request.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )

        context = "\n\n".join(
            (
                f"[{document.document_id} | "
                f"chunk={document.chunk_id} | "
                f"rank={document.rank}]\n"
                f"{document.text}"
            )
            for document in request.context
        )

        user_content = (
            f"Question:\n{request.question}\n\n"
            f"Retrieved context:\n{context}"
        )

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evalforge_response",
                    "schema": request.response_schema,
                },
            }

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start_time = perf_counter()

        with self.tracer.generation(
            name="generation",
            input={
                "question": request.question,
                "model": model,
                "context_count": len(request.context),
                "structured_output": (
                    request.response_schema is not None
                ),
            },
            metadata={
                "provider": "openai-compatible",
                "model": model,
                "base_url": self.base_url,
            },
        ) as observation:

            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            try:
                answer = data["choices"][0]["message"]["content"]
            except (
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                raise ValueError(
                    "Provider returned an invalid chat completion response"
                ) from exc

            structured_output = None

            if request.response_schema is not None:
                structured_output = self._parse_structured_output(
                    answer,
                    request.response_schema,
                )

            raw_usage = data.get("usage") or {}

            usage = GenerationUsage(
                input_tokens=int(
                    raw_usage.get("prompt_tokens", 0)
                ),
                output_tokens=int(
                    raw_usage.get("completion_tokens", 0)
                ),
            )

            pricing = self.pricing_registry.get(
                provider="openai-compatible",
                model=model,
            )

            estimated_cost_usd = self.cost_calculator.calculate(
                usage=usage,
                pricing=pricing,
            )

            latency_ms = (
                perf_counter() - start_time
            ) * 1000

            result = GenerationResult(
                answer=answer,
                model=model,
                provider="openai-compatible",
                usage=usage,
                estimated_cost_usd=estimated_cost_usd,
                latency_ms=latency_ms,
                finish_reason=data.get(
                    "choices",
                    [{}],
                )[0].get("finish_reason"),
                structured_output=structured_output,
                metadata={
                    "usage": raw_usage,
                },
            )

            if observation is not None:
                observation.update(
                    output={
                        "answer": answer,
                        "structured_output": structured_output,
                    },
                    metadata={
                        "provider": result.provider,
                        "model": result.model,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "estimated_cost_usd": (
                            result.estimated_cost_usd
                        ),
                        "latency_ms": latency_ms,
                        "structured_output": (
                            request.response_schema is not None
                        ),
                    },
                )

            return result

    @staticmethod
    def _parse_structured_output(
        content: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Structured output is not valid JSON"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Structured output must be a JSON object"
            )

        OpenAICompatibleGenerator._validate_json_schema(
            parsed,
            schema,
            path="$",
        )

        return parsed

    @staticmethod
    def _validate_json_schema(
        value: Any,
        schema: dict[str, Any],
        path: str,
    ) -> None:
        expected_type = schema.get("type")

        if expected_type == "object":
            if not isinstance(value, dict):
                raise ValueError(
                    f"Structured output at {path} must be an object"
                )

            properties = schema.get("properties", {})
            required = schema.get("required", [])

            for field in required:
                if field not in value:
                    raise ValueError(
                        f"Structured output missing required field: "
                        f"{path}.{field}"
                    )

            if schema.get("additionalProperties") is False:
                unexpected = set(value) - set(properties)

                if unexpected:
                    raise ValueError(
                        "Structured output contains unexpected fields: "
                        f"{sorted(unexpected)}"
                    )

            for field, field_schema in properties.items():
                if field in value:
                    OpenAICompatibleGenerator._validate_json_schema(
                        value[field],
                        field_schema,
                        f"{path}.{field}",
                    )

            return

        if expected_type == "string":
            if not isinstance(value, str):
                raise ValueError(
                    f"Structured output at {path} must be a string"
                )
            return

        if expected_type == "number":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise ValueError(
                    f"Structured output at {path} must be a number"
                )
            return

        if expected_type == "integer":
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
            ):
                raise ValueError(
                    f"Structured output at {path} must be an integer"
                )
            return

        if expected_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Structured output at {path} must be a boolean"
                )
            return

        if expected_type == "array":
            if not isinstance(value, list):
                raise ValueError(
                    f"Structured output at {path} must be an array"
                )

            item_schema = schema.get("items")

            if item_schema is not None:
                for index, item in enumerate(value):
                    OpenAICompatibleGenerator._validate_json_schema(
                        item,
                        item_schema,
                        f"{path}[{index}]",
                    )

            return

        if expected_type is None:
            return

        raise ValueError(
            f"Unsupported JSON Schema type at {path}: "
            f"{expected_type}"
        )
