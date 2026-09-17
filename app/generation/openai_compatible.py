from time import perf_counter

import httpx

from app.generation.base import GenerationRequest, GenerationResult, Generator
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

        messages = []

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

        payload = {
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
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    "Provider returned an invalid chat completion response"
                ) from exc

            raw_usage = data.get("usage") or {}

            usage = GenerationUsage(
                input_tokens=int(raw_usage.get("prompt_tokens", 0)),
                output_tokens=int(raw_usage.get("completion_tokens", 0)),
            )

            pricing = self.pricing_registry.get(
                provider="openai-compatible",
                model=model,
            )

            estimated_cost_usd = self.cost_calculator.calculate(
                usage=usage,
                pricing=pricing,
            )

            latency_ms = (perf_counter() - start_time) * 1000

            result = GenerationResult(
                answer=answer,
                model=model,
                provider="openai-compatible",
                usage=usage,
                estimated_cost_usd=estimated_cost_usd,
                latency_ms=latency_ms,
                finish_reason=data.get("choices", [{}])[0].get("finish_reason"),
                metadata={
                    "usage": raw_usage,
                },
            )

            if observation is not None:
                observation.update(
                    output={
                        "answer": answer,
                    },
                    metadata={
                        "provider": result.provider,
                        "model": result.model,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "estimated_cost_usd": result.estimated_cost_usd,
                        "latency_ms": latency_ms,
                    },
                )

            return result
