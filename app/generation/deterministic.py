from time import perf_counter

from app.generation.base import GenerationRequest, GenerationResult, Generator
from app.generation.usage import GenerationUsage
from app.observability.tracing import Tracer


class DeterministicGenerator(Generator):
    """Deterministic generator used for local integration tests."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer()

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        start_time = perf_counter()

        with self.tracer.generation(
            name="generation",
            input={
                "question": request.question,
                "context": [
                    {
                        "chunk_id": document.chunk_id,
                        "document_id": document.document_id,
                        "rank": document.rank,
                        "text": document.text,
                    }
                    for document in request.context
                ],
            },
            metadata={
                "provider": "deterministic",
                "model": request.model or "deterministic-test",
            },
        ) as observation:

            answer = (
                request.context[0].text
                if request.context
                else "No relevant context was retrieved."
            )

            latency_ms = (perf_counter() - start_time) * 1000

            result = GenerationResult(
                answer=answer,
                model=request.model or "deterministic-test",
                provider="deterministic",
                usage=GenerationUsage(
                    input_tokens=0,
                    output_tokens=0,
                ),
                estimated_cost_usd=0.0,
                latency_ms=latency_ms,
                finish_reason="stop",
            )

            if observation is not None:
                observation.update(
                    output={"answer": result.answer},
                    metadata={
                        "provider": result.provider,
                        "model": result.model,
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                        "estimated_cost_usd": result.estimated_cost_usd,
                        "latency_ms": result.latency_ms,
                        "finish_reason": result.finish_reason,
                    },
                )

            return result
