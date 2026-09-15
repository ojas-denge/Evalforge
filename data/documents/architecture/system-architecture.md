# System Architecture

EvalForge is an LLM evaluation and observability platform designed to test retrieval-augmented generation and agentic LLM applications.

The platform consists of an API layer, orchestration layer, retrieval layer, generation layer, validation layer, observability layer, and evaluation layer.

The API layer exposes application endpoints through FastAPI.

The orchestration layer controls execution flow and may route requests through retrieval, tools, generation, and validation.

The retrieval layer searches the indexed knowledge base and provides relevant context to the generation layer.

The generation layer uses an LLM to produce an answer from the user request and retrieved context.

The validation layer verifies that generated responses conform to required schemas and application constraints.

The observability layer records traces, latency, token usage, costs, errors, and execution metadata.

The evaluation layer measures system quality and detects regressions between application versions.
