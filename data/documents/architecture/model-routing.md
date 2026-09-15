# Model Routing

EvalForge may support multiple LLM providers and models.

Model selection should consider task complexity, latency, reliability, and cost.

A simple request does not necessarily require the most capable model.

Model routing experiments should record which model handled each request.

Changing the model can affect answer quality, latency, token usage, and cost.

Therefore model changes must be evaluated against a fixed evaluation dataset rather than judged from a small number of manually selected examples.
