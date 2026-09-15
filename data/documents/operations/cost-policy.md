# Cost Policy

EvalForge records token usage and estimates the cost of each LLM request.

Input and output tokens are tracked separately.

Cost should be calculated using the configured model's applicable input and output token prices.

The baseline system should report average cost per request.

A lower cost is not automatically an improvement if answer quality decreases.

Cost optimization experiments must therefore be evaluated together with quality metrics and latency.
