# Output Validation Policy

Generated answers must conform to the application's structured response schema.

The response schema contains an answer, citations, confidence score, human-review indicator, and execution metrics.

Confidence values must be between zero and one.

Citations should identify the source documents used to construct the answer.

Malformed structured output must not be silently accepted.

Validation failures should be recorded and may trigger a retry or repair operation.

A successful LLM call does not imply that the output is valid.
