# Latency Policy

EvalForge tracks latency for every query.

Latency is measured from the beginning of request processing until the final response is produced.

The system records total latency as well as component-level latency when instrumentation is available.

The target baseline for a normal query is a p95 latency below 2500 milliseconds.

A p95 latency above 2500 milliseconds is considered a performance regression for the baseline application.

Latency measurements must be evaluated together with answer quality.

Reducing latency at the expense of retrieval quality or groundedness should not automatically be considered an improvement.
