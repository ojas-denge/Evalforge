\# Reliability Policy



EvalForge tracks failures occurring during retrieval, generation, validation, and external service calls.



A successful request should not hide component-level failures.



Transient failures may be retried when retry behavior is explicitly configured.



Retries should be observable so that additional latency and repeated model calls can be measured.



Reliability improvements should be evaluated together with latency, cost, and answer quality.

