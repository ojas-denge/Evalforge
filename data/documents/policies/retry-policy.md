\# Retry Policy



Retries are intended to recover from transient failures.



A retry should not silently replace the original failed execution in observability data.



Each retry attempt should be recorded with its reason and outcome.



Retries can increase latency and cost, so retry behavior must be included in system evaluation.



Repeated retries should be bounded to prevent uncontrolled execution.

