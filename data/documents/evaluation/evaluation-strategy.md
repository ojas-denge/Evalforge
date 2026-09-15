\# Evaluation Strategy



EvalForge evaluates LLM applications using a fixed golden dataset.



Evaluation should separate retrieval quality, answer quality, latency, cost, and structured-output validity.



A single metric should not determine whether a system change is better.



Evaluation runs should record the application configuration used during the run.



The same evaluation dataset should be used when comparing two application versions so that changes can be attributed to the system rather than differences in test cases.



Evaluation results should preserve per-case evidence so that aggregate regressions can be investigated.

