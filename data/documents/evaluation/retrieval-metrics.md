\# Retrieval Metrics



EvalForge uses retrieval metrics to measure whether relevant evidence is returned by the retriever.



Hit@K measures whether at least one expected document appears within the first K retrieved results.



Recall@K measures the proportion of expected documents retrieved within the first K results.



Reciprocal rank measures how highly the first relevant document appears in the ranked results.



Metrics should be interpreted together because a system can retrieve all required evidence while still ranking the most important evidence poorly.



Retrieval metrics should be evaluated across a representative golden dataset rather than a small collection of manually selected queries.

