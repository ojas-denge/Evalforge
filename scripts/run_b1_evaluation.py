import json
from collections import Counter

from app.evaluation import EvaluationDataset
from app.evaluator import Evaluator


def main() -> None:
    evaluator = Evaluator()
    run = evaluator.evaluate_dataset(EvaluationDataset.load())

    print(
        json.dumps(
            {
                "dataset_size": run.dataset_size,
                "mode": evaluator.retriever.mode,
                "hit_at_1": run.mean_hit_at_1,
                "hit_at_3": run.mean_hit_at_3,
                "hit_at_5": run.mean_hit_at_5,
                "recall_at_1": run.mean_recall_at_1,
                "recall_at_3": run.mean_recall_at_3,
                "recall_at_5": run.mean_recall_at_5,
                "mrr": run.mean_mrr,
                "mean_retrieval_latency_ms": (
                    run.mean_retrieval_latency_ms
                ),
                "diagnostics": Counter(
                    result.failure_type
                    for result in run.results
                ),
            },
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
