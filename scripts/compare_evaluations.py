from app.evaluation.comparison import compare_runs
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.regression import RegressionPolicy
from app.evaluation.reporting import print_comparison
from app.evaluation.runner import Evaluator
from app.retrieval.retriever import Retriever

def main():
    dataset = EvaluationDataset.load(
        "data/evaluation_cases.json"
    )

    baseline_retriever = Retriever(
        reranking_enabled=False,
    )

    candidate_retriever = Retriever(
        reranking_enabled=True,
        reranker_candidate_k=10,
    )

    baseline_evaluator = Evaluator(
        retriever=baseline_retriever,
    )

    candidate_evaluator = Evaluator(
        retriever=candidate_retriever,
    )

    print("Running baseline evaluation...")
    baseline = baseline_evaluator.evaluate_dataset(dataset)

    print("Running candidate evaluation...")
    candidate = candidate_evaluator.evaluate_dataset(dataset)

    comparison = compare_runs(
        baseline,
        candidate,
    )

    policy = RegressionPolicy(
        max_mrr_drop=0.02,
        max_latency_increase_ms=100.0,
    )

    regression = policy.evaluate(comparison)

    print("\n================================")
    print("        EVALFORGE COMPARISON")
    print("================================")

    print(f"\nBaseline:  {baseline.run_id}")
    print(f"Candidate: {candidate.run_id}")

    print_comparison(comparison)

    print("\n=== Regression Policy ===")
    print(
        f"Regression detected: "
        f"{regression.regression_detected}"
    )

    if regression.reasons:
        print("Reasons:")
        for reason in regression.reasons:
            print(f"  - {reason}")
    else:
        print("Reasons: none")


if __name__ == "__main__":
    main()