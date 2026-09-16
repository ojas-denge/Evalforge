import json
from pathlib import Path

from app.evaluation import EvaluationDataset
from app.evaluation.ci_gate import EvaluationCIGate
from app.evaluation.comparison import compare_runs
from app.evaluation.regression import RegressionPolicy
from app.evaluation.runner import Evaluator
from app.models.evaluation import EvaluationRun


BASELINE_PATH = Path("evaluation_baselines/dense_baseline.json")


def main() -> None:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline evaluation not found: {BASELINE_PATH}"
        )

    baseline = EvaluationRun.model_validate(
        json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    )

    dataset = EvaluationDataset.load()
    candidate = Evaluator().evaluate_dataset(dataset)

    comparison = compare_runs(baseline, candidate)

    policy = RegressionPolicy(
        max_mrr_drop=0.0,
        max_latency_increase_ms=None,
    )

    gate = EvaluationCIGate(policy)
    result = gate.evaluate(comparison)

    print("=" * 80)
    print("EVALFORGE CI EVALUATION")
    print("=" * 80)

    print(f"Baseline:  {baseline.run_id}")
    print(f"Candidate: {candidate.run_id}")
    print()

    for metric, delta in comparison.metric_deltas.items():
        print(f"{metric}: {delta:+.6f}")

    print()
    print(f"CI Gate: {'PASS' if result.passed else 'FAIL'}")

    if result.reasons:
        print("Regression reasons:")
        for reason in result.reasons:
            print(f"  - {reason}")

    print("=" * 80)

    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

