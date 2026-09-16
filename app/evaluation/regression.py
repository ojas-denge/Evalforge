from dataclasses import dataclass, field

from app.evaluation.comparison import EvaluationComparison


@dataclass(frozen=True)
class RegressionResult:
    regression_detected: bool
    reasons: list[str] = field(default_factory=list)


class RegressionPolicy:
    def __init__(
        self,
        *,
        max_mrr_drop: float = 0.0,
        max_latency_increase_ms: float = 0.0,
    ) -> None:
        if max_mrr_drop < 0:
            raise ValueError("max_mrr_drop must be non-negative")

        if max_latency_increase_ms < 0:
            raise ValueError(
                "max_latency_increase_ms must be non-negative"
            )

        self.max_mrr_drop = max_mrr_drop
        self.max_latency_increase_ms = max_latency_increase_ms

    def evaluate(
        self,
        comparison: EvaluationComparison,
    ) -> RegressionResult:
        reasons: list[str] = []

        mrr_delta = comparison.metric_deltas["mean_mrr"]

        if mrr_delta < -self.max_mrr_drop:
            reasons.append("mean_mrr")

        latency_delta = comparison.metric_deltas[
            "mean_retrieval_latency_ms"
        ]

        if latency_delta > self.max_latency_increase_ms:
            reasons.append("mean_retrieval_latency_ms")

        return RegressionResult(
            regression_detected=bool(reasons),
            reasons=reasons,
        )