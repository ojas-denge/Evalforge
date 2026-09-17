from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_1k_tokens: float = 0.0
    output_cost_per_1k_tokens: float = 0.0
