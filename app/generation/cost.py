from app.generation.pricing import ModelPricing
from app.generation.usage import GenerationUsage


class CostCalculator:
    """Calculate generation cost from normalized token usage and pricing."""

    def calculate(
        self,
        usage: GenerationUsage,
        pricing: ModelPricing,
    ) -> float:
        input_cost = (
            usage.input_tokens / 1000
        ) * pricing.input_cost_per_1k_tokens

        output_cost = (
            usage.output_tokens / 1000
        ) * pricing.output_cost_per_1k_tokens

        return round(input_cost + output_cost, 10)
