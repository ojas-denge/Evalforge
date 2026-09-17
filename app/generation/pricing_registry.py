from app.generation.pricing import ModelPricing


DEFAULT_PRICING: dict[tuple[str, str], ModelPricing] = {
    # Pricing can be added here as providers/models are supported.
    # Values are USD per 1K tokens.
}


class PricingRegistry:
    """Resolve model pricing independently of generation providers."""

    def __init__(
        self,
        pricing: dict[tuple[str, str], ModelPricing] | None = None,
    ):
        self._pricing = dict(pricing or DEFAULT_PRICING)

    def get(self, provider: str, model: str) -> ModelPricing:
        return self._pricing.get(
            (provider, model),
            ModelPricing(),
        )
