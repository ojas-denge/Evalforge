from app.generation.pricing import ModelPricing
from app.generation.pricing_registry import PricingRegistry


def test_pricing_registry_returns_registered_model_pricing():
    pricing = ModelPricing(
        input_cost_per_1k_tokens=0.003,
        output_cost_per_1k_tokens=0.015,
    )

    registry = PricingRegistry(
        {
            ("openai-compatible", "test-model"): pricing,
        }
    )

    result = registry.get(
        provider="openai-compatible",
        model="test-model",
    )

    assert result == pricing


def test_pricing_registry_returns_zero_pricing_for_unknown_model():
    registry = PricingRegistry()

    result = registry.get(
        provider="unknown",
        model="unknown-model",
    )

    assert result == ModelPricing()
