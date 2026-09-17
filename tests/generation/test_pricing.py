from app.generation.pricing import ModelPricing


def test_model_pricing_defaults_to_zero():
    pricing = ModelPricing()

    assert pricing.input_cost_per_1k_tokens == 0.0
    assert pricing.output_cost_per_1k_tokens == 0.0


def test_model_pricing_stores_token_rates():
    pricing = ModelPricing(
        input_cost_per_1k_tokens=0.003,
        output_cost_per_1k_tokens=0.015,
    )

    assert pricing.input_cost_per_1k_tokens == 0.003
    assert pricing.output_cost_per_1k_tokens == 0.015
