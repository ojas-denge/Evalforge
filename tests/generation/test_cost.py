from app.generation.cost import CostCalculator
from app.generation.pricing import ModelPricing
from app.generation.usage import GenerationUsage


def test_cost_calculator_calculates_input_and_output_cost():
    usage = GenerationUsage(
        input_tokens=2000,
        output_tokens=1000,
    )

    pricing = ModelPricing(
        input_cost_per_1k_tokens=0.003,
        output_cost_per_1k_tokens=0.015,
    )

    calculator = CostCalculator()

    cost = calculator.calculate(
        usage=usage,
        pricing=pricing,
    )

    assert cost == 0.021


def test_cost_calculator_returns_zero_for_zero_usage():
    usage = GenerationUsage()

    pricing = ModelPricing(
        input_cost_per_1k_tokens=0.003,
        output_cost_per_1k_tokens=0.015,
    )

    calculator = CostCalculator()

    assert calculator.calculate(usage, pricing) == 0.0
