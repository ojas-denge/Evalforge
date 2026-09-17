from app.core.config import Settings
from app.generation.cost import CostCalculator
from app.generation.deterministic import DeterministicGenerator
from app.generation.factory import create_generator
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.generation.pricing_registry import PricingRegistry


def test_factory_creates_deterministic_generator():
    settings = Settings(
        llm_provider="deterministic",
        llm_model="deterministic-test",
    )

    generator = create_generator(settings)

    assert isinstance(generator, DeterministicGenerator)


def test_factory_creates_openai_compatible_generator():
    settings = Settings(
        llm_provider="openai-compatible",
        llm_model="test-model",
        llm_api_key="test-key",
        llm_base_url="http://localhost:9000/v1",
    )

    generator = create_generator(settings)

    assert isinstance(generator, OpenAICompatibleGenerator)
    assert generator.default_model == "test-model"
    assert generator.base_url == "http://localhost:9000/v1"


def test_factory_injects_cost_dependencies():
    settings = Settings(
        llm_provider="openai-compatible",
        llm_model="test-model",
        llm_api_key="test-key",
        llm_base_url="http://localhost:9000/v1",
    )

    cost_calculator = CostCalculator()
    pricing_registry = PricingRegistry()

    generator = create_generator(
        settings,
        cost_calculator=cost_calculator,
        pricing_registry=pricing_registry,
    )

    assert generator.cost_calculator is cost_calculator
    assert generator.pricing_registry is pricing_registry


def test_factory_requires_base_url_for_openai_compatible():
    settings = Settings(
        llm_provider="openai-compatible",
        llm_model="test-model",
        llm_api_key="test-key",
    )

    try:
        create_generator(settings)
    except ValueError as exc:
        assert "LLM_BASE_URL" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
