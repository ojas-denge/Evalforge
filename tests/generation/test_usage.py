from app.generation.usage import GenerationUsage


def test_generation_usage_calculates_total_tokens():
    usage = GenerationUsage(
        input_tokens=25,
        output_tokens=8,
    )

    assert usage.input_tokens == 25
    assert usage.output_tokens == 8
    assert usage.total_tokens == 33


def test_generation_usage_defaults_to_zero():
    usage = GenerationUsage()

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0
