from unittest.mock import patch

from app.generation.base import GenerationRequest
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.retrieval.retriever import RetrievedDocument


def test_openai_compatible_generator_parses_response():
    document = RetrievedDocument(
        rank=1,
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Python is a programming language.",
        distance=0.1,
    )

    request = GenerationRequest(
        question="What is Python?",
        context=[document],
        model="test-model",
    )

    response_payload = {
        "choices": [
            {
                "message": {
                    "content": "Python is a programming language."
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 8,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        result = generator.generate(request)

    assert result.answer == "Python is a programming language."
    assert result.structured_output is None
    assert result.model == "test-model"
    assert result.provider == "openai-compatible"
    assert result.usage.input_tokens == 25
    assert result.usage.output_tokens == 8
    assert result.usage.total_tokens == 33
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0
    assert result.estimated_cost_usd == 0.0

    mock_post.assert_called_once()

    call = mock_post.call_args
    assert call.args[0] == "http://localhost:9000/v1/chat/completions"
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-key"

    payload = call.kwargs["json"]

    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.0
    assert payload["messages"][0]["role"] == "user"
    assert "What is Python?" in payload["messages"][0]["content"]
    assert "Python is a programming language." in payload["messages"][0]["content"]


def test_openai_compatible_generator_supports_structured_output():
    document = RetrievedDocument(
        rank=1,
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Python is a programming language.",
        distance=0.1,
    )

    response_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }

    request = GenerationRequest(
        question="What is Python?",
        context=[document],
        model="test-model",
        response_schema=response_schema,
    )

    response_payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"answer":"Python is a programming language.",'
                        '"confidence":0.95}'
                    )
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 15,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        result = generator.generate(request)

    assert result.answer == (
        '{"answer":"Python is a programming language.",'
        '"confidence":0.95}'
    )

    assert result.structured_output == {
        "answer": "Python is a programming language.",
        "confidence": 0.95,
    }

    payload = mock_post.call_args.kwargs["json"]

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "evalforge_response",
            "schema": response_schema,
        },
    }
from unittest.mock import patch

import pytest

from app.generation.base import GenerationRequest
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.retrieval.retriever import RetrievedDocument


def _structured_request() -> GenerationRequest:
    document = RetrievedDocument(
        rank=1,
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Python is a programming language.",
        distance=0.1,
    )

    return GenerationRequest(
        question="What is Python?",
        context=[document],
        model="test-model",
        response_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        },
    )


def test_openai_compatible_generator_rejects_invalid_structured_json():
    request = _structured_request()

    response_payload = {
        "choices": [
            {
                "message": {
                    "content": '{"answer":"Python is a language."'
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 10,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        with pytest.raises(ValueError, match="valid JSON"):
            generator.generate(request)


def test_openai_compatible_generator_rejects_schema_violation():
    request = _structured_request()

    response_payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"answer":"Python is a language.",'
                        '"confidence":"high"}'
                    )
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 10,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        with pytest.raises(ValueError, match="must be a number"):
            generator.generate(request)
