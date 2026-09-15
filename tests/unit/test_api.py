from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "evalforge"
    assert data["version"] == "0.1.0"


def test_query_contract():
    response = client.post(
        "/query",
        json={"question": "What is EvalForge?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "citations" in data
    assert "confidence" in data
    assert "latency_ms" in data
    assert "input_tokens" in data
    assert "output_tokens" in data
    assert "estimated_cost_usd" in data


def test_query_rejects_empty_question():
    response = client.post(
        "/query",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_query_rejects_missing_question():
    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 422