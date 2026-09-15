# EvalForge

LLM Evaluation & Observability Platform.

## Phase 0

Foundation scaffold with FastAPI health endpoint.

## Run locally

```bash
python -m venv .venv
# activate the environment
pip install -e ".[test]"
uvicorn app.main:app --reload
```

Health: `GET /health`
Docs: `/docs`

## Test

```bash
pytest
```

## Docker

```bash
docker compose up --build
```
