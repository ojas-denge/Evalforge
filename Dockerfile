FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY app ./app
COPY scripts ./scripts
COPY data ./data

# Install CPU-only PyTorch first.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Install EvalForge dependencies.
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
