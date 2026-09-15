from collections.abc import Sequence
from typing import Protocol

from sentence_transformers import CrossEncoder

from app.core.logging import get_logger

logger = get_logger(__name__)


class Reranker(Protocol):
    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

        logger.info("Reranker initialized with model=%s", model_name)

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        if not documents:
            return []

        scores = self.model.predict(
            [(query, document) for document in documents]
        )

        return [float(score) for score in scores]
