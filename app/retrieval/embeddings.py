from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()

        self.model_name = settings.embedding_model
        self.model = SentenceTransformer(self.model_name)

        logger.info(
            "Embedding service initialized with model=%s",
            self.model_name,
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        embeddings = self.model.encode(
            [text],
            normalize_embeddings=True,
        )

        return embeddings[0].tolist()
