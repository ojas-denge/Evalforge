from contextlib import contextmanager
from typing import Any, Iterator

from app.core.config import get_settings


class Tracer:
    """Optional observability layer for EvalForge."""

    def __init__(self) -> None:
        settings = get_settings()

        self.enabled = settings.langfuse_enabled
        self.client = None

        if self.enabled:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
            )

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return

        with self.client.start_as_current_observation(
            name=name,
            as_type="chain",
            input=input,
            metadata=metadata,
        ) as observation:
            yield observation

    @contextmanager
    def retrieval(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return

        with self.client.start_as_current_observation(
            name=name,
            as_type="retriever",
            input=input,
            metadata=metadata,
        ) as observation:
            yield observation

    def flush(self) -> None:
        if self.client is not None:
            self.client.flush()