from __future__ import annotations

from typing import Any

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - runtime dependency guard
    SentenceTransformer = None

from config.settings import settings


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

    def embed(self, text: str) -> list[float]:
        if self.model is None:
            raise RuntimeError("Embedding model could not be initialized. Check the installed Python packages and model availability.")
        vector = self.model.encode(text)
        return [float(value) for value in vector]
