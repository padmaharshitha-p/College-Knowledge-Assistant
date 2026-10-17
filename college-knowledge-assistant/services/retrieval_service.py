from __future__ import annotations

from typing import Any

from config.settings import settings
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService


class RetrievalService:
    def __init__(self, vector_store: VectorStoreService, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    @staticmethod
    def _distance_to_similarity(distance: float | None) -> float | None:
        if distance is None:
            return None
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return None

        if value < 0:
            return 0.0
        if settings.retrieval_distance_metric.lower() in {"l2", "euclidean"}:
            return 1.0 / (1.0 + value)
        return max(0.0, min(1.0, 1.0 - value))

    def retrieve(self, query: str, top_k: int = 5, subject_filter: str | None = None) -> list[dict[str, Any]]:
        embedding = self.embedding_service.embed(query)
        filters = {"subject": subject_filter} if subject_filter else None
        results = self.vector_store.query(query_vector=embedding, top_k=top_k, where=filters)
        cleaned: list[dict[str, Any]] = []
        for item in results:
            meta = item.get("metadata", {})
            content = (item.get("content") or "").strip()
            if not content:
                continue

            distance = item.get("distance")
            similarity = self._distance_to_similarity(distance)
            if similarity is not None and similarity < settings.retrieval_similarity_threshold:
                continue

            cleaned.append({
                "content": content,
                "score": item.get("score", 0.0),
                "distance": distance,
                "similarity": similarity,
                "filename": meta.get("filename"),
                "page_number": meta.get("page_number"),
                "document_id": meta.get("document_id"),
                "document_type": meta.get("document_type"),
                "subject": meta.get("subject"),
                "semester": meta.get("semester"),
                "academic_year": meta.get("academic_year"),
            })
        return cleaned
