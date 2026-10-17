from __future__ import annotations

import uuid
from typing import Any

import chromadb

from config.settings import settings
from services.embedding_service import EmbeddingService


class VectorStoreService:
    def __init__(self, path: str, embedding_service: EmbeddingService | None = None):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name="college_knowledge")
        self.embedding_service = embedding_service

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            cleaned[key] = value
        return cleaned

    def add_documents(
        self,
        document_id: str,
        filename: str,
        chunks: list[dict[str, Any]],
        metadata: dict[str, Any],
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        ids: list[str] = []
        documents: list[str] = []
        vectors: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks):
            content = chunk["content"]
            chunk_id = chunk.get("chunk_id") or uuid.uuid4().hex
            chunk_meta = {
                "document_id": document_id,
                "filename": filename,
                "chunk_id": chunk_id,
                "page_number": chunk.get("page_number"),
                "document_type": metadata.get("document_type"),
                "subject": metadata.get("subject"),
                "semester": metadata.get("semester"),
                "academic_year": metadata.get("academic_year"),
                "source_index": index,
            }
            documents.append(content)
            metadatas.append(self._clean_metadata(chunk_meta))
            ids.append(chunk_id)

        if embeddings is None:
            if self.embedding_service is None:
                raise ValueError("Embedding service is required to index documents.")
            embeddings = [self.embedding_service.embed(chunk["content"]) for chunk in chunks]

        if documents:
            self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
        return ids

    def query(self, query_vector: list[float], top_k: int = 5, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)
        output: list[dict[str, Any]] = []
        for idx, doc in enumerate(results.get("documents", [[]])[0]):
            meta = results.get("metadatas", [[]])[0][idx]
            distance = float(results.get("distances", [[]])[0][idx]) if results.get("distances") else None
            output.append({
                "content": doc,
                "metadata": meta,
                "distance": distance,
                "score": distance if distance is not None else 0.0,
            })
        return output

    def delete_document(self, document_id: str) -> None:
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            pass
