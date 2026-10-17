from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

import fitz

from config.settings import settings
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from utils.logging_config import logger
from utils.validators import safe_filename, validate_upload


class DocumentService:
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path = self.upload_dir.parent / "documents_index.json"
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService(path=settings.chroma_dir, embedding_service=self.embedding_service)

    def _read_manifest(self) -> dict[str, Any]:
        if not self.documents_path.exists():
            return {}
        try:
            with self.documents_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_manifest(self, data: dict[str, Any]) -> None:
        with self.documents_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def _hash_file(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def index_pdf_upload(self, uploaded_file: Any) -> dict[str, Any]:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        validate_upload(filename, len(file_bytes))

        manifest = self._read_manifest()
        if any(doc.get("filename") == filename for doc in manifest.values()):
            raise ValueError("This document has already been indexed for the current workspace.")

        file_hash = self._hash_file(file_bytes)
        safe_name = safe_filename(filename)
        file_path = self.upload_dir / f"{safe_name}.pdf"
        file_path.write_bytes(file_bytes)

        doc_id = uuid.uuid4().hex
        metadata = {
            "document_id": doc_id,
            "filename": filename,
            "file_path": str(file_path),
            "file_hash": file_hash,
            "document_type": "syllabus",
            "subject": None,
            "semester": None,
            "academic_year": None,
            "status": "processing",
        }

        try:
            page_text_map = self.extract_pdf_text(file_path)
            document_text = "\n".join(page_text_map.values())
            if not document_text.strip():
                raise ValueError("The uploaded PDF contains no extractable text. Scanned or image-only PDFs may require OCR.")

            chunks = self.chunk_text(document_text)
            if not chunks:
                raise ValueError("No usable text was produced from the uploaded PDF.")
            vector_ids = self.vector_store.add_documents(
                document_id=doc_id,
                filename=filename,
                chunks=chunks,
                metadata={
                    "document_id": doc_id,
                    "filename": filename,
                    "document_type": metadata["document_type"],
                    "subject": None,
                    "semester": None,
                    "academic_year": None,
                },
            )

            metadata["status"] = "indexed"
            metadata["document_id"] = doc_id
            manifest[doc_id] = metadata
            self._write_manifest(manifest)
            logger.info("Indexed document %s with %s chunks", filename, len(chunks))
            return {"document_id": doc_id, "chunks": len(chunks), "vector_ids": vector_ids}
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = str(exc)
            manifest[doc_id] = metadata
            self._write_manifest(manifest)
            logger.exception("Document indexing failed for %s", filename)
            raise

    def extract_pdf_text(self, file_path: Path) -> dict[int, str]:
        try:
            document = fitz.open(file_path)
        except Exception as exc:  # pragma: no cover - runtime guard
            raise ValueError(f"Unable to read PDF: {exc}") from exc

        text_map: dict[int, str] = {}
        for page_index in range(document.page_count):
            page = document[page_index]
            text = page.get_text("text")
            if text and text.strip():
                text_map[page_index + 1] = text.strip()
        document.close()
        if not text_map:
            raise ValueError("The PDF has no extractable text. Scanned or image-only PDFs may need OCR support.")
        return text_map

    def chunk_text(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[dict[str, Any]]:
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap if overlap is not None else settings.chunk_overlap

        if len(text) <= chunk_size:
            return [{"content": text.strip(), "page_number": None, "chunk_id": uuid.uuid4().hex}]

        chunks: list[dict[str, Any]] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "content": chunk,
                    "page_number": None,
                    "chunk_id": uuid.uuid4().hex,
                })
            start = end - overlap if end < len(text) else len(text)
        return chunks

    def list_documents(self) -> list[dict[str, Any]]:
        manifest = self._read_manifest()
        documents = []
        for doc_id, doc in manifest.items():
            item = {"id": doc_id, **doc}
            documents.append(item)
        return sorted(documents, key=lambda item: item.get("filename", ""))

    def delete_document(self, document_id: str) -> None:
        manifest = self._read_manifest()
        doc = manifest.pop(document_id, None)
        if not doc:
            return
        file_path = Path(doc.get("file_path", ""))
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        self.vector_store.delete_document(document_id)
        self._write_manifest(manifest)

    def all_documents(self) -> dict[str, Any]:
        return self._read_manifest()
