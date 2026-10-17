from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class UploadedDocument(BaseModel):
    id: str
    filename: str
    file_path: str
    file_size_bytes: int
    document_type: str = "unknown"
    subject: str | None = None
    semester: str | None = None
    academic_year: str | None = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["uploaded", "processing", "indexed", "failed"] = "uploaded"
    indexing_error: str | None = None
    page_count: int | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    content: str
    score: float
    page_number: int | None = None
    subject: str | None = None
    semester: str | None = None
    document_type: str | None = None
    academic_year: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationRecord(BaseModel):
    id: str
    title: str
    messages: list[ChatMessage] = Field(default_factory=list)
    model_name: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelStatus(BaseModel):
    name: str
    size: str | None = None
    modified: str | None = None
    available: bool = True
    error: str | None = None


class AnswerResult(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    follow_up_questions: list[str] = Field(default_factory=list)
    model_name: str
    retrieval_summary: str
    grounded: bool = True
