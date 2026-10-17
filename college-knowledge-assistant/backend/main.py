from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.llm_service import OllamaService
from config.settings import settings

app = FastAPI(title="College Knowledge Assistant API")
llm_service = OllamaService(base_url=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds)


class ChatRequest(BaseModel):
    question: str
    model_name: str = settings.default_ollama_model


class ChatResponse(BaseModel):
    answer: str
    model_name: str
    sources: list[str] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def list_models() -> list[str]:
    return llm_service.get_available_models()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = llm_service.generate(request.question, model_name=request.model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatResponse(answer=answer, model_name=request.model_name)
