from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    project_name: str = "College Knowledge Assistant"
    app_dir: str = str(BASE_DIR)
    data_dir: str = str(BASE_DIR / "data")
    uploads_dir: str = str(BASE_DIR / "data" / "uploads")
    chroma_dir: str = str(BASE_DIR / "data" / "chroma")
    sqlite_db_path: str = str(BASE_DIR / "data" / "app.db")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_ollama_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 60
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    retrieval_similarity_threshold: float = 0.35
    retrieval_distance_metric: str = "cosine"
    max_upload_size_mb: int = 20
    allowed_extensions: set[str] = {".pdf"}
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_dirs(self) -> None:
        for directory in [self.data_dir, self.uploads_dir, self.chroma_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
