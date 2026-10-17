from __future__ import annotations

import re
from pathlib import Path

from config.settings import settings


def safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "document"
    return f"{safe_stem}_{abs(hash(filename + str(Path().stat().st_mtime_ns)))}"


def validate_upload(filename: str, file_size: int) -> None:
    extension = Path(filename).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise ValueError("Unsupported file type. Please upload a PDF document.")

    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise ValueError(f"File exceeds the {settings.max_upload_size_mb} MB upload limit.")


def is_valid_query(question: str) -> bool:
    if question is None:
        return False
    stripped = question.strip()
    return len(stripped) >= 3 and len(stripped.split()) >= 1
