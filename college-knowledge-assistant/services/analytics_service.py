from __future__ import annotations

import json
from pathlib import Path


class AnalyticsService:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def read_summary(self) -> dict:
        try:
            if not self.db_path.exists():
                return {}
            with self.db_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def save_summary(self, payload: dict) -> None:
        with self.db_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
