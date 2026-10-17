from __future__ import annotations

import json
from typing import Any

import requests

from config.settings import settings


class OllamaService:
    def __init__(self, base_url: str, timeout: int):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_available_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            return [item.get("name") for item in models if item.get("name")]
        except requests.RequestException:
            return []

    def validate_model(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("No model was selected.")
        available = self.get_available_models()
        if model_name not in available:
            raise ValueError(f"Model '{model_name}' is not available in Ollama. Run 'ollama list' to inspect installed models and try again.")

    def generate(self, prompt: str, model_name: str, stream: bool = False) -> str:
        self.validate_model(model_name)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        raise RuntimeError("The model returned an unreadable response.")
