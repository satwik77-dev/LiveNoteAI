from __future__ import annotations

import os
from abc import ABC, abstractmethod

import httpx


class BaseLLMClient(ABC):
    """Thin wrapper around pre-trained inference backends used without fine-tuning."""

    @abstractmethod
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.timeout_sec = timeout_sec or float(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))
        self.temperature = temperature

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response.")
        return content


class GroqClient(BaseLLMClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.timeout_sec = timeout_sec or float(os.getenv("GROQ_TIMEOUT_SEC", "30"))
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required when LLM_MODE=groq.")

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        content = (((choices[0] if choices else {}).get("message") or {}).get("content")) if choices else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Groq returned an empty response.")
        return content


class OpenAICompatClient(BaseLLMClient):
    """Generic client for any OpenAI-compatible API.

    Works with Deepseek, Kimi (Moonshot), Together AI, OpenRouter, and others.
    Set via env:
        LLM_MODE=openai_compat
        OPENAI_COMPAT_BASE_URL=https://api.deepseek.com/v1
        OPENAI_COMPAT_API_KEY=sk-...
        OPENAI_COMPAT_MODEL=deepseek-chat
        OPENAI_COMPAT_TIMEOUT_SEC=60  (optional)
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = (base_url or os.getenv("OPENAI_COMPAT_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY", "")
        self.model = model or os.getenv("OPENAI_COMPAT_MODEL", "deepseek-chat")
        self.timeout_sec = timeout_sec or float(os.getenv("OPENAI_COMPAT_TIMEOUT_SEC", "60"))
        self.temperature = temperature
        if not self.base_url:
            raise RuntimeError("OPENAI_COMPAT_BASE_URL is required when LLM_MODE=openai_compat.")
        if not self.api_key:
            raise RuntimeError("OPENAI_COMPAT_API_KEY is required when LLM_MODE=openai_compat.")

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        content = (((choices[0] if choices else {}).get("message") or {}).get("content")) if choices else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenAI-compat API ({self.base_url}) returned an empty response.")
        return content


def create_llm_client(mode: str | None = None) -> BaseLLMClient:
    resolved_mode = (mode or os.getenv("LLM_MODE", "ollama")).strip().lower()
    if resolved_mode == "ollama":
        return OllamaClient()
    if resolved_mode == "groq":
        return GroqClient()
    if resolved_mode == "openai_compat":
        return OpenAICompatClient()
    raise ValueError(f"Unsupported LLM_MODE: {resolved_mode}")
