"""
Provider-agnostic LLM layer.

A single `LLM` interface with concrete implementations for:
    * openai
    * gemini
    * ollama
    * nvidia (NVIDIA NIM / build.nvidia.com — OpenAI-compatible)

Each implementation supports:
    * generate(messages) -> str            (one-shot)
    * stream(messages) -> Iterator[str]     (token streaming)

Use `get_llm()` to get the configured provider.
"""
from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from typing import Iterator, Literal

from app.config import settings

logger = logging.getLogger(__name__)


class LLM(ABC):
    """Abstract chat LLM."""

    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        """messages: [{"role": "system"|"user"|"assistant", "content": "..."}]"""

    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield response tokens/chunks as they arrive."""


# ------------------------------------------------------------------
# OpenAI
# ------------------------------------------------------------------
class OpenAILLM(LLM):
    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = settings.openai_chat_model

    def generate(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.3
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.3, stream=True
        )
        for chunk in stream:
            # Some providers send chunks with empty choices (e.g. final chunk).
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ------------------------------------------------------------------
# Gemini
# ------------------------------------------------------------------
class GeminiLLM(LLM):
    def __init__(self) -> None:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_chat_model)
        self._genai = genai

    def _to_gemini_history(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Gemini (system, history) format."""
        system = ""
        history: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})
        return system.strip(), history

    def generate(self, messages: list[dict]) -> str:
        import copy

        system, history = self._to_gemini_history(messages)
        history = copy.deepcopy(history)  # avoid mutating caller's data
        chat = self._model.start_chat(history=history)
        # Gemini doesn't have a native system role in chat; prepend it.
        if system and history:
            history[0]["parts"][0] = f"{system}\n\n{history[0]['parts'][0]}"
            chat = self._model.start_chat(history=history)
        elif system:
            chat = self._model.start_chat(history=[{"role": "user", "parts": [system]}])
        resp = chat.send_message(history[-1]["parts"][0] if history else system)
        return resp.text

    def stream(self, messages: list[dict]) -> Iterator[str]:
        import copy

        system, history = self._to_gemini_history(messages)
        history = copy.deepcopy(history)  # avoid mutating caller's data
        if system and history:
            history[0]["parts"][0] = f"{system}\n\n{history[0]['parts'][0]}"
        chat = self._model.start_chat(history=history[:-1] if len(history) > 1 else [])
        last = history[-1]["parts"][0] if history else system
        resp = chat.send_message(last, stream=True)
        for chunk in resp:
            yield chunk.text


# ------------------------------------------------------------------
# Ollama (local)
# ------------------------------------------------------------------
class OllamaLLM(LLM):
    def __init__(self) -> None:
        import requests

        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_chat_model
        self._requests = requests

    def generate(self, messages: list[dict]) -> str:
        resp = self._requests.post(
            f"{self._base_url}/api/chat",
            json={"model": self._model, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def stream(self, messages: list[dict]) -> Iterator[str]:
        with self._requests.post(
            f"{self._base_url}/api/chat",
            json={"model": self._model, "messages": messages, "stream": True},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break


# ------------------------------------------------------------------
# NVIDIA NIM (build.nvidia.com — OpenAI-compatible chat endpoint)
# ------------------------------------------------------------------
class NvidiaLLM(LLM):
    """
    Calls NVIDIA's hosted NIM endpoint (integrate.api.nvidia.com), which is
    OpenAI-compatible. This lets us use chat models hosted on NVIDIA's
    platform (e.g. `z-ai/glm-5.2`) with the same OpenAI client.
    """

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )
        self._model = settings.nvidia_chat_model

    def generate(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.3
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.3, stream=True
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
_LLMs: dict[str, type[LLM]] = {
    "openai": OpenAILLM,
    "gemini": GeminiLLM,
    "ollama": OllamaLLM,
    "nvidia": NvidiaLLM,
}

_llm_instance: LLM | None = None
_llm_lock = threading.Lock()


def get_llm() -> LLM:
    """Return a cached, thread-safe LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            if _llm_instance is None:  # double-check after acquiring lock
                provider = settings.llm_provider
                cls = _LLMs.get(provider)
                if cls is None:
                    raise ValueError(f"Unknown LLM provider: {provider}")
                logger.info("Loading LLM: %s", provider)
                _llm_instance = cls()
    return _llm_instance
