"""
Provider-agnostic embedding layer.

A single `Embedder` interface with concrete implementations for:
    * sentence-transformers (local, free, no API key)
    * openai
    * gemini
    * ollama
    * nvidia (NVIDIA NIM / build.nvidia.com — OpenAI-compatible, asymmetric models)

Use `get_embedder()` to get the configured provider.

Note on `input_type`:
    Some providers (e.g. NVIDIA's asymmetric retrieval models such as
    `nvidia/llama-nemotron-embed-1b-v2`) require an `input_type` field:
        - "passage"  -> for indexing documents into the store
        - "query"    -> for searching with a user question
    The `embed()` method accepts an optional `input_type` argument so the
    vector store can pass the correct value at index time vs. query time.
    Providers that don't use it simply ignore the argument.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Literal

from app.config import settings

import logging

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract embedder: text -> vector."""

    @abstractmethod
    def embed(
        self, texts: list[str], input_type: str | None = None
    ) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text.

        Args:
            texts: The strings to embed.
            input_type: Optional hint for asymmetric models. Use "passage"
                when indexing documents and "query" when searching. Providers
                that don't support it ignore this argument.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""


# ------------------------------------------------------------------
# Sentence Transformers (local)
# ------------------------------------------------------------------
class SentenceTransformerEmbedder(Embedder):
    def __init__(self) -> None:
        import os
        from sentence_transformers import SentenceTransformer

        # Set HuggingFace token for downloading gated/private models.
        if settings.hf_token:
            os.environ["HF_TOKEN"] = settings.hf_token

        self._model = SentenceTransformer(
            settings.sentence_transformer_model,
            token=settings.hf_token or None,
        )
        self._dim: int | None = None

    def embed(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        if self._dim is None:
            self._dim = int(vectors.shape[1])
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Embed a dummy text to discover the dimension.
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim


# ------------------------------------------------------------------
# OpenAI
# ------------------------------------------------------------------
class OpenAIEmbedder(Embedder):
    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = settings.openai_embedding_model
        self._dim: int | None = None

    def embed(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        # OpenAI accepts up to 2048 inputs per request.
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), 2048):
            batch = texts[i : i + 2048]
            resp = self._client.embeddings.create(input=batch, model=self._model)
            all_vectors.extend([d.embedding for d in resp.data])
        if self._dim is None and all_vectors:
            self._dim = len(all_vectors[0])
        return all_vectors

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim


# ------------------------------------------------------------------
# Gemini
# ------------------------------------------------------------------
class GeminiEmbedder(Embedder):
    def __init__(self) -> None:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embedding_model
        self._genai = genai
        self._dim: int | None = None

    def embed(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        # Batch up to 100 texts per API call instead of one-at-a-time.
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            result = self._genai.embed_content(
                model=f"models/{self._model}",
                content=batch,
                task_type="retrieval_document",
            )
            # When content is a list, result["embedding"] is a list of lists.
            all_vectors.extend(result["embedding"])
        if self._dim is None and all_vectors:
            self._dim = len(all_vectors[0])
        return all_vectors

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim


# ------------------------------------------------------------------
# Ollama (local)
# ------------------------------------------------------------------
class OllamaEmbedder(Embedder):
    def __init__(self) -> None:
        import requests

        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_embedding_model
        self._requests = requests
        self._dim: int | None = None

    def embed(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        # Use the batch /api/embed endpoint instead of one-at-a-time.
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            resp = self._requests.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": batch},
                timeout=120,
            )
            resp.raise_for_status()
            all_vectors.extend(resp.json()["embeddings"])
        if self._dim is None and all_vectors:
            self._dim = len(all_vectors[0])
        return all_vectors

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim


# ------------------------------------------------------------------
# NVIDIA NIM (build.nvidia.com — OpenAI-compatible, asymmetric models)
# ------------------------------------------------------------------
class NvidiaEmbedder(Embedder):
    """
    Calls NVIDIA's hosted NIM endpoint (integrate.api.nvidia.com), which is
    OpenAI-compatible. Many NVIDIA retrieval models (e.g.
    `nvidia/llama-nemotron-embed-1b-v2`) are *asymmetric* and require an
    `input_type` field: "passage" for documents, "query" for search.
    """

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )
        self._model = settings.nvidia_embedding_model
        self._dim: int | None = None

    def embed(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        # NVIDIA asymmetric models require input_type; default to "passage"
        # (indexing) when not specified, which is the safe default for bulk
        # embedding. The vector store passes "query" at search time.
        itype = input_type or "passage"
        all_vectors: list[list[float]] = []
        # Keep batches small to stay well under NVIDIA's token/size limits.
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            resp = self._client.embeddings.create(
                input=batch,
                model=self._model,
                extra_body={"input_type": itype, "encoding_format": "float"},
            )
            all_vectors.extend([d.embedding for d in resp.data])
        if self._dim is None and all_vectors:
            self._dim = len(all_vectors[0])
        return all_vectors

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dimension probe"], input_type="query")[0])
        return self._dim


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
_EMBEDDERS: dict[str, type[Embedder]] = {
    "sentence-transformers": SentenceTransformerEmbedder,
    "openai": OpenAIEmbedder,
    "gemini": GeminiEmbedder,
    "ollama": OllamaEmbedder,
    "nvidia": NvidiaEmbedder,
}

_embedder_instance: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Return a cached, thread-safe embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:  # double-check after acquiring lock
                provider = settings.embedding_provider
                cls = _EMBEDDERS.get(provider)
                if cls is None:
                    raise ValueError(f"Unknown embedding provider: {provider}")
                logger.info("Loading embedder: %s", provider)
                _embedder_instance = cls()
    return _embedder_instance
