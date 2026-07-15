"""
Pinecone vector store wrapper.

Handles:
    * Creating / loading a persistent Pinecone collection.
    * Adding chunks (with pre-computed embeddings from our provider-agnostic embedder).
    * Semantic search (query) returning the top-k chunks + metadata.
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.embeddings import get_embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around a persistent Pinecone index."""

    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")
            
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._embedder = get_embedder()

        existing_indexes = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name not in existing_indexes:
            logger.info("Creating Pinecone index '%s' with dimension %d...", self._index_name, self._embedder.dimension)
            self._pc.create_index(
                name=self._index_name,
                dimension=self._embedder.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = self._pc.Index(self._index_name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: list[dict], batch_size: int = 200) -> None:
        """Embed and add a list of chunk dicts to the collection."""
        if not chunks:
            return

        total = len(chunks)
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c["text"] for c in batch]
            ids = [str(uuid.uuid4()) for _ in batch]
            metadatas = [self._sanitize_meta(c["metadata"]) for c in batch]

            # Inject the text into metadata so we can retrieve it later
            for i, meta in enumerate(metadatas):
                meta["text"] = texts[i]

            # "passage" tells asymmetric models (e.g. NVIDIA nemotron) that
            # these are documents to be indexed, not search queries.
            embeddings = self._embedder.embed(texts, input_type="passage")

            vectors = [
                {"id": _id, "values": _emb, "metadata": _meta}
                for _id, _emb, _meta in zip(ids, embeddings, metadatas)
            ]

            self._index.upsert(vectors=vectors)
            logger.info("  Indexed %d/%d chunks to Pinecone", min(start + batch_size, total), total)

    def reset(self) -> None:
        """Delete the collection (used before a full re-index)."""
        try:
            self._pc.delete_index(self._index_name)
        except Exception:  # noqa: BLE001
            pass
        
        self._pc.create_index(
            name=self._index_name,
            dimension=self._embedder.dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        self._index = self._pc.Index(self._index_name)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None, where: dict | None = None) -> list[dict]:
        """Return the top-k most similar chunks for a query."""
        k = top_k or settings.retrieval_top_k
        count = self.count()
        if count == 0:
            return []
        k = min(k, count)

        # "query" tells asymmetric models that this is a search query.
        query_embedding = self._embedder.embed([query], input_type="query")[0]

        # Convert filter dict to explicit Pinecone format.
        # e.g. {"is_archived": False} -> {"is_archived": {"$eq": False}}
        pc_filter = None
        if where:
            pc_filter = {k: {"$eq": v} for k, v in where.items()}

        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            filter=pc_filter,
        )

        docs = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {}).copy()
            # Extract text back from metadata
            text = meta.pop("text", "")
            score = match.get("score", 0.0)
            
            docs.append({
                "text": text,
                "metadata": meta,
                "score": score,
            })

        return docs

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.get("total_vector_count", 0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_meta(meta: dict) -> dict:
        """Pinecone metadata values must be str/int/float/bool/list of str."""
        clean = {}
        for k, v in meta.items():
            if v is None:
                clean[k] = ""          
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean


# Singleton accessor
_store_instance: VectorStore | None = None
_store_lock = threading.Lock()


def get_store() -> VectorStore:
    """Return a cached, thread-safe VectorStore instance."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:  # double-check after acquiring lock
                _store_instance = VectorStore()
    return _store_instance
