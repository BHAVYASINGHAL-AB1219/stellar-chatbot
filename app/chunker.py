"""
Text chunker — recursive character splitting.

This is the same strategy used by LangChain's RecursiveCharacterTextSplitter.
It tries to split text using a hierarchy of separators, from most natural
(paragraphs) to least (single characters), so that sentences and paragraphs
stay intact whenever possible.

Separator hierarchy (tried in order):
    1. "\\n\\n"  (paragraph breaks)
    2. "\\n"    (line breaks)
    3. ". "     (sentence ends)
    4. " "      (word boundaries)
    5. ""        (hard character cut — last resort)

After splitting, chunks are greedily merged up to `chunk_size` characters
with `chunk_overlap` characters of overlap between consecutive chunks.
"""
from __future__ import annotations

from app.config import settings

# Ordered separators — most natural first.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, separator: str) -> list[str]:
    """Split text on a separator. Empty separator => list of characters."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # Re-attach the separator to preserve it (except at the very end).
    return [p + separator for p in parts[:-1]] + [parts[-1]] if parts else []


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    Recursively split `text` into pieces no larger than `chunk_size`.
    Try the first separator; if any resulting piece is still too big,
    recurse on that piece with the next separator.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for i, sep in enumerate(separators):
        splits = _split_text(text, sep)
        # If this separator produced more than one piece, try it.
        if len(splits) > 1:
            result: list[str] = []
            for piece in splits:
                if len(piece) <= chunk_size:
                    if piece.strip():
                        result.append(piece)
                else:
                    # Piece still too big — recurse with remaining separators.
                    result.extend(_recursive_split(piece, separators[i + 1:], chunk_size))
            return result

    # All separators exhausted and text still > chunk_size — hard cut.
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _merge_splits(splits: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily merge small splits into chunks of ~chunk_size with overlap."""
    chunks: list[str] = []
    current = ""

    for split in splits:
        if not split.strip():
            continue
        # Would adding this split exceed the limit?
        if current and len(current) + len(split) > chunk_size:
            chunks.append(current.strip())
            # Start next chunk with overlap from the tail of the current chunk.
            if overlap > 0:
                tail = current[-overlap:]
                current = tail + split
            else:
                current = split
        else:
            current += split

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_text(text: str, metadata: dict) -> list[dict]:
    """Split a single document into overlapping chunks with metadata."""
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    if not text or not text.strip():
        return []

    splits = _recursive_split(text, SEPARATORS, chunk_size)
    merged = _merge_splits(splits, chunk_size, chunk_overlap)

    # Minimum chunk quality threshold — skip tiny chunks that are just
    # headers, page numbers, or other noise that slipped through.
    MIN_CHUNK_LENGTH = 50

    chunks: list[dict] = []
    for index, chunk in enumerate(merged):
        stripped = chunk.strip()
        if not stripped or len(stripped) < MIN_CHUNK_LENGTH:
            continue
        meta = dict(metadata)
        meta["chunk_index"] = index
        chunks.append({"text": stripped, "metadata": meta})

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk a list of loaded documents into a flat list of chunks."""
    all_chunks: list[dict] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc["text"], doc["metadata"]))
    return all_chunks
