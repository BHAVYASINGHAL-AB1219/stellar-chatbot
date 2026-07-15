"""
Ingestion pipeline: crawl -> load -> chunk -> embed -> index.

This is the one command that builds the entire knowledge base.

Usage:
    python -m app.ingest            # full crawl + index
    python -m app.ingest --no-crawl # index from existing crawled data only
"""
from __future__ import annotations

import argparse
import logging
import sys

from app.chunker import chunk_documents
from app.config import settings
from app.loaders import load_all_from_manifest
from app.vectorstore import get_store

logger = logging.getLogger(__name__)


def run_ingestion(crawl: bool = True) -> None:
    print("=" * 60)
    print("  NIT Sikkim College Chatbot — Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Crawl (optional)
    if crawl:
        print("\n[1/4] Crawling website & downloading documents...")
        from app.crawler import Crawler

        Crawler().crawl()
    else:
        print("\n[1/4] Skipping crawl (using existing data).")

    # Step 2: Load documents
    print("\n[2/4] Loading documents from disk...")
    documents = load_all_from_manifest()
    print(f"   Loaded {len(documents)} document sections.")
    if not documents:
        print("   ⚠ No documents found. Run with crawl enabled: python -m app.ingest")
        sys.exit(1)

    # Show breakdown by type
    from collections import Counter
    type_counts = Counter(d["metadata"]["type"] for d in documents)
    nav_count = sum(1 for d in documents if "(Navigation)" in d["metadata"].get("title", ""))
    table_count = sum(1 for d in documents if d["metadata"].get("title", "").endswith(")") and "Table" in d["metadata"].get("title", ""))
    for doc_type, count in type_counts.most_common():
        chars = sum(len(d["text"]) for d in documents if d["metadata"]["type"] == doc_type)
        print(f"     {doc_type}: {count} docs ({chars:,} chars)")
    print(f"     Navigation docs (deduplicated): {nav_count}")
    print(f"     Extracted tables: {table_count}")

    # Step 3: Chunk
    print("\n[3/4] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunks "
          f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}).")

    # Step 4: Embed + index
    print("\n[4/4] Embedding & indexing into vector store...")
    store = get_store()
    store.reset()
    store.add_chunks(chunks)

    print("\n" + "=" * 60)
    print(f"  ✅ Ingestion complete! Indexed {store.count()} chunks.")
    print(f"  📊 Retrieval top_k: {settings.retrieval_top_k}")
    print("  Start the API:  uvicorn app.main:app --reload --port 8000")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest college website into the vector store.")
    parser.add_argument(
        "--no-crawl",
        action="store_true",
        help="Skip crawling and index from existing data/raw files only.",
    )
    args = parser.parse_args()
    run_ingestion(crawl=not args.no_crawl)


if __name__ == "__main__":
    main()
