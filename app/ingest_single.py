"""
Ingest a single URL without resetting the vector store.

Usage:
    python -m app.ingest_single <url>
"""
import argparse
import logging
import sys
import tempfile
from pathlib import Path

import requests

from app.chunker import chunk_documents
from app.loaders import load_file, _enrich_metadata, _reset_nav_dedup
from app.vectorstore import get_store

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Ingest a single URL into the vector store without wiping it.")
    parser.add_argument("url", help="The URL to ingest")
    args = parser.parse_args()

    url = args.url
    print(f"Fetching {url}...")
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        sys.exit(1)
        
    content_type = resp.headers.get("content-type", "").lower()
    
    # Simple extension guessing based on content-type
    suffix = ".html"
    if "pdf" in content_type:
        suffix = ".pdf"
    elif "word" in content_type or "docx" in content_type:
        suffix = ".docx"
    elif "text/plain" in content_type:
        suffix = ".txt"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = Path(tmp.name)
        
    try:
        print("Parsing document...")
        _reset_nav_dedup()
        docs = load_file(tmp_path, source_url=url)
        _enrich_metadata(docs)
        
        if not docs:
            print("No text could be extracted from the document.")
            sys.exit(1)
            
        print(f"Chunking {len(docs)} document sections...")
        chunks = chunk_documents(docs)
        
        if not chunks:
            print("No chunks were created (document might be too short or noisy).")
            sys.exit(1)
            
        print(f"Created {len(chunks)} chunks. Adding to Pinecone...")
        store = get_store()
        # Notice we DO NOT call store.reset() here, so the DB remains intact!
        store.add_chunks(chunks)
        
        print("✅ Successfully ingested the single URL!")
    finally:
        tmp_path.unlink()

if __name__ == "__main__":
    main()
