import sys
import os

# Ensure the app module can be imported
sys.path.insert(0, os.getcwd())

from app.vectorstore import get_store
from app.rag import _search_and_rerank
from app.config import settings

def debug_query(question: str, top_k: int = 5):
    print(f"==================================================")
    print(f"🔍 DEBUGGING QUERY: '{question}'")
    print(f"==================================================\n")
    
    store = get_store()
    
    # Check total chunks in the database
    count = store.count()
    print(f"📊 Total chunks in Vector Database: {count}\n")
    
    if count == 0:
        print("❌ Database is empty. Have you run the ingestion pipeline? (python -m app.ingest)")
        return
        
    print(f"Fetching top {top_k} results...\n")
    
    # Perform the search using the RAG layer (which handles filtering & reranking)
    results = _search_and_rerank(question, top_k=top_k)
    
    if not results:
        print("❌ No results found.")
        return
        
    for i, res in enumerate(results, 1):
        print(f"--- Result {i} (Similarity Score: {res['score']:.4f}) ---")
        print(f"Source URL:  {res['metadata'].get('source', 'Unknown')}")
        print(f"Page Title:  {res['metadata'].get('title', 'Unknown')}")
        print(f"File Type:   {res['metadata'].get('type', 'Unknown')}")
        print(f"Is Archived: {res['metadata'].get('is_archived', 'Unknown')}")
        print(f"Last Updated:{res['metadata'].get('last_updated', 'Unknown')}")
        if res['metadata'].get('page'):
            print(f"Page Number: {res['metadata']['page']}")
            
        print("\n📝 EXTRACTED TEXT CHUNK:")
        print("-" * 40)
        print(res['text'])
        print("-" * 40)
        print("\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_retrieval.py \"YOUR QUESTION HERE\"")
        print("Example: python debug_retrieval.py \"What is the fee structure for B.Tech?\"")
        sys.exit(1)
        
    user_question = sys.argv[1]
    # You can change the number 5 to retrieve more or fewer chunks
    debug_query(user_question, top_k=5)
