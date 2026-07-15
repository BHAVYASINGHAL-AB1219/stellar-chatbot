"""
RAG (Retrieval-Augmented Generation) pipeline.

Flow:
    user question
        -> semantic search in vector store (top-k chunks)
        -> build a grounded prompt with the retrieved context
        -> LLM generates an answer
        -> return answer + source citations

Supports both one-shot and streaming responses.
"""
from __future__ import annotations

import logging
from typing import Iterator
import math
from datetime import datetime, timezone

from app.config import settings
from app.llm import get_llm
from app.vectorstore import get_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the official assistant for National Institute of Technology (NIT) Sikkim.
You are a friendly, conversational chatbot that also has access to a knowledge base
of information extracted from the official college website and its documents.

Your goal is to be the primary place people come for information about NIT Sikkim,
so they don't have to visit the website themselves. Answer confidently and helpfully.

How you should behave:
1. Be a normal, friendly chatbot. Greet users warmly, make small talk, and answer
   general conversational messages (e.g. "hi", "hello", "how are you", "thanks")
   naturally and politely.
2. When the user asks about the college (admissions, courses, fees, faculty,
   tenders, campus, hostels, exams, notices, etc.), use the provided context to
   answer accurately and completely. Cite sources by referencing the number in
   brackets, e.g. [1], [2], and list the sources at the end of your answer.
3. **SYNTHESIZE across all context chunks.** The context may contain multiple
   pieces of relevant information from different pages and documents. Combine
   them into a single, comprehensive answer instead of only quoting one chunk.
   For example, if one chunk has department names and another has department
   details, merge them together.
4. **Handle tables and structured data carefully.** The context may include data
   formatted as pipe-separated tables (e.g. "Header1 | Header2 | Header3").
   Present this data clearly using bullet points, numbered lists, or markdown
   tables in your answer.
5. If the context contains PARTIAL information that is relevant to the question,
   share EVERYTHING you know from it — do not tell the user to go look elsewhere.
   State clearly what information you do have, and if there are gaps, briefly
   note what is missing.
6. If the context contains NOTHING relevant to a college-specific question, say
   so honestly and briefly (e.g. "I don't have that information right now."). Do
   NOT repeatedly tell the user to visit the website, and do NOT mention the
   website URL unless the user explicitly asks for it. Do not invent facts,
   dates, names, or policies.
7. Be concise, accurate, and helpful. Use bullet points where appropriate.
8. When listing items (departments, courses, fees, etc.), be thorough — list
   ALL items found in the context, not just a few examples.
9. If no context is provided at all, simply converse naturally — you do not need
   context for greetings or general conversation.
"""


HISTORICAL_KEYWORDS = [
    "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024",
    "history", "used to", "earlier", "previous", "old", "past"
]

def is_historical_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in HISTORICAL_KEYWORDS)


def recency_score(last_updated: str, half_life_days: int = 365) -> float:
    try:
        doc_date = datetime.fromisoformat(last_updated)
        # Ensure timezone-aware comparison.
        if doc_date.tzinfo is None:
            doc_date = doc_date.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - doc_date).days
        if age_days < 0:
            age_days = 0
        return math.exp(-age_days * math.log(2) / half_life_days)
    except Exception:
        return 0.5


SOURCE_AUTHORITY = {
    "webpage": 1.0,
    "html": 1.0,
    "php": 1.0,
    "pdf": 0.6,
    "docx": 0.6,
    "doc": 0.6,
}

def rerank_by_recency_authority(results: list[dict], alpha: float = 0.6) -> list[dict]:
    for res in results:
        meta = res["metadata"]
        sim_score = res.get("score", 0.0)
        
        rec = recency_score(meta.get("last_updated", "2026-01-01"))
        auth = SOURCE_AUTHORITY.get(meta.get("source_type", "unknown"), 0.5)
        
        # Blend cosine similarity with recency and authority
        res["score"] = alpha * sim_score + (1 - alpha) * (rec * auth)
        
    return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)


def _search_and_rerank(question: str, top_k: int | None = None) -> list[dict]:
    is_hist = is_historical_query(question)
    where = None if is_hist else {"is_archived": False}
    
    store = get_store()
    k = top_k or settings.retrieval_top_k
    fetch_k = k * 2
    
    results = store.search(question, top_k=fetch_k, where=where)
    
    if results:
        results = rerank_by_recency_authority(results)
        results = results[:k]
        
    return results


def _build_context(results: list[dict]) -> tuple[str, list[dict]]:
    """Turn search results into a numbered context string + source list."""
    context_parts: list[str] = []
    sources: list[dict] = []
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        source_label = meta.get("source", "unknown")
        title = meta.get("title", "")
        page = meta.get("page")
        page_str = f", page {page}" if page else ""
        context_parts.append(f"[{i}] (Source: {title}{page_str} — {source_label})\n{r['text']}")
        sources.append({
            "id": i,
            "title": title,
            "url": source_label,
            "page": page,
            "score": round(r.get("score", 0.0), 4),
        })
    return "\n\n---\n\n".join(context_parts), sources


def _build_messages(
    question: str,
    context: str | None,
    history: list[dict] | None = None,
) -> list[dict]:
    """Build the LLM message list.

    When `context` is provided, the user prompt includes the retrieved context
    so the LLM can ground its answer (RAG mode). When `context` is None, the
    prompt is just the user's message — letting the bot converse naturally
    (pure chat mode for greetings / small talk).
    """
    if context:
        user_prompt = (
            f"Context from the college website and documents:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {question}\n\n"
            f"Answer the question using the context above when it is relevant. "
            f"Cite sources as [1], [2], etc."
        )
    else:
        user_prompt = question

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def answer(
    question: str,
    history: list[dict] | None = None,
    top_k: int | None = None,
) -> dict:
    """
    One-shot answer.

    Hybrid behaviour:
        * If relevant context is retrieved, the answer is grounded in it (RAG).
        * If nothing is retrieved (or the store is empty), the bot still
          converses naturally — so greetings and small talk work fine.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "question": str,
        }
    """
    results = _search_and_rerank(question, top_k=top_k)

    context, sources = (
        _build_context(results) if results else (None, [])
    )
    messages = _build_messages(question, context, history)
    llm = get_llm()
    answer_text = llm.generate(messages)

    return {"answer": answer_text, "sources": sources, "question": question}


def answer_stream(
    question: str,
    history: list[dict] | None = None,
    top_k: int | None = None,
) -> tuple[Iterator[str], list[dict]]:
    """
    Streaming answer. Returns a (token_iterator, sources) tuple.

    The sources are computed up-front from the same search used to build
    the context, so the caller doesn't need to run a second search.

    Hybrid behaviour: grounded in retrieved context when available, otherwise
    the bot converses naturally (no hard fallback for greetings/small talk).
    """
    results = _search_and_rerank(question, top_k=top_k)

    context, sources = _build_context(results) if results else (None, [])
    messages = _build_messages(question, context, history)
    llm = get_llm()
    return llm.stream(messages), sources


def get_sources(question: str, top_k: int | None = None) -> list[dict]:
    """Return only the source citations for a question (standalone use)."""
    results = _search_and_rerank(question, top_k=top_k)
    _, sources = _build_context(results)
    return sources
