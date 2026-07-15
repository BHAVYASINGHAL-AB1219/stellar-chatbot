"""
FastAPI server for the College Chatbot.

Endpoints:
    GET  /            -> health + basic info
    GET  /health      -> service health + indexed chunk count
    POST /chat        -> one-shot RAG answer (JSON)
    POST /chat/stream -> streaming RAG answer (Server-Sent Events)
    POST /ingest      -> trigger crawl + index (background task, API-key protected)
    GET  /sources     -> retrieve source citations for a question

Run:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.rag import answer, answer_stream, get_sources
from app.vectorstore import get_store

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Rate-limiting middleware (no external dependency)
# ------------------------------------------------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory per-IP rate limiter."""

    def __init__(self, app, max_requests: int = 30, window: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        # Prune expired timestamps
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < self.window
        ]
        if len(self._requests[client_ip]) >= self.max_requests:
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again shortly."},
                status_code=429,
            )
        self._requests[client_ip].append(now)
        return await call_next(request)


# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------
app = FastAPI(
    title="NIT Sikkim College Chatbot API",
    description="RAG-powered chatbot answering questions from the official college website & documents.",
    version="1.0.0",
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production (e.g. ["https://yourdomain.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Rate limiting ----
app.add_middleware(RateLimitMiddleware, max_requests=30, window=60)

# ---- Static files ----
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 20  # cap to prevent context overflow / abuse


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    history: Optional[list[dict]] = Field(
        default=None,
        description='Optional conversation history: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]',
    )
    top_k: Optional[int] = Field(default=None, description="Number of chunks to retrieve")


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    indexed_chunks: int


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/")
def root():
    """Serve the chat UI at the root, or fall back to JSON info."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "service": "NIT Sikkim College Chatbot",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "health": "GET /health",
            "ingest": "POST /ingest",
            "sources": "GET /sources?question=...",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        count = get_store().count()
    except Exception:  # noqa: BLE001
        count = -1
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        indexed_chunks=count,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    logger.info("Received chat request payload: %s", req.model_dump_json())
    # Cap history length to prevent abuse / context overflow
    history = (req.history or [])[-MAX_HISTORY_MESSAGES:]
    result = answer(question=req.question, history=history or None, top_k=req.top_k)
    return ChatResponse(**result)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Streams the answer as Server-Sent Events.
    Format:
        data: {"token": "..."}\n\n
        ...
        data: {"sources": [...]}\n\n
        data: [DONE]\n\n
    """
    logger.info("Received streaming chat request payload: %s", req.model_dump_json())
    # Cap history length
    history = (req.history or [])[-MAX_HISTORY_MESSAGES:]

    def event_stream():
        try:
            # answer_stream now returns (iterator, sources) in one search
            stream, sources = answer_stream(
                req.question, history=history or None, top_k=req.top_k
            )
            for token in stream:
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'sources': sources})}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming error for question: %s", req.question)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sources")
def sources(question: str, top_k: Optional[int] = None) -> dict:
    return {"question": question, "sources": get_sources(question, top_k=top_k)}


@app.post("/ingest")
def ingest(
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None),
) -> dict:
    """
    Trigger a full crawl + re-index in the background.
    Protected by an API key when INGEST_API_KEY is configured.
    Returns immediately with a status message.
    """
    # Require API key if one is configured
    if settings.ingest_api_key:
        if x_api_key != settings.ingest_api_key:
            raise HTTPException(status_code=403, detail="Invalid or missing API key.")

    from app.ingest import run_ingestion

    background_tasks.add_task(run_ingestion)
    return {
        "status": "started",
        "message": "Crawling and indexing started in the background. Check /health for indexed_chunks.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
