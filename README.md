# 🎓 NIT Sikkim College Chatbot

A **RAG (Retrieval-Augmented Generation) chatbot** that answers questions about **National Institute of Technology (NIT) Sikkim** using content crawled from the official website (`nitsikkim.ac.in`) — including all pages, PDFs, and documents.

It is **provider-agnostic**: swap between **OpenAI**, **Google Gemini**, or a **local Ollama** model by changing one config value. Exposes a clean **FastAPI** REST API (with streaming) so you can plug in any frontend later.

---

## 🏗️ Architecture

```
                         ┌──────────────────────────────────┐
                         │           USER QUESTION           │
                         └───────────────┬──────────────────┘
                                         │
   ┌─────────────┐    embed    ┌──────────▼──────────┐  top-k chunks
   │  Embeddings  │◄──────────│   Vector Store      │◄─────────────┐
   │ (ST/OpenAI/  │           │   (ChromaDB)        │              │
   │  Gemini/Oll) │           └──────────┬──────────┘              │
   └──────┬───────┘                      │ retrieve                 │
          │                             │                          │
          ▲                             ▼                          │
          │                    ┌────────────────┐                  │
          │  embed chunks      │   RAG Pipeline  │  grounded prompt │
          │                    │  (app/rag.py)   │─────────────────┘
          │                    └────────┬───────┘
          │                             │ context + question
          │                             ▼
   ┌──────┴───────┐  generate   ┌────────────────┐
   │     LLM      │◄────────────│   FastAPI      │──► /chat, /chat/stream
   │(OpenAI/Gem/  │             │   Server       │──► /health, /ingest
   │  Ollama)     │────────────►│  (app/main.py) │
   └──────────────┘   answer    └────────────────┘
```

**Data flow (ingestion):**
1. **Crawler** (`app/crawler.py`) → BFS-crawls every internal page, downloads all PDF/DOCX/TXT files.
2. **Loaders** (`app/loaders.py`) → Extracts clean text from HTML, PDF, DOCX, TXT.
3. **Chunker** (`app/chunker.py`) → Recursive character splitting — keeps sentences & paragraphs intact, ~1000-char chunks with 200-char overlap.
4. **Embeddings** (`app/embeddings.py`) → Converts chunks → vectors (local or API).
5. **Vector Store** (`app/vectorstore.py`) → Stores vectors in ChromaDB for semantic search.

**Data flow (query):**
1. User question → embedded → semantic search returns top-k chunks.
2. Chunks + question → grounded prompt → LLM generates answer with **citations**.

---

## 📁 Project Structure

```
stellar-chatbot/
├── app/
│   ├── __init__.py        # Package marker
│   ├── config.py         # All settings (reads .env)
│   ├── crawler.py        # Website crawler + document downloader
│   ├── loaders.py        # HTML / PDF / DOCX / TXT text extractors
│   ├── chunker.py       # Text splitter (overlapping chunks)
│   ├── embeddings.py    # Provider-agnostic embeddings
│   ├── vectorstore.py   # ChromaDB wrapper
│   ├── llm.py           # Provider-agnostic LLM (OpenAI/Gemini/Ollama)
│   ├── rag.py           # RAG pipeline (retrieve + generate + cite)
│   ├── ingest.py        # One-command: crawl → load → chunk → index
│   └── main.py          # FastAPI server
├── data/                 # Generated: raw files + vectorstore (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Then edit `.env`. **The default works out-of-the-box with zero API keys** because it uses:
- `EMBEDDING_PROVIDER=sentence-transformers` (local, free)
- `LLM_PROVIDER=openai` (needs a key — see below to switch to free/local)

#### Option A — OpenAI (easiest, cloud)
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...your key...
```

#### Option B — Google Gemini (generous free tier)
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=...your key...
```
Get a key at https://aistudio.google.com/apikey

#### Option C — 100% Free & Private (Ollama, local)
1. Install Ollama: https://ollama.com
2. Pull a model:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```
3. Set in `.env`:
```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_CHAT_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### 3. Build the knowledge base (ingest)

This crawls `nitsikkim.ac.in`, downloads all documents, extracts text, chunks it, embeds it, and stores it in ChromaDB:

```bash
python -m app.ingest
```

> ⏱️ First run downloads the embedding model (~80 MB) and crawls the site. Expect a few minutes depending on site size and `CRAWL_MAX_PAGES`.

To re-index from already-crawled data without re-crawling:
```bash
python -m app.ingest --no-crawl
```

### 4. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive API docs are available at **http://localhost:8000/docs**

---

## 📡 API Endpoints

### `POST /chat` — One-shot answer
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the admission process for B.Tech?"}'
```
**Response:**
```json
{
  "question": "What is the admission process for B.Tech?",
  "answer": "Admission to B.Tech is through JEE Main... [1] [2]",
  "sources": [
    {"id": 1, "title": "Admissions", "url": "https://www.nitsikkim.ac.in/admissions", "page": null, "score": 0.82},
    {"id": 2, "title": "Information Brochure", "url": "https://www.nitsikkim.ac.in/brochure.pdf", "page": 5, "score": 0.79}
  ]
}
```

### `POST /chat/stream` — Streaming answer (Server-Sent Events)
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What courses are offered?"}'
```
Streams `data: {"token": "..."}` lines, then `data: {"sources": [...]}`, then `data: [DONE]`.

### `GET /health` — Service status
```bash
curl http://localhost:8000/health
```

### `POST /ingest` — Trigger background re-crawl + re-index
```bash
curl -X POST http://localhost:8000/ingest
```

### `GET /sources?question=...` — Just the citations
```bash
curl "http://localhost:8000/sources?question=hostel%20fees"
```

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` / `gemini` / `ollama` |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | `sentence-transformers` / `openai` / `gemini` / `ollama` |
| `SITE_URL` | `https://www.nitsikkim.ac.in` | Root URL to crawl |
| `CRAWL_MAX_PAGES` | `500` | Max pages to visit |
| `CRAWL_DELAY` | `1.0` | Seconds between requests (politeness) |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |

---

## 🧪 Example Questions

- "What is the fee structure for B.Tech?"
- "How do I apply for a hostel?"
- "What departments are available?"
- "Tell me about the placement statistics."
- "What is the location of the campus?"
- "How to reach NIT Sikkim?"

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `No documents found` | Run `python -m app.ingest` first |
| Crawler finds few pages | Increase `CRAWL_MAX_PAGES` in `.env` |
| OpenAI auth error | Check `OPENAI_API_KEY` in `.env` |
| Ollama connection refused | Run `ollama serve` and check `OLLAMA_BASE_URL` |
| Slow first embedding | Sentence-Transformers downloads the model once (~80 MB) |
| PDF text is garbled | Some scanned PDFs need OCR (not included); text-based PDFs work fine |

---

## 🛣️ Next Steps / Extensions

- **Frontend**: Build a React/Next.js chat UI on top of `/chat/stream`.
- **OCR**: Add `pytesseract` for scanned PDFs.
- **Re-ranking**: Add a cross-encoder re-ranker for better retrieval accuracy.
- **Auth**: Add API key middleware for production.
- **Scheduled re-ingest**: Cron job calling `POST /ingest` weekly to stay current.
- **Analytics**: Log questions to improve the knowledge base over time.

---

## 🐳 Deployment to a Server (Docker)

The chatbot uses **embedded ChromaDB** — the vector store runs in-process
with your FastAPI app and stores files in a Docker volume. No separate
database server or cloud account is needed. This is the simplest, cheapest,
and fastest option for a college-scale knowledge base (<100k chunks).

### One-time setup

```bash
# 1. Create your config
cp .env.example .env
#    → edit .env with your LLM provider + API key (or use Ollama for free)

# 2. Build the image & run ingestion once (crawls site + builds vector store)
docker compose run --rm chatbot python -m app.ingest
```

The ingestion output (crawled files + ChromaDB vectors) is saved to the
`chatbot-data` named volume, so it **persists across restarts and rebuilds**.

### Start the server

```bash
docker compose up -d
```

### Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok","indexed_chunks":1234,...}
```

### Re-ingest (refresh the knowledge base)

```bash
# Re-crawl + re-index the latest website content
docker compose run --rm chatbot python -m app.ingest
```

### Stop / clean up

```bash
docker compose down          # stop (keeps data volume)
docker compose down -v       # stop + DELETE the vector store
```

### Deploy to a cloud server (Render, Railway, DigitalOcean, AWS EC2)

1. Push this project to a Git repo (GitHub/GitLab).
2. On your server, `git clone` and run the Docker commands above.
3. Point your domain/reverse proxy to port `8000`.
4. Your frontend app calls `https://your-domain/chat` or `/chat/stream`.

> **Why embedded ChromaDB over a managed cloud DB (Pinecone)?**
> A college website produces a few thousand chunks — tiny for a vector DB.
> Embedded mode is free, has zero network latency, needs no account, and
> has no vendor lock-in. You can always migrate to a managed DB later if
> you ever scale to millions of documents.

---

## 📜 License

MIT — free to use and modify for your college.
