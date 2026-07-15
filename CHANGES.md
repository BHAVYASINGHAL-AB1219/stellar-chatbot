# NIT Sikkim Chatbot — Session Change Log

This document details every change made during this session to migrate the
chatbot from local embedding models to NVIDIA's hosted NIM endpoint, make it
a hybrid conversational + RAG bot, and switch the LLM to `glm-5.2`.

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1 — Removed Local Embedding Models](#phase-1--removed-local-embedding-models)
3. [Phase 2 — NVIDIA Hosted Embeddings](#phase-2--nvidia-hosted-embeddings)
4. [Phase 3 — Hybrid Conversational + RAG Bot](#phase-3--hybrid-conversational--rag-bot)
5. [Phase 4 — Removed Website-Redirect Behavior](#phase-4--removed-website-redirect-behavior)
6. [Phase 5 — Switched LLM to glm-5.2 via NVIDIA](#phase-5--switched-llm-to-glm-52-via-nvidia)
7. [Files Modified](#files-modified)
8. [Final Architecture](#final-architecture)
9. [How to Run](#how-to-run)

---

## Overview

The chatbot was originally configured to load embedding models **locally** via
`sentence-transformers` (consuming RAM/CPU on every run) and used a strict
RAG-only system prompt that made greetings feel cold. This session:

- **Deleted** all locally cached embedding models (~772 MB freed).
- **Migrated embeddings** to NVIDIA's hosted NIM endpoint
  (`nvidia/llama-nemotron-embed-1b-v2`, 2048-dim, asymmetric).
- **Made the bot conversational** — it now handles greetings/small-talk
  naturally while still grounding college answers in retrieved context.
- **Removed website-redirect language** so the bot acts as the primary
  information source instead of pushing users to the website.
- **Switched the LLM** from inference.net's `gpt-4o-mini` to NVIDIA's
  `z-ai/glm-5.2`.

Both the LLM and embeddings now run entirely through NVIDIA's hosted endpoint —
**no local models are loaded**.

---

## Phase 1 — Removed Local Embedding Models

### What was found

The HuggingFace cache at `~/.cache/huggingface/hub/` contained:

| Model | Type | Size |
|-------|------|------|
| `LiquidAI/LFM2.5-Embedding-350M` | Embedding | 681 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | Embedding | 87 MB |
| `mistralai/Mistral-7B-Instruct-v0.3` | LLM | 2.6 MB |

Total cache: **~772 MB**.

### What was done

Deleted the two embedding models:

```bash
rm -rf ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-Embedding-350M
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
```

Cache dropped from **772 MB → 3.7 MB** (only the Mistral LLM remains).

### Why

Loading embedding models locally consumes RAM/CPU on every run. By moving to a
hosted endpoint, the app stays lightweight and embeddings are computed server-side.

---

## Phase 2 — NVIDIA Hosted Embeddings

### Investigation

Before implementing, several endpoints were tested:

| Endpoint | Result |
|----------|--------|
| `api-inference.huggingface.co` (old HF API) | **Dead** — DNS no longer resolves (deprecated) |
| `router.huggingface.co` (new HF API) | Works, but `LiquidAI/LFM2.5-Embedding-350M` is **not hosted** for inference |
| `router.huggingface.co` with `all-MiniLM-L6-v2` | Works (returns embeddings) |
| OpenRouter `/v1/embeddings` | Works but account had **0 credits** (HTTP 402) |
| NVIDIA `integrate.api.nvidia.com/v1` | **Works** — key valid, model available |

### Model chosen

`nvidia/llama-nemotron-embed-1b-v2` — available on NVIDIA's NIM endpoint,
produces **2048-dimensional** vectors.

### Key technical detail: asymmetric embeddings

This model is **asymmetric** — it requires an `input_type` field:

- `"passage"` → for indexing documents into the vector store
- `"query"` → for searching with a user question

Without `input_type`, NVIDIA returns:
`{"error":"'input_type' parameter is required for asymmetric models"}`

### Changes made

#### `app/embeddings.py`

1. **Extended the [`Embedder`](app/embeddings.py:33) interface** with an optional
   `input_type` parameter on the `embed()` method. Providers that don't use it
   simply ignore the argument.

2. **Updated all existing embedders** (`SentenceTransformerEmbedder`,
   `OpenAIEmbedder`, `GeminiEmbedder`, `OllamaEmbedder`) to accept
   `input_type` in their `embed()` signatures.

3. **Added [`NvidiaEmbedder`](app/embeddings.py:184)** — calls NVIDIA's
   OpenAI-compatible endpoint with the `input_type` field passed via
   `extra_body`. Batches in groups of 64 to stay within size limits.

4. **Registered `"nvidia"`** in the `_EMBEDDERS` factory dict.

#### `app/config.py`

Added NVIDIA embedding settings:

```python
nvidia_api_key: str = ""
nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
nvidia_embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
```

Added `"nvidia"` to the `embedding_provider` Literal options.

#### `app/vectorstore.py`

Updated the two `embed()` call sites to pass the correct `input_type`:

- [`add_chunks()`](app/vectorstore.py:52) → `input_type="passage"` (indexing documents)
- [`search()`](app/vectorstore.py:85) → `input_type="query"` (user search)

#### `.env` / `.env.example`

Switched `EMBEDDING_PROVIDER=nvidia` and added the NVIDIA key, base URL, and
embedding model.

### Verification

- **Smoke test**: `NvidiaEmbedder` loaded, both `passage` and `query` embeddings
  returned 2048-dim vectors.
- **Full re-ingestion**: 830 chunks indexed via the NVIDIA endpoint.
- **RAG query**: *"Where is NIT Sikkim located?"* → retrieved relevant chunks
  and answered correctly with cited sources.

---

## Phase 3 — Hybrid Conversational + RAG Bot

### Problem

The original [`SYSTEM_PROMPT`](app/rag.py:21) forced the LLM to answer
**only** from retrieved context. Greetings like "hi" / "hello" got the cold
fallback response instead of a natural reply.

### Changes made

#### `app/rag.py`

1. **Rewrote `SYSTEM_PROMPT`** — the bot now behaves as a friendly, normal
   chatbot that greets users and makes small talk naturally, while still
   grounding college-specific answers in retrieved context with source
   citations.

2. **Updated [`_build_messages()`](app/rag.py:66)** — when context is available
   it's included (RAG mode); when there's no context the prompt is just the
   user's message (pure chat mode for greetings/small talk).

3. **Rewrote [`answer()`](app/rag.py:82) and
   [`answer_stream()`](app/rag.py:118)** — removed the hard
   "I don't have any indexed information" fallback. The bot now always
   converses naturally via the LLM, and uses RAG grounding when relevant
   context is retrieved.

### Verification

- **"hi"** → *"Hello! How are you today?..."* (natural greeting)
- **"how are you doing today?"** → *"Hello! I'm doing great, thank you for
  asking!..."* (natural small talk)
- **"Where is NIT Sikkim located?"** → *"NIT Sikkim is located in Ravangla,
  Sikkim... [1][2]"* (RAG-grounded, cited sources)

---

## Phase 4 — Removed Website-Redirect Behavior

### Problem

The bot kept telling users to "check the official website at
https://www.nitsikkim.ac.in", which defeats the goal of making the bot the
primary place people come for information.

### Changes made

#### `app/rag.py`

Rewrote `SYSTEM_PROMPT` again to:

- Add an explicit goal: *"Your goal is to be the primary place people come for
  information about NIT Sikkim, so they don't have to visit the website
  themselves."*
- Remove the rule that told the bot to point users to the official website.
- New rule: when there's no relevant info, say so honestly and briefly
  (e.g. *"I don't have that information right now."*) — **do NOT repeatedly
  tell the user to visit the website, and do NOT mention the website URL
  unless the user explicitly asks for it.**
- Kept the no-invention rule (don't make up facts/dates/names/policies).

### Verification

- **"Where is NIT Sikkim located?"** → confident answer, **no website mention**.
- **"What is the weather like on Mars?"** → *"I don't have that information
  right now. However, if you have any questions about NIT Sikkim, feel free to
  ask!"* — honest gap, **no website push**.
- **"hello"** → natural friendly greeting.

---

## Phase 5 — Switched LLM to glm-5.2 via NVIDIA

### Investigation

Queried NVIDIA's `/v1/models` endpoint to find available chat models matching
"glm". Found: `z-ai/glm-5.2`.

Verified it works for chat completions:

```bash
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Authorization: Bearer nvapi-..." \
  -d '{"model":"z-ai/glm-5.2","messages":[{"role":"user","content":"Say hello"}]}'
# → HTTP 200, returned "Hello!"
```

### Changes made

#### `app/llm.py`

1. **Added [`NvidiaLLM`](app/llm.py:155) class** — reuses the OpenAI client
   pointed at NVIDIA's endpoint (`integrate.api.nvidia.com/v1`) with the
   NVIDIA API key. Supports both `generate()` (one-shot) and `stream()`
   (token streaming).

2. **Registered `"nvidia"`** in the `_LLMs` factory dict.

3. **Fixed a pre-existing bug** — the factory dict was named `_LLMs` but
   `get_llm()` referenced `_LLMS` (different casing), causing a
   `NameError`. Standardized on `_LLMs`.

#### `app/config.py`

Added `nvidia_chat_model` setting and `"nvidia"` to the `llm_provider`
Literal options:

```python
llm_provider: Literal["openai", "gemini", "ollama", "nvidia"] = "openai"
nvidia_chat_model: str = "z-ai/glm-5.2"
```

#### `.env` / `.env.example`

Switched `LLM_PROVIDER=nvidia` and added `NVIDIA_CHAT_MODEL=z-ai/glm-5.2`.

### Verification

- `🔧 Loading LLM: nvidia` — LLM loaded via NVIDIA endpoint.
- **Greeting** → *"Hi there! 👋 Welcome to the NIT Sikkim assistant..."*
- **College question** → *"NIT Sikkim is located at its **Ravangla Campus,
  South Sikkim 77139** [1]..."* (RAG-grounded, cited sources)
- **Streaming** → streamed a complete answer with citations `[1][2][3]`

---

## Files Modified

| File | Changes |
|------|---------|
| [`app/embeddings.py`](app/embeddings.py) | Added `input_type` to `Embedder` interface; updated all embedders; added `NvidiaEmbedder` |
| [`app/llm.py`](app/llm.py) | Added `NvidiaLLM`; registered in factory; fixed `_LLMs` casing bug |
| [`app/config.py`](app/config.py) | Added `nvidia` provider options + `nvidia_chat_model` / `nvidia_embedding_model` settings |
| [`app/vectorstore.py`](app/vectorstore.py) | Pass `input_type="passage"` at index time, `input_type="query"` at search time |
| [`app/rag.py`](app/rag.py) | Rewrote `SYSTEM_PROMPT` (conversational + no website redirect); hybrid `_build_messages` / `answer` / `answer_stream` |
| [`.env`](.env) | `LLM_PROVIDER=nvidia`, `EMBEDDING_PROVIDER=nvidia`, NVIDIA key + models |
| [`.env.example`](.env.example) | Documented NVIDIA LLM + embedding options |

---

## Final Architecture

```
User question
    │
    ▼
┌─────────────────────────────────────────────┐
│  app/rag.py  (answer / answer_stream)       │
│                                             │
│  1. Retrieve context from vector store      │
│  2. Build messages (RAG mode or chat mode)  │
│  3. Send to LLM                             │
└──────────────┬───────────────┬───────────────┘
               │               │
               ▼               ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Embeddings          │  │  LLM                  │
│  NVIDIA NIM endpoint │  │  NVIDIA NIM endpoint  │
│  llama-nemotron-     │  │  z-ai/glm-5.2         │
│  embed-1b-v2 (2048d) │  │                       │
│  input_type:         │  │  OpenAI-compatible    │
│    passage / query   │  │  chat completions     │
└──────────────────────┘  └──────────────────────┘
               │
               ▼
┌──────────────────────┐
│  Vector Store         │
│  ChromaDB (local)     │
│  830 chunks indexed   │
│  cosine similarity    │
└──────────────────────┘
```

**No local models are loaded.** Both the LLM and embeddings run through
NVIDIA's hosted NIM endpoint. Only ChromaDB (the vector store) runs locally.

---

## How to Run

### 1. Re-ingest (build the vector store with NVIDIA embeddings)

```bash
python -m app.ingest --no-crawl
```

### 2. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Open the chat UI

Navigate to `http://localhost:8000` in your browser.

### 4. Test the API directly

```bash
# One-shot answer
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is NIT Sikkim located?"}'

# Streaming answer (Server-Sent Events)
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "hi"}'
```

### 5. Check health

```bash
curl http://localhost:8000/health
# → {"status":"ok","llm_provider":"nvidia","embedding_provider":"nvidia","indexed_chunks":830}
```
