# ============================================================
# NIT Sikkim College Chatbot — Docker image
# ============================================================
# Multi-stage: build a slim runtime image.
# The Pinecone vector store is cloud-managed; the local data
# volume only stores raw crawled files for re-ingestion.
# ============================================================

FROM python:3.11-slim AS runtime

# System deps for pypdf / python-docx / lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY .env.example ./.env.example

# Create data directory (mounted as a volume in compose)
RUN mkdir -p /app/data/raw /app/data/vectorstore

# Expose the FastAPI port
EXPOSE 8000

# Run the API server.
# NOTE: Run ingestion ONCE before serving:
#   docker compose run --rm chatbot python -m app.ingest
# After that, the indexed data lives in the volume and persists.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
