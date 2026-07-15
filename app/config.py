"""
Central configuration for the College Chatbot.
All settings are read from environment variables (see .env).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Provider selection ----
    llm_provider: Literal["openai", "gemini", "ollama", "nvidia"] = "openai"
    embedding_provider: Literal[
        "sentence-transformers", "openai", "gemini", "ollama", "nvidia"
    ] = "sentence-transformers"

    # ---- OpenAI (or any OpenAI-compatible endpoint like inference.net) ----
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ---- Pinecone ----
    pinecone_api_key: str = ""
    pinecone_index_name: str = "college-chatbot"

    # ---- API Security ----
    ingest_api_key: str = ""

    # ---- Gemini ----
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "text-embedding-004"

    # ---- Ollama ----
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    # ---- Sentence Transformers ----
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    # ---- HuggingFace (for downloading gated/private models) ----
    hf_token: str = ""

    # ---- NVIDIA NIM (build.nvidia.com — OpenAI-compatible, asymmetric models) ----
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_chat_model: str = "z-ai/glm-5.2"
    nvidia_embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"

    # ---- Crawler ----
    site_url: str = "https://www.nitsikkim.ac.in"
    crawl_max_pages: int = 500
    crawl_delay: float = 1.0
    crawl_user_agent: str = "StellarCollegeBot/1.0 (+college-chatbot)"

    # ---- Storage ----
    data_dir: str = "./data"
    raw_dir: str = "./data/raw"

    # ---- RAG tuning ----
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5

    # ---- Derived paths (computed) ----
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_dir)



settings = Settings()
