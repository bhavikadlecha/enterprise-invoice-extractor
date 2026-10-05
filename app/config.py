"""
config.py — Central environment-driven configuration (Pydantic-settings v2).
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM Provider ---
    llm_provider: Literal["openai", "anthropic", "gemini"] = "openai"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # --- Anthropic ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # --- Extraction ---
    max_retries: int = 3

    # --- RAG ---
    # Path where ChromaDB persists its on-disk collection
    chroma_persist_dir: str = "./chroma_db"

    # Embedding model for PO vector store (runs locally, no API key needed)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Number of PO candidates to retrieve before re-ranking
    rag_top_k: int = 3

    # --- Observability ---
    # SQLite database file for storing extraction metrics
    metrics_db_path: str = "./metrics.db"

    # --- Batch ---
    # Maximum number of PDFs allowed in a single batch ZIP upload
    batch_max_files: int = 50


settings = Settings()
