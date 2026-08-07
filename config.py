"""Application configuration kept separate from application behavior."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Store the values that configure the local RAG application."""

    docs_dir: Path
    data_dir: Path
    chroma_dir: Path
    memory_db: Path
    chat_model: str
    embedding_model: str
    embedding_num_ctx: int
    ollama_url: str
    collection_name: str
    memory_turns: int
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    mmr_fetch_k: int
    mmr_lambda_mult: float
    rerank_fetch_k: int
    rerank_model: str


def get_settings() -> Settings:
    """Build the default settings, allowing the Ollama URL to be overridden."""
    data_dir = Path("data")
    return Settings(
        docs_dir=Path("docs"),
        data_dir=data_dir,
        chroma_dir=data_dir / "chroma",
        memory_db=data_dir / "memory.db",
        chat_model="qwen3.5:4b",
        embedding_model="qwen3-embedding:0.6b",
        embedding_num_ctx=int(os.getenv("EMBEDDING_NUM_CTX", "1024")),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        collection_name="personal_documents",
        memory_turns=10,
        chunk_size=900,
        chunk_overlap=150,
        retrieval_k=4,
        mmr_fetch_k=20,
        mmr_lambda_mult=0.5,
        rerank_fetch_k=12,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
