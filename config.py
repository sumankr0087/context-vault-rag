"""Application configuration kept separate from application behavior."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Store the values that configure the RAG application."""

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
    # ---- Provider selection (local Ollama by default; cloud for deployment) ----
    llm_provider: str          # "ollama" | "groq" | "openai"
    llm_api_key: str | None
    llm_base_url: str
    embedding_provider: str    # "ollama" | "huggingface" | "fastembed"
    hf_embedding_model: str
    fastembed_model: str
    # ---- RAG grounding / guardrails ----
    min_relevance_score: float


def get_settings() -> Settings:
    """Build settings from the environment (sensible local defaults)."""
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    # Cloud providers (Groq/OpenAI) use a chat model id; Ollama uses a local tag.
    default_chat = "openai/gpt-oss-120b" if llm_provider == "groq" else "qwen3.5:4b"
    return Settings(
        docs_dir=Path("docs"),
        data_dir=data_dir,
        chroma_dir=data_dir / "chroma",
        memory_db=data_dir / "memory.db",
        chat_model=os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", default_chat)),
        embedding_model=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        embedding_num_ctx=int(os.getenv("EMBEDDING_NUM_CTX", "1024")),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        collection_name=os.getenv("COLLECTION_NAME", "personal_documents"),
        memory_turns=10,
        chunk_size=900,
        chunk_overlap=150,
        retrieval_k=4,
        mmr_fetch_k=20,
        mmr_lambda_mult=0.5,
        rerank_fetch_k=12,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_provider=llm_provider,
        llm_api_key=(
            os.getenv("LLM_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        ),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama").lower(),
        hf_embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        fastembed_model=os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        min_relevance_score=float(os.getenv("MIN_RELEVANCE_SCORE", "0.15")),
    )
