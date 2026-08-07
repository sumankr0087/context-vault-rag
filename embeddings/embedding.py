"""Ollama embedding-model construction."""

from config import Settings
from langchain_ollama import OllamaEmbeddings


def create_embeddings(settings: Settings) -> OllamaEmbeddings:
    """Create the embedding client used by Chroma for indexing and retrieval."""
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_url,
        num_ctx=settings.embedding_num_ctx,
    )
