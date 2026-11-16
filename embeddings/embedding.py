"""Embedding-model construction — local Ollama or serverless HuggingFace.

HuggingFace (sentence-transformers) runs in-process with no external server, so
it works on free cloud hosting. Ollama remains the default for local/private use.
"""

from config import Settings


def create_embeddings(settings: Settings):
    """Create the embedding client used by Chroma for indexing and retrieval."""
    if settings.embedding_provider == "fastembed":
        # ONNX-based, no PyTorch. Single-threaded with a small batch keeps the
        # memory spike during indexing low enough for a 512MB instance.
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(
            model_name=settings.fastembed_model,
            threads=1,
            batch_size=1,
        )

    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_url,
        num_ctx=settings.embedding_num_ctx,
    )
