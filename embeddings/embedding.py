"""Embedding-model construction — local Ollama or serverless HuggingFace.

HuggingFace (sentence-transformers) runs in-process with no external server, so
it works on free cloud hosting. Ollama remains the default for local/private use.
"""

from config import Settings


def create_embeddings(settings: Settings):
    """Create the embedding client used by Chroma for indexing and retrieval."""
    if settings.embedding_provider == "fastembed":
        # ONNX-based, no PyTorch — fits small (512MB) cloud instances.
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=settings.fastembed_model)

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
