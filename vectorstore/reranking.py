"""Cross-encoder reranking for a small set of retrieved candidate chunks."""

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document


def rerank_documents(
    question: str,
    documents: list[Document],
    k: int,
    model_name: str,
) -> list[Document]:
    """Use a cross-encoder to return the best ``k`` question-chunk pairs."""
    if k < 1:
        raise ValueError("Reranking k must be at least 1.")
    if not documents:
        return []
    reranker = _load_cross_encoder(model_name)
    scores = reranker.predict([(question, document.page_content) for document in documents])
    ranked_indexes = sorted(range(len(documents)), key=lambda index: (-float(scores[index]), index))
    return [documents[index] for index in ranked_indexes[:k]]


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> Any:
    """Load and cache the optional local cross-encoder only when reranking is used."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "Reranking requires sentence-transformers. Install it with: "
            "./venv/bin/pip install sentence-transformers"
        ) from error
    return CrossEncoder(model_name)
