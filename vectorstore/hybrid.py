"""Fuse dense MMR and BM25 keyword results with reciprocal-rank fusion."""

from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document

from vectorstore.keyword import retrieve_with_bm25
from vectorstore.mmr import retrieve_with_mmr


def retrieve_with_hybrid(
    vectorstore: Chroma,
    question: str,
    k: int,
    fetch_k: int,
    lambda_mult: float,
    metadata_filter: dict[str, str] | None = None,
    rrf_k: int = 60,
) -> list[Document]:
    """Fuse MMR and BM25 rankings, returning at most ``k`` unique chunks."""
    if rrf_k < 1:
        raise ValueError("RRF k must be at least 1.")
    dense_results = retrieve_with_mmr(
        vectorstore,
        question,
        k,
        fetch_k,
        lambda_mult,
        metadata_filter,
    )
    keyword_results = retrieve_with_bm25(vectorstore, question, k, metadata_filter)
    scores: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
    documents: dict[tuple[str, tuple[tuple[str, str], ...]], Document] = {}
    first_seen: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}

    for ranking in (dense_results, keyword_results):
        for rank, document in enumerate(ranking, start=1):
            key = _document_key(document)
            documents[key] = document
            first_seen.setdefault(key, len(first_seen))
            scores[key] += 1 / (rrf_k + rank)

    ranked_keys = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return [documents[key] for key in ranked_keys[:k]]


def _document_key(document: Document) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Build a stable, hashable chunk identity for rank fusion deduplication."""
    metadata = tuple(sorted((str(key), repr(value)) for key, value in document.metadata.items()))
    return document.page_content, metadata
