"""Dependency-free BM25 keyword retrieval over persisted Chroma chunks."""

from collections import Counter
from math import log
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document


def tokenize(text: str) -> list[str]:
    """Return lowercase word tokens suitable for local keyword matching."""
    return re.findall(r"\b\w+\b", text.lower())


def retrieve_with_bm25(
    vectorstore: Chroma,
    question: str,
    k: int,
    metadata_filter: dict[str, str] | None = None,
) -> list[Document]:
    """Return up to ``k`` chunks with a positive BM25 score for ``question``.

    Chroma remains the durable source of chunks. This intentionally keeps the
    learning project dependency-free; larger systems should maintain a native
    inverted index instead of rebuilding BM25 statistics at query time.
    """
    if k < 1:
        raise ValueError("BM25 k must be at least 1.")
    result = vectorstore.get(
        where=metadata_filter,
        include=["documents", "metadatas"],
    )
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(result.get("documents", []), result.get("metadatas", []), strict=True)
        if text
    ]
    query_terms = tokenize(question)
    if not documents or not query_terms:
        return []

    document_tokens = [tokenize(document.page_content) for document in documents]
    average_length = sum(len(tokens) for tokens in document_tokens) / len(document_tokens)
    if average_length == 0:
        return []
    document_frequency = Counter(
        term for tokens in document_tokens for term in set(tokens)
    )
    query_frequency = Counter(query_terms)
    scores: list[float] = []
    k1 = 1.5
    b = 0.75
    document_count = len(documents)
    for tokens in document_tokens:
        term_frequency = Counter(tokens)
        length_normalizer = k1 * (1 - b + b * len(tokens) / average_length)
        score = 0.0
        for term, count in query_frequency.items():
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = log(
                1 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            score += count * inverse_document_frequency * (frequency * (k1 + 1)) / (frequency + length_normalizer)
        scores.append(score)

    ranked_indexes = sorted(
        (index for index, score in enumerate(scores) if score > 0),
        key=lambda index: (-scores[index], index),
    )
    return [documents[index] for index in ranked_indexes[:k]]
