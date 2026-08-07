"""Application-facing retrieval boundary using Maximal Marginal Relevance."""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from vectorstore.mmr import retrieve_with_mmr
from vectorstore.hybrid import retrieve_with_hybrid
from vectorstore.reranking import rerank_documents


def retrieve_documents(
    vectorstore: Chroma,
    question: str,
    k: int,
    metadata_filter: dict[str, str] | None = None,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    hybrid: bool = False,
    rerank: bool = False,
    rerank_fetch_k: int = 12,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[Document]:
    """Return dense/hybrid chunks, optionally cross-encoder reranked."""
    if rerank and rerank_fetch_k < k:
        raise ValueError("Rerank fetch_k must be greater than or equal to k.")
    candidate_k = rerank_fetch_k if rerank else k
    candidate_fetch_k = max(fetch_k, candidate_k)
    if hybrid:
        documents = retrieve_with_hybrid(
            vectorstore,
            question,
            candidate_k,
            candidate_fetch_k,
            lambda_mult,
            metadata_filter,
        )
    else:
        documents = retrieve_with_mmr(
            vectorstore,
            question,
            candidate_k,
            candidate_fetch_k,
            lambda_mult,
            metadata_filter,
        )
    return rerank_documents(question, documents, k, rerank_model) if rerank else documents
