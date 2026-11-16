"""Maximal Marginal Relevance retrieval for diverse document context."""

from langchain_core.documents import Document


def retrieve_with_mmr(
    vectorstore: object,
    question: str,
    k: int,
    fetch_k: int,
    lambda_mult: float,
    metadata_filter: dict[str, str] | None = None,
) -> list[Document]:
    """Select ``k`` relevant but non-redundant chunks from ``fetch_k`` candidates."""
    if fetch_k < k:
        raise ValueError("MMR fetch_k must be greater than or equal to k.")
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("MMR lambda_mult must be between 0.0 and 1.0.")
    return vectorstore.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        filter=metadata_filter,
    )
