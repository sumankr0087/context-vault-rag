"""Tests for local cross-encoder reranking without loading a model."""

import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from vectorstore.reranking import rerank_documents
from vectorstore.retrieval import retrieve_documents


class FakeCrossEncoder:
    """Return deterministic pair scores in the same order as supplied pairs."""

    def predict(self, _pairs: list[tuple[str, str]]) -> list[float]:
        """Make the second supplied chunk the most relevant one."""
        return [0.1, 0.9, 0.4]


class RerankingTests(unittest.TestCase):
    """Verify score ordering and widened first-stage retrieval offline."""

    @patch("vectorstore.reranking._load_cross_encoder", return_value=FakeCrossEncoder())
    def test_reranker_orders_chunks_by_cross_encoder_score(self, _loader: Mock) -> None:
        """The final prompt receives the highest pair scores first."""
        documents = [
            Document(page_content="first"),
            Document(page_content="second"),
            Document(page_content="third"),
        ]

        reranked = rerank_documents("question", documents, 2, "test-model")

        self.assertEqual([document.page_content for document in reranked], ["second", "third"])

    @patch("vectorstore.retrieval.rerank_documents")
    def test_retrieval_widens_candidates_before_reranking(self, rerank: Mock) -> None:
        """Twelve dense candidates are retrieved before the final four are scored."""
        candidates = [Document(page_content=f"chunk {index}") for index in range(12)]
        vectorstore = Mock()
        vectorstore.max_marginal_relevance_search.return_value = candidates
        rerank.return_value = candidates[:4]

        documents = retrieve_documents(
            vectorstore,
            "question",
            4,
            rerank=True,
            rerank_fetch_k=12,
            rerank_model="test-model",
        )

        vectorstore.max_marginal_relevance_search.assert_called_once_with(
            "question", k=12, fetch_k=20, lambda_mult=0.5, filter=None
        )
        rerank.assert_called_once_with("question", candidates, 4, "test-model")
        self.assertEqual(documents, candidates[:4])

    def test_retrieval_rejects_a_rerank_candidate_count_smaller_than_result_count(self) -> None:
        """The reranker cannot select four final chunks from fewer candidates."""
        with self.assertRaises(ValueError):
            retrieve_documents(Mock(), "question", 4, rerank=True, rerank_fetch_k=3)
