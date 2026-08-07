"""Tests for Maximal Marginal Relevance retrieval configuration."""

import unittest
from unittest.mock import Mock

from vectorstore.mmr import retrieve_with_mmr


class MmrTests(unittest.TestCase):
    """Verify MMR delegates correctly and rejects invalid tuning values."""

    def test_mmr_passes_candidate_count_and_diversity_weight_to_chroma(self) -> None:
        """Chroma receives the full MMR query contract."""
        vectorstore = Mock()
        vectorstore.max_marginal_relevance_search.return_value = []

        retrieve_with_mmr(vectorstore, "Explain RAG", 4, 20, 0.3, {"extension": ".pdf"})

        vectorstore.max_marginal_relevance_search.assert_called_once_with(
            "Explain RAG",
            k=4,
            fetch_k=20,
            lambda_mult=0.3,
            filter={"extension": ".pdf"},
        )

    def test_mmr_rejects_a_candidate_count_smaller_than_result_count(self) -> None:
        """MMR cannot select four results after fetching fewer than four candidates."""
        with self.assertRaises(ValueError):
            retrieve_with_mmr(Mock(), "Explain RAG", 4, 3, 0.5)

    def test_mmr_rejects_an_invalid_lambda_value(self) -> None:
        """The relevance/diversity balance must remain within its mathematical range."""
        with self.assertRaises(ValueError):
            retrieve_with_mmr(Mock(), "Explain RAG", 4, 20, 1.1)
