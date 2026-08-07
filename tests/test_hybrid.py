"""Tests for BM25 keyword retrieval and reciprocal-rank hybrid fusion."""

import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from vectorstore.hybrid import retrieve_with_hybrid
from vectorstore.keyword import retrieve_with_bm25


class HybridRetrievalTests(unittest.TestCase):
    """Verify exact-term retrieval, filtering, and rank fusion offline."""

    def test_bm25_prefers_a_chunk_with_exact_query_terms(self) -> None:
        """Keyword retrieval gives a strong rank to literal technical terms."""
        vectorstore = Mock()
        vectorstore.get.return_value = {
            "documents": ["The SELECT statement reads rows from a table.", "Python uses functions."],
            "metadatas": [{"source": "docs/sql.pdf"}, {"source": "docs/python.txt"}],
        }

        documents = retrieve_with_bm25(vectorstore, "What does SELECT read?", 1)

        self.assertEqual(documents[0].metadata["source"], "docs/sql.pdf")
        vectorstore.get.assert_called_once_with(where=None, include=["documents", "metadatas"])

    def test_bm25_passes_the_metadata_filter_to_chroma(self) -> None:
        """The lexical branch obeys the same extension restriction as dense search."""
        vectorstore = Mock()
        vectorstore.get.return_value = {"documents": [], "metadatas": []}

        retrieve_with_bm25(vectorstore, "SELECT", 4, {"extension": ".pdf"})

        vectorstore.get.assert_called_once_with(
            where={"extension": ".pdf"}, include=["documents", "metadatas"]
        )

    def test_hybrid_fuses_and_deduplicates_dense_and_keyword_rankings(self) -> None:
        """A chunk ranked by both methods wins fusion and appears only once."""
        shared = Document(page_content="SELECT reads rows.", metadata={"source": "docs/sql.pdf"})
        dense_only = Document(page_content="Vectors find semantic matches.", metadata={"source": "docs/rag.txt"})
        keyword_only = Document(page_content="SELECT filters columns.", metadata={"source": "docs/sql.pdf"})
        vectorstore = Mock()
        vectorstore.max_marginal_relevance_search.return_value = [shared, dense_only]
        vectorstore.get.return_value = {
            "documents": [shared.page_content, keyword_only.page_content],
            "metadatas": [shared.metadata, keyword_only.metadata],
        }

        documents = retrieve_with_hybrid(vectorstore, "SELECT", 2, 4, 0.5)

        self.assertEqual(documents, [shared, dense_only])
        self.assertEqual(len({document.page_content for document in documents}), 2)
