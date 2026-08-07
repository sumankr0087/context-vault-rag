"""Tests for metadata enrichment and filtered retrieval delegation."""

import unittest
from pathlib import Path
from unittest.mock import Mock

from langchain_core.documents import Document

from utils.metadata import add_document_metadata, build_extension_filter, build_retrieval_filter
from vectorstore.retrieval import retrieve_documents


class MetadataTests(unittest.TestCase):
    """Verify the Milestone 2 metadata-filtering contract."""

    def test_add_document_metadata_records_source_and_normalized_extension(self) -> None:
        """Each loaded document keeps attributes needed for attribution and filtering."""
        document = Document(page_content="Example")

        add_document_metadata(document, Path("docs/Guide.PDF"))

        self.assertEqual(document.metadata["source"], "docs/Guide.PDF")
        self.assertEqual(document.metadata["extension"], ".pdf")

    def test_build_extension_filter_normalizes_a_user_value(self) -> None:
        """Users may write pdf or .PDF while Chroma receives a canonical filter."""
        self.assertEqual(build_extension_filter("PDF"), {"extension": ".pdf"})
        self.assertIsNone(build_extension_filter(None))

    def test_build_retrieval_filter_combines_extension_and_document_type(self) -> None:
        """Multiple UI retrieval scopes become an explicit Chroma AND filter."""
        self.assertEqual(
            build_retrieval_filter(".pdf", "resume"),
            {"$and": [{"extension": ".pdf"}, {"document_type": "resume"}]},
        )

    def test_build_retrieval_filter_limits_all_uploads_to_known_categories(self) -> None:
        """The UI's all-uploads mode excludes the original sample corpus."""
        self.assertEqual(
            build_retrieval_filter(uploaded_only=True),
            {
                "$or": [
                    {"document_type": "resume"},
                    {"document_type": "job_description"},
                    {"document_type": "other"},
                ]
            },
        )

    def test_build_retrieval_filter_limits_search_to_selected_sources(self) -> None:
        """A document picker narrows retrieval to the exact uploaded files selected."""
        self.assertEqual(
            build_retrieval_filter(uploaded_only=True, sources=["docs/uploaded/other/certificate.pdf"]),
            {
                "$and": [
                    {
                        "$or": [
                            {"document_type": "resume"},
                            {"document_type": "job_description"},
                            {"document_type": "other"},
                        ]
                    },
                    {"source": {"$in": ["docs/uploaded/other/certificate.pdf"]}},
                ]
            },
        )

    def test_retrieval_passes_metadata_filter_to_chroma(self) -> None:
        """Chroma performs metadata filtering before ranking semantic matches."""
        vectorstore = Mock()
        vectorstore.max_marginal_relevance_search.return_value = []

        retrieve_documents(vectorstore, "What is RAG?", 4, {"extension": ".pdf"})

        vectorstore.max_marginal_relevance_search.assert_called_once_with(
            "What is RAG?",
            k=4,
            fetch_k=20,
            lambda_mult=0.5,
            filter={"extension": ".pdf"},
        )
