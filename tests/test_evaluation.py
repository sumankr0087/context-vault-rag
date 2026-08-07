"""Tests for source-level retrieval evaluation metrics."""

import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from config import get_settings
from evaluation.retrieval import (
    EvaluationCase,
    evaluate_retrieval,
    format_evaluation_report,
)


class EvaluationTests(unittest.TestCase):
    """Verify hit-rate and MRR calculations without contacting a vector database."""

    @patch("evaluation.retrieval.retrieve_documents")
    def test_evaluation_calculates_hit_rate_and_mrr(self, retrieve: Mock) -> None:
        """An expected first rank and expected second rank produce known metrics."""
        retrieve.side_effect = [
            [Document(page_content="SQL", metadata={"source": "docs/sql.pdf"})],
            [
                Document(page_content="Other", metadata={"source": "docs/other.txt"}),
                Document(page_content="Python", metadata={"source": "docs/python.pdf"}),
            ],
            [Document(page_content="Miss", metadata={"source": "docs/miss.pdf"})],
        ]
        cases = [
            EvaluationCase("SQL?", ["docs/sql.pdf"]),
            EvaluationCase("Python?", ["docs/python.pdf"]),
            EvaluationCase("AI?", ["docs/ai.txt"]),
        ]

        report = evaluate_retrieval(Mock(), cases, get_settings())

        self.assertAlmostEqual(report.hit_rate_at_k, 2 / 3)
        self.assertAlmostEqual(report.mean_reciprocal_rank, 0.5)
        self.assertIn("2. PASS — Python?", format_evaluation_report(report))
        self.assertIn("3. MISS — AI?", format_evaluation_report(report))

    def test_empty_evaluation_report_has_zero_metrics(self) -> None:
        """An empty dataset has defined metrics and does not divide by zero."""
        report = evaluate_retrieval(Mock(), [], get_settings())

        self.assertEqual(report.hit_rate_at_k, 0.0)
        self.assertEqual(report.mean_reciprocal_rank, 0.0)
