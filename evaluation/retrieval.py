"""Evaluate retrieval quality against question-to-source expectations."""

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import Settings
from vectorstore.retrieval import retrieve_documents


@dataclass(frozen=True)
class EvaluationCase:
    """One question and the source paths that should be retrieved for it."""

    question: str
    expected_sources: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and retrieved source paths for one evaluation question."""

    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    reciprocal_rank: float

    @property
    def hit(self) -> bool:
        """Report whether at least one expected source was retrieved."""
        return self.reciprocal_rank > 0


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate retrieval quality measures and their per-question evidence."""

    results: list[EvaluationResult]

    @property
    def hit_rate_at_k(self) -> float:
        """Return the fraction of questions with an expected source in top-k."""
        return sum(result.hit for result in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_reciprocal_rank(self) -> float:
        """Return the average reciprocal rank of the first expected source."""
        return sum(result.reciprocal_rank for result in self.results) / len(self.results) if self.results else 0.0


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate a JSON list of evaluation questions and expected sources."""
    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Evaluation dataset was not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Evaluation dataset contains invalid JSON: {path}") from error
    if not isinstance(raw_cases, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    cases: list[EvaluationCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Evaluation case {index} must be an object.")
        question = raw_case.get("question")
        expected_sources = raw_case.get("expected_sources")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Evaluation case {index} needs a non-empty question.")
        if not isinstance(expected_sources, list) or not all(isinstance(source, str) for source in expected_sources):
            raise ValueError(f"Evaluation case {index} needs an expected_sources string list.")
        cases.append(EvaluationCase(question.strip(), expected_sources))
    return cases


def evaluate_retrieval(
    vectorstore: Chroma,
    cases: list[EvaluationCase],
    settings: Settings,
    metadata_filter: dict[str, str] | None = None,
    mmr_lambda_mult: float | None = None,
    hybrid: bool = False,
    rerank: bool = False,
) -> EvaluationReport:
    """Run the configured retrieval strategy and calculate source-level metrics."""
    lambda_mult = settings.mmr_lambda_mult if mmr_lambda_mult is None else mmr_lambda_mult
    results: list[EvaluationResult] = []
    for case in cases:
        documents = retrieve_documents(
            vectorstore,
            case.question,
            settings.retrieval_k,
            metadata_filter,
            settings.mmr_fetch_k,
            lambda_mult,
            hybrid,
            rerank,
            settings.rerank_fetch_k,
            settings.rerank_model,
        )
        retrieved_sources = _ordered_sources(documents)
        results.append(
            EvaluationResult(
                question=case.question,
                expected_sources=case.expected_sources,
                retrieved_sources=retrieved_sources,
                reciprocal_rank=_reciprocal_rank(retrieved_sources, case.expected_sources),
            )
        )
    return EvaluationReport(results)


def format_evaluation_report(report: EvaluationReport) -> str:
    """Render a concise terminal report with per-question retrieval evidence."""
    lines = [
        f"Evaluation questions: {len(report.results)}",
        f"Hit rate@k: {report.hit_rate_at_k:.1%}",
        f"Mean reciprocal rank: {report.mean_reciprocal_rank:.3f}",
    ]
    for index, result in enumerate(report.results, start=1):
        status = "PASS" if result.hit else "MISS"
        lines.extend(
            [
                "",
                f"{index}. {status} — {result.question}",
                f"   Expected: {', '.join(result.expected_sources)}",
                f"   Retrieved: {', '.join(result.retrieved_sources) or 'none'}",
            ]
        )
    return "\n".join(lines)


def _ordered_sources(documents: list[Document]) -> list[str]:
    """Return unique source paths in retrieval rank order."""
    sources: list[str] = []
    for document in documents:
        source = str(document.metadata.get("source", "unknown"))
        if source not in sources:
            sources.append(source)
    return sources


def _reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Return 1/rank for the first expected source, or zero when absent."""
    expected = set(expected_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return 1 / rank
    return 0.0
