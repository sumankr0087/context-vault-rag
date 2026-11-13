"""Policy version comparison — diff two uploaded documents and explain changes."""

from dataclasses import dataclass, field

from langchain_chroma import Chroma

from config import Settings
from llm.ollama_client import create_chat_model
from rag.context import final_answer_text


@dataclass(frozen=True)
class ComparisonResult:
    """Carry a plain-language diff of two document versions."""

    report: str
    source_a: str
    source_b: str


COMPARE_SYSTEM = """You compare two versions of a company policy / document and explain what changed.

Use ONLY the provided excerpts. Produce clear Markdown with these sections:
## Summary (1-2 sentences on the overall change)
## What changed (bullets: old → new, be specific with numbers/terms)
## Added (new clauses/obligations, or "None")
## Removed (clauses dropped, or "None")
## ⚠️ Compliance impact (new risks, stricter limits, or obligations to note)

If the excerpts don't show a meaningful difference, say so plainly."""


def _chunks_for_source(vectorstore: Chroma, source: str, limit: int = 40) -> str:
    """Return the concatenated text of a document's chunks (bounded)."""
    records = vectorstore.get(where={"source": source}, include=["documents"])
    texts = [t for t in (records.get("documents") or []) if t][:limit]
    return "\n\n".join(texts)


def compare_documents(
    vectorstore: Chroma,
    settings: Settings,
    source_a: str,
    name_a: str,
    source_b: str,
    name_b: str,
    topic: str | None = None,
) -> ComparisonResult:
    """Compare two document versions and return a plain-language diff report."""
    text_a = _chunks_for_source(vectorstore, source_a)
    text_b = _chunks_for_source(vectorstore, source_b)
    focus = f"\n\nFocus especially on: {topic}" if topic else ""
    human = (
        f"Version A ({name_a}):\n{text_a}\n\n"
        f"=====\n\nVersion B ({name_b}):\n{text_b}{focus}\n\n"
        "Compare Version A (older) with Version B (newer) and report what changed."
    )
    chat_model = create_chat_model(settings)
    response = chat_model.invoke(
        [("system", COMPARE_SYSTEM), ("human", human)]
    )
    return ComparisonResult(
        report=final_answer_text(response.content),
        source_a=name_a,
        source_b=name_b,
    )
