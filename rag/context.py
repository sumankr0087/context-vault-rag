"""Shared context and answer-text transformations for RAG pipelines."""

from pathlib import Path

from langchain_core.documents import Document


def _display_source(document: Document) -> str:
    """Human-friendly source name (original filename, not the stored path)."""
    name = str(document.metadata.get("document_name") or "")
    if name:
        return name
    source = str(document.metadata.get("source", "unknown"))
    filename = Path(source).name
    prefix, sep, original = filename.partition("_")
    return original if sep and len(prefix) == 32 else filename


def _page_of(document: Document) -> int | None:
    """Return a 1-based page number when the loader recorded one."""
    page = document.metadata.get("page")
    if page is None:
        page = document.metadata.get("page_number")
    try:
        return int(page) + 1 if document.metadata.get("page") is not None else int(page)
    except (TypeError, ValueError):
        return None


def format_document_context(documents: list[Document]) -> str:
    """Number each retrieved chunk [n] with its source/page so the LLM can cite it."""
    blocks = []
    for index, document in enumerate(documents, start=1):
        page = _page_of(document)
        label = _display_source(document) + (f", p.{page}" if page else "")
        blocks.append(f"[{index}] Source: {label}\n{document.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_citations(documents: list[Document]) -> list[dict]:
    """Return structured citations (index, source, page, snippet) for the UI."""
    citations = []
    for index, document in enumerate(documents, start=1):
        snippet = " ".join(document.page_content.split())
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "…"
        citations.append(
            {
                "index": index,
                "source": _display_source(document),
                "page": _page_of(document),
                "snippet": snippet,
            }
        )
    return citations


def final_answer_text(content: object) -> str:
    """Return visible model content or a clear message when no final text exists."""
    answer_text = str(content).strip()
    return answer_text or "The language model returned an empty answer. Please ask again."
