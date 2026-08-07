"""Shared context and answer-text transformations for RAG pipelines."""

from langchain_core.documents import Document


def format_document_context(documents: list[Document]) -> str:
    """Format retrieved chunks with source labels for inclusion in an LLM prompt."""
    return "\n\n---\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n{document.page_content}"
        for document in documents
    )


def final_answer_text(content: object) -> str:
    """Return visible model content or a clear message when no final text exists."""
    answer_text = str(content).strip()
    return answer_text or "The language model returned an empty answer. Please ask again."
