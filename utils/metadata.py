"""Metadata creation, filtering, and formatting helpers."""

from pathlib import Path

from langchain_core.documents import Document


def add_document_metadata(document: Document, source_path: Path) -> None:
    """Attach searchable source and file-extension metadata to one document."""
    document.metadata["source"] = str(source_path)
    document.metadata["extension"] = source_path.suffix.lower()


def build_extension_filter(extension: str | None) -> dict[str, str] | None:
    """Build a Chroma metadata filter for an extension, or no filter when omitted."""
    if extension is None:
        return None
    normalized_extension = extension.strip().lower()
    if not normalized_extension:
        return None
    if not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"
    return {"extension": normalized_extension}


def build_retrieval_filter(
    extension: str | None = None,
    document_type: str | None = None,
    uploaded_only: bool = False,
    sources: list[str] | None = None,
) -> dict[str, object] | None:
    """Build one Chroma filter from file type, upload, and exact-source scopes."""
    filters: list[dict[str, object]] = []
    extension_filter = build_extension_filter(extension)
    if extension_filter:
        filters.append(extension_filter)
    if document_type:
        filters.append({"document_type": document_type})
    elif uploaded_only:
        filters.append(
            {
                "$or": [
                    {"document_type": "resume"},
                    {"document_type": "job_description"},
                    {"document_type": "other"},
                ]
            }
        )
    if sources:
        filters.append({"source": {"$in": sources}})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def source_paths(documents: list[Document]) -> list[str]:
    """Return unique document source paths in stable alphabetical order."""
    return sorted({str(document.metadata.get("source", "unknown")) for document in documents})
