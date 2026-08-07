"""Select the appropriate loader without leaking file-format rules to callers."""

from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from loaders.pdf_loader import PdfDocumentLoader
from loaders.text_loader import TextDocumentLoader
from utils.metadata import add_document_metadata


class DocumentLoader(Protocol):
    """Define the behavior every document-format adapter must provide."""

    def load(self, path: Path) -> list[Document]:
        """Load documents represented by one local file."""


class LoaderFactory:
    """Map supported filename extensions to document-loading adapters."""

    def __init__(self) -> None:
        """Register the formats supported by the first milestone."""
        text_loader = TextDocumentLoader()
        self._loaders: dict[str, DocumentLoader] = {
            ".pdf": PdfDocumentLoader(),
            ".txt": text_loader,
            ".md": text_loader,
        }

    def get_loader(self, path: Path) -> DocumentLoader | None:
        """Return a loader for ``path``, or ``None`` when it is unsupported."""
        return self._loaders.get(path.suffix.lower())


def load_documents(documents_dir: Path, factory: LoaderFactory | None = None) -> list[Document]:
    """Load all supported files below ``documents_dir`` and attach their source path."""
    loader_factory = factory or LoaderFactory()
    documents: list[Document] = []

    for path in documents_dir.rglob("*"):
        if not path.is_file():
            continue
        loader = loader_factory.get_loader(path)
        if loader is None:
            continue
        loaded_documents = loader.load(path)
        for document in loaded_documents:
            add_document_metadata(document, path)
        documents.extend(loaded_documents)

    return documents
