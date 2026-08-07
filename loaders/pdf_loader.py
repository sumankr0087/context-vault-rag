"""PDF document loading adapter."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PdfDocumentLoader:
    """Load textual pages from a PDF using LangChain's PDF adapter."""

    def load(self, path: Path) -> list[Document]:
        """Load every text-bearing page from ``path``."""
        return PyPDFLoader(str(path)).load()
