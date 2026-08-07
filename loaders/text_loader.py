"""Plain-text and Markdown document loading adapter."""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


class TextDocumentLoader:
    """Load UTF-8-compatible text files, detecting another encoding when needed."""

    def load(self, path: Path) -> list[Document]:
        """Load the textual content of ``path`` as LangChain documents."""
        return TextLoader(str(path), autodetect_encoding=True).load()
