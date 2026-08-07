"""Document chunking policy."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings


def chunk_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Split documents into overlapping chunks suitable for semantic retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
