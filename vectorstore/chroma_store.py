"""Chroma persistence adapter."""

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import Settings
from embeddings.embedding import create_embeddings


def get_vectorstore(settings: Settings) -> Chroma:
    """Open the configured persistent Chroma collection."""
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=create_embeddings(settings),
    )


def rebuild_vectorstore(settings: Settings, chunks: list[Document]) -> Chroma:
    """Replace the persisted index with ``chunks`` and return the new collection."""
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    vectorstore = get_vectorstore(settings)
    vectorstore.add_documents(chunks)
    return vectorstore


def has_index(vectorstore: Chroma) -> bool:
    """Report whether a Chroma collection contains at least one indexed chunk."""
    return vectorstore._collection.count() > 0
