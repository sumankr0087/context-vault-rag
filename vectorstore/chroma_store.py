"""Lightweight in-memory vector store.

Replaces ChromaDB (which pulls in grpc, duckdb, opentelemetry, a kubernetes
client and a second onnxruntime) so the service fits in a 512MB instance.
It implements the small subset of the previous Chroma interface the app uses:
``add_documents``, ``get``, ``delete`` and ``max_marginal_relevance_search``.

Storage is in-process; on free hosting the filesystem is ephemeral anyway, so
documents live for the lifetime of the running instance (re-upload after a
restart), exactly as before.
"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document

from config import Settings
from embeddings.embedding import create_embeddings


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class SimpleVectorStore:
    """A minimal dense vector store with cosine similarity and MMR search."""

    def __init__(self, settings: Settings) -> None:
        self._embeddings = create_embeddings(settings)
        self._vectors: list[np.ndarray] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        # Load the embedding model now (at startup) so the one-time memory spike
        # happens at boot, not during the first upload request.
        try:
            self._embeddings.embed_query("warmup")
        except Exception:
            pass

    # ---- writes ----
    def add_documents(self, documents: list[Document]) -> None:
        texts = [doc.page_content for doc in documents]
        if not texts:
            return
        embeddings = self._embeddings.embed_documents(texts)
        for doc, embedding in zip(texts, embeddings, strict=True):
            self._vectors.append(_normalize(np.asarray(embedding, dtype=np.float32)))
        for document in documents:
            self._texts.append(document.page_content)
            self._metadatas.append(dict(document.metadata or {}))

    def delete(self, where: dict | None = None) -> None:
        kept = [
            (vector, text, metadata)
            for vector, text, metadata in zip(self._vectors, self._texts, self._metadatas, strict=True)
            if not self._matches(metadata, where)
        ]
        self._vectors = [item[0] for item in kept]
        self._texts = [item[1] for item in kept]
        self._metadatas = [item[2] for item in kept]

    # ---- reads ----
    def get(self, where: dict | None = None, include: list[str] | None = None) -> dict:
        documents: list[str] = []
        metadatas: list[dict] = []
        for text, metadata in zip(self._texts, self._metadatas, strict=True):
            if self._matches(metadata, where):
                documents.append(text)
                metadatas.append(metadata)
        return {"documents": documents, "metadatas": metadatas}

    def count(self) -> int:
        return len(self._texts)

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[Document]:
        indexes, scores = self._scored_indexes(query, filter)
        if not indexes:
            return []
        order = np.argsort(-scores)[:k]
        return [self._document(indexes[int(pos)]) for pos in order]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
    ) -> list[Document]:
        indexes, scores = self._scored_indexes(query, filter)
        if not indexes:
            return []
        top = np.argsort(-scores)[: max(fetch_k, k)]
        candidates = [(int(indexes[int(pos)]), float(scores[int(pos)])) for pos in top]
        selected: list[tuple[int, float]] = []
        while candidates and len(selected) < k:
            best_pos, best_score = 0, -1e18
            for pos, (store_index, similarity) in enumerate(candidates):
                diversity = 0.0
                if selected:
                    diversity = max(
                        float(self._vectors[store_index] @ self._vectors[chosen])
                        for chosen, _ in selected
                    )
                score = lambda_mult * similarity - (1 - lambda_mult) * diversity
                if score > best_score:
                    best_score, best_pos = score, pos
            selected.append(candidates.pop(best_pos))
        return [self._document(store_index) for store_index, _ in selected]

    # ---- helpers ----
    def _document(self, index: int) -> Document:
        return Document(page_content=self._texts[index], metadata=self._metadatas[index])

    def _scored_indexes(self, query: str, where: dict | None):
        indexes = [i for i, metadata in enumerate(self._metadatas) if self._matches(metadata, where)]
        if not indexes:
            return [], np.array([])
        query_vector = _normalize(np.asarray(self._embeddings.embed_query(query), dtype=np.float32))
        matrix = np.vstack([self._vectors[i] for i in indexes])
        return indexes, matrix @ query_vector

    @staticmethod
    def _matches(metadata: dict, where: dict | None) -> bool:
        if not where:
            return True
        return all(str(metadata.get(key)) == str(value) for key, value in where.items())


def get_vectorstore(settings: Settings) -> SimpleVectorStore:
    """Create the in-memory vector store used for indexing and retrieval."""
    return SimpleVectorStore(settings)


def rebuild_vectorstore(settings: Settings, chunks: list[Document]) -> SimpleVectorStore:
    """Create a fresh store populated with ``chunks``."""
    vectorstore = get_vectorstore(settings)
    vectorstore.add_documents(chunks)
    return vectorstore


def has_index(vectorstore: SimpleVectorStore) -> bool:
    """Report whether the store contains at least one indexed chunk."""
    return vectorstore.count() > 0
