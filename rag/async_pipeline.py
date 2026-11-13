"""Asynchronous retrieval-augmented generation orchestration."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage

from config import Settings
from llm.ollama_client import create_chat_model
from llm.prompts import build_rag_prompt
from memory.history import get_chat_history
from memory.sqlite_memory import SQLiteMemory
from rag.context import build_citations, final_answer_text, format_document_context
from rag.pipeline import Answer
from rag.streaming import async_stream_response
from utils.metadata import source_paths
from vectorstore.retrieval import retrieve_documents


@dataclass(frozen=True)
class AsyncStreamingAnswer:
    """Carry an asynchronous token iterator and its resolved document sources."""

    tokens: AsyncIterator[str]
    sources: list[str]
    citations: list[dict] = field(default_factory=list)


class AsyncRagPipeline:
    """Coordinate concurrent context preparation with asynchronous LLM generation."""

    def __init__(self, settings: Settings, memory: SQLiteMemory) -> None:
        """Compose the collaborators required for the asynchronous RAG use case."""
        self._settings = settings
        self._memory = memory
        self._prompt = build_rag_prompt()
        self._chat_model = create_chat_model(settings)

    async def answer(
        self,
        vectorstore: Chroma,
        session: str,
        question: str,
        metadata_filter: dict[str, str] | None = None,
        mmr_lambda_mult: float | None = None,
        hybrid: bool = False,
        rerank: bool = False,
    ) -> Answer:
        """Concurrently prepare context and asynchronously generate one answer."""
        prompt_input = await self._build_prompt_input(
            vectorstore,
            session,
            question,
            metadata_filter,
            mmr_lambda_mult,
            hybrid,
            rerank,
        )
        if prompt_input is None:
            return Answer("Based on the available documents, there is insufficient evidence to answer this question.", [])
        messages, sources, citations = prompt_input
        response = await self._chat_model.ainvoke(messages)
        return Answer(final_answer_text(response.content), sources, citations)

    async def stream_answer(
        self,
        vectorstore: Chroma,
        session: str,
        question: str,
        metadata_filter: dict[str, str] | None = None,
        mmr_lambda_mult: float | None = None,
        hybrid: bool = False,
        rerank: bool = False,
    ) -> AsyncStreamingAnswer:
        """Prepare context concurrently and return an asynchronous answer-token stream."""
        prompt_input = await self._build_prompt_input(
            vectorstore,
            session,
            question,
            metadata_filter,
            mmr_lambda_mult,
            hybrid,
            rerank,
        )
        if prompt_input is None:
            return AsyncStreamingAnswer(self._no_documents_stream(), [], [])
        messages, sources, citations = prompt_input
        return AsyncStreamingAnswer(async_stream_response(self._chat_model, messages), sources, citations)

    async def _build_prompt_input(
        self,
        vectorstore: Chroma,
        session: str,
        question: str,
        metadata_filter: dict[str, str] | None,
        mmr_lambda_mult: float | None,
        hybrid: bool,
        rerank: bool,
    ) -> tuple[list[BaseMessage], list[str], list[dict]] | None:
        """Retrieve documents and history concurrently, then build a shared prompt."""
        lambda_mult = self._settings.mmr_lambda_mult if mmr_lambda_mult is None else mmr_lambda_mult
        documents, chat_history = await asyncio.gather(
            asyncio.to_thread(
                retrieve_documents,
                vectorstore,
                question,
                self._settings.retrieval_k,
                metadata_filter,
                self._settings.mmr_fetch_k,
                lambda_mult,
                hybrid,
                rerank,
                self._settings.rerank_fetch_k,
                self._settings.rerank_model,
            ),
            asyncio.to_thread(get_chat_history, self._memory, session),
        )
        if not documents:
            return None
        messages = self._prompt.format_messages(
            chat_history=chat_history,
            question=question,
            context=format_document_context(documents),
        )
        return messages, source_paths(documents), build_citations(documents)

    async def _no_documents_stream(self) -> AsyncIterator[str]:
        """Provide a stream-shaped response when retrieval produces no context."""
        yield "Based on the available documents, there is insufficient evidence to answer this question."
