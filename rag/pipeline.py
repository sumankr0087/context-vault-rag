"""Synchronous retrieval-augmented generation orchestration."""

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage

from config import Settings
from llm.ollama_client import create_chat_model
from llm.prompts import build_rag_prompt
from memory.history import get_chat_history
from memory.sqlite_memory import SQLiteMemory
from rag.context import final_answer_text, format_document_context
from rag.streaming import stream_response
from utils.metadata import source_paths
from vectorstore.retrieval import retrieve_documents


@dataclass(frozen=True)
class Answer:
    """Carry a generated answer together with the sources used to produce it."""

    text: str
    sources: list[str]


@dataclass(frozen=True)
class StreamingAnswer:
    """Carry a token iterator and its already-resolved document sources."""

    tokens: Iterator[str]
    sources: list[str]


class RagPipeline:
    """Coordinate retrieval, conversational memory, prompt creation, and generation."""

    def __init__(self, settings: Settings, memory: SQLiteMemory) -> None:
        """Compose the collaborators required to answer questions consistently."""
        self._settings = settings
        self._memory = memory
        self._prompt = build_rag_prompt()
        self._chat_model = create_chat_model(settings)

    def answer(
        self,
        vectorstore: Chroma,
        session: str,
        question: str,
        metadata_filter: dict[str, str] | None = None,
        mmr_lambda_mult: float | None = None,
        hybrid: bool = False,
        rerank: bool = False,
    ) -> Answer:
        """Retrieve filtered, diverse context and generate an answer for one question."""
        prompt_input = self._build_prompt_input(
            vectorstore,
            session,
            question,
            metadata_filter,
            mmr_lambda_mult,
            hybrid,
            rerank,
        )
        if prompt_input is None:
            return Answer("I could not find relevant information in your documents.", [])
        messages, sources = prompt_input
        response = self._chat_model.invoke(messages)
        return Answer(final_answer_text(response.content), sources)

    def stream_answer(
        self,
        vectorstore: Chroma,
        session: str,
        question: str,
        metadata_filter: dict[str, str] | None = None,
        mmr_lambda_mult: float | None = None,
        hybrid: bool = False,
        rerank: bool = False,
    ) -> StreamingAnswer:
        """Retrieve context and return a token stream for one session question."""
        prompt_input = self._build_prompt_input(
            vectorstore,
            session,
            question,
            metadata_filter,
            mmr_lambda_mult,
            hybrid,
            rerank,
        )
        if prompt_input is None:
            return StreamingAnswer(
                iter(["I could not find relevant information in your documents."]),
                [],
            )
        messages, sources = prompt_input
        return StreamingAnswer(stream_response(self._chat_model, messages), sources)

    def _build_prompt_input(
        self,
        vectorstore: Chroma,
        session: str,
        question: str,
        metadata_filter: dict[str, str] | None,
        mmr_lambda_mult: float | None,
        hybrid: bool,
        rerank: bool,
    ) -> tuple[list[BaseMessage], list[str]] | None:
        """Retrieve documents and create the prompt shared by normal and streamed answers."""
        lambda_mult = self._settings.mmr_lambda_mult if mmr_lambda_mult is None else mmr_lambda_mult
        documents = retrieve_documents(
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
        )
        if not documents:
            return None

        messages = self._prompt.format_messages(
            chat_history=get_chat_history(self._memory, session),
            question=question,
            context=format_document_context(documents),
        )
        return messages, source_paths(documents)
