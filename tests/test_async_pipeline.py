"""Tests for asynchronous RAG orchestration."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from config import get_settings
from memory.sqlite_memory import SQLiteMemory
from rag.async_pipeline import AsyncRagPipeline


class FakeAsyncChatModel:
    """Return a deterministic async response without contacting Ollama."""

    async def ainvoke(self, _messages: object) -> SimpleNamespace:
        """Provide a response shaped like LangChain's chat-model result."""
        return SimpleNamespace(content="Python is a programming language.")

    async def astream(self, _messages: object):
        """Yield a deterministic answer as async response chunks."""
        yield SimpleNamespace(content="Python")
        yield SimpleNamespace(content=" is a programming language.")


class AsyncRagPipelineTests(unittest.TestCase):
    """Verify asynchronous retrieval, history loading, and generation compose correctly."""

    def test_answer_retrieves_and_generates_with_ainvoke(self) -> None:
        """The async pipeline returns a grounded answer and its source path."""
        settings = get_settings()
        with tempfile.TemporaryDirectory() as temporary_directory:
            memory = SQLiteMemory(Path(temporary_directory) / "memory.db", settings.memory_turns)
            memory.initialize()
            vectorstore = Mock()
            vectorstore.max_marginal_relevance_search.return_value = [
                Document(page_content="Python is used for programming.", metadata={"source": "docs/python.pdf"})
            ]
            pipeline = AsyncRagPipeline(settings, memory)
            pipeline._chat_model = FakeAsyncChatModel()  # type: ignore[assignment]

            answer = asyncio.run(pipeline.answer(vectorstore, "test", "What is Python?"))

        self.assertEqual(answer.text, "Python is a programming language.")
        self.assertEqual(answer.sources, ["docs/python.pdf"])
        vectorstore.max_marginal_relevance_search.assert_called_once()

    def test_stream_answer_returns_async_tokens_and_sources(self) -> None:
        """The async pipeline preserves source attribution while streaming."""
        settings = get_settings()
        with tempfile.TemporaryDirectory() as temporary_directory:
            memory = SQLiteMemory(Path(temporary_directory) / "memory.db", settings.memory_turns)
            memory.initialize()
            vectorstore = Mock()
            vectorstore.max_marginal_relevance_search.return_value = [
                Document(page_content="Python is used for programming.", metadata={"source": "docs/python.pdf"})
            ]
            pipeline = AsyncRagPipeline(settings, memory)
            pipeline._chat_model = FakeAsyncChatModel()  # type: ignore[assignment]

            async def collect_stream() -> tuple[list[str], list[str]]:
                streamed_answer = await pipeline.stream_answer(vectorstore, "test", "What is Python?")
                return [token async for token in streamed_answer.tokens], streamed_answer.sources

            tokens, sources = asyncio.run(collect_stream())

        self.assertEqual(tokens, ["Python", " is a programming language."])
        self.assertEqual(sources, ["docs/python.pdf"])
        vectorstore.max_marginal_relevance_search.assert_called_once()
