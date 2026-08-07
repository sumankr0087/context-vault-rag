"""Tests for terminal-oriented synchronous response streaming."""

import unittest
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from rag.streaming import async_stream_response, stream_response


class FakeChatModel:
    """Provide deterministic response chunks without calling Ollama."""

    def __init__(self, chunks: list[str]) -> None:
        """Store the chunks that the fake model should stream."""
        self._chunks = chunks

    def stream(self, _messages: list[HumanMessage]):
        """Yield objects shaped like LangChain response chunks."""
        for chunk in self._chunks:
            yield SimpleNamespace(content=chunk)


class FakeAsyncChatModel:
    """Provide deterministic asynchronous response chunks without Ollama."""

    def __init__(self, chunks: list[str]) -> None:
        """Store the chunks that the fake model should stream."""
        self._chunks = chunks

    async def astream(self, _messages: list[HumanMessage]):
        """Yield objects shaped like asynchronous LangChain response chunks."""
        for chunk in self._chunks:
            yield SimpleNamespace(content=chunk)


class StreamingTests(unittest.TestCase):
    """Verify visible content is yielded incrementally and empty output is handled."""

    def test_stream_response_yields_text_chunks_in_order(self) -> None:
        """The terminal can print response text as each model chunk arrives."""
        model = FakeChatModel(["RAG", " retrieves", " context."])

        tokens = list(stream_response(model, [HumanMessage(content="Explain RAG")]))  # type: ignore[arg-type]

        self.assertEqual(tokens, ["RAG", " retrieves", " context."])

    def test_stream_response_replaces_an_empty_model_response(self) -> None:
        """A model that emits no visible text cannot create a blank CLI answer."""
        model = FakeChatModel([])

        tokens = list(stream_response(model, [HumanMessage(content="Explain RAG")]))  # type: ignore[arg-type]

        self.assertEqual(tokens, ["The language model returned an empty answer. Please ask again."])

    def test_async_stream_response_yields_text_chunks_in_order(self) -> None:
        """Async streaming preserves content and order without a model server."""
        async def collect_tokens() -> list[str]:
            model = FakeAsyncChatModel(["RAG", " streams", " tokens."])
            return [
                token
                async for token in async_stream_response(
                    model,  # type: ignore[arg-type]
                    [HumanMessage(content="Explain RAG")],
                )
            ]

        import asyncio

        self.assertEqual(asyncio.run(collect_tokens()), ["RAG", " streams", " tokens."])
