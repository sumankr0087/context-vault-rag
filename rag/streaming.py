"""Synchronous and asynchronous token streaming for terminal responses."""

from collections.abc import AsyncIterator, Iterator

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama


def stream_response(chat_model: ChatOllama, messages: list[BaseMessage]) -> Iterator[str]:
    """Yield visible answer text from a chat model as each response chunk arrives."""
    emitted_text = False
    for chunk in chat_model.stream(messages):
        if isinstance(chunk.content, str) and chunk.content:
            emitted_text = True
            yield chunk.content
    if not emitted_text:
        yield "The language model returned an empty answer. Please ask again."


async def async_stream_response(chat_model: ChatOllama, messages: list[BaseMessage]) -> AsyncIterator[str]:
    """Yield visible answer text as asynchronous response chunks arrive."""
    emitted_text = False
    async for chunk in chat_model.astream(messages):
        if isinstance(chunk.content, str) and chunk.content:
            emitted_text = True
            yield chunk.content
    if not emitted_text:
        yield "The language model returned an empty answer. Please ask again."
