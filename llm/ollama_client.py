"""Ollama chat-model construction."""

from config import Settings
from langchain_ollama import ChatOllama


def create_chat_model(settings: Settings) -> ChatOllama:
    """Create the locally hosted chat model used to generate final answers."""
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_url,
        temperature=0.2,
        num_ctx=8192,
        reasoning=False,
    )
