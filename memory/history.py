"""Translate persisted memory records into prompt-ready chat messages."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from memory.sqlite_memory import SQLiteMemory


def get_chat_history(memory: SQLiteMemory, session: str) -> list[BaseMessage]:
    """Convert a session's stored messages into LangChain chat-message objects."""
    return [
        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        for role, content in memory.recent(session)
    ]
