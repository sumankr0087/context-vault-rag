"""Prompt templates used by the RAG pipeline."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_rag_prompt() -> ChatPromptTemplate:
    """Create the prompt that constrains answers to retrieved document context."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful personal RAG assistant.
Answer using only the supplied document context when the question is about the
user's documents. If the answer is not in the context, say so clearly. Be
concise and mention source filenames when useful.""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {question}\n\nDocument context:\n{context}"),
        ]
    )
