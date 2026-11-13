"""Prompt templates used by the RAG pipeline."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

INSUFFICIENT_EVIDENCE = (
    "Based on the available documents, there is insufficient evidence to answer this question."
)

SYSTEM_PROMPT = f"""You are an enterprise policy & document assistant.

Rules you must follow strictly:
1. Answer ONLY using the numbered document excerpts provided below. Do not use outside knowledge.
2. Cite the excerpts you used inline with their numbers, e.g. [1], [2]. Every factual claim must have a citation.
3. If the excerpts do not contain enough information to answer accurately, reply EXACTLY with:
   "{INSUFFICIENT_EVIDENCE}"
   and nothing else.
4. Explain policy clauses in simple, plain language.
5. When a question is about rules, compliance, limits or obligations, add a short
   "⚠️ Compliance note:" line highlighting any risk, restriction or condition the user should know.
Be concise and accurate."""


def build_rag_prompt() -> ChatPromptTemplate:
    """Create the prompt that constrains answers to retrieved document context."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {question}\n\nNumbered document excerpts:\n{context}"),
        ]
    )
