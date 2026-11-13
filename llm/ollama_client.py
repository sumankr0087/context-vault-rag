"""Chat-model construction — local Ollama or a cloud OpenAI-compatible provider."""

from config import Settings


def create_chat_model(settings: Settings):
    """Create the chat model used to generate final answers.

    - ``ollama``  -> local ChatOllama (private, free, needs a running Ollama).
    - ``groq``/``openai`` -> ChatOpenAI pointed at the provider's base URL
      (lets the app run on free cloud hosting with no local models).
    """
    provider = settings.llm_provider
    if provider in ("groq", "openai"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0.2,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_url,
        temperature=0.2,
        num_ctx=8192,
        reasoning=False,
    )
