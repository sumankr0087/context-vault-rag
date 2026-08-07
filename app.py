"""Command-line entry point for the local RAG assistant."""

import argparse
import asyncio
from pathlib import Path

from config import get_settings
from evaluation.retrieval import evaluate_retrieval, format_evaluation_report, load_evaluation_cases
from loaders.loader_factory import load_documents
from memory.sqlite_memory import SQLiteMemory
from rag.async_pipeline import AsyncRagPipeline
from rag.pipeline import RagPipeline
from utils.chunking import chunk_documents
from utils.metadata import build_extension_filter
from vectorstore.chroma_store import get_vectorstore, has_index, rebuild_vectorstore


def parse_args() -> argparse.Namespace:
    """Parse CLI options without coupling the rest of the application to argparse."""
    parser = argparse.ArgumentParser(description="Chat with local PDF and text documents.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the document index from docs/.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Measure retrieval against the local evaluation question set and exit.",
    )
    parser.add_argument(
        "--evaluation-file",
        type=Path,
        default=Path("eval/questions.json"),
        help="JSON file containing questions and their expected source paths.",
    )
    parser.add_argument("--session", default="default", help="Conversation name to remember and resume.")
    parser.add_argument(
        "--extension",
        help="Only retrieve one file type, for example .pdf, .md, or .txt.",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        help="MMR relevance/diversity balance from 0.0 to 1.0; default is 0.5.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Fuse dense MMR retrieval with local BM25 keyword retrieval.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use a local cross-encoder to rerank a wider set of retrieved chunks.",
    )
    response_mode = parser.add_mutually_exclusive_group()
    response_mode.add_argument(
        "--stream",
        action="store_true",
        help="Print the answer as response chunks arrive from Ollama.",
    )
    response_mode.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Prepare context concurrently and generate through async Ollama calls.",
    )
    response_mode.add_argument(
        "--async-stream",
        action="store_true",
        help="Prepare context concurrently and print async Ollama response chunks as they arrive.",
    )
    return parser.parse_args()


def reindex() -> bool:
    """Load, chunk, and persist all supported documents; return whether indexing occurred."""
    settings = get_settings()
    documents = load_documents(settings.docs_dir)
    if not documents:
        print("No supported documents found in docs/. Add PDF, TXT, or Markdown files.")
        return False
    chunks = chunk_documents(documents, settings)
    rebuild_vectorstore(settings, chunks)
    print(f"Indexed {len(documents)} document pages/files into {len(chunks)} chunks.")
    return True


def main() -> None:
    """Run the CLI's indexing mode or its interactive question-answering loop."""
    args = parse_args()
    if args.mmr_lambda is not None and not 0.0 <= args.mmr_lambda <= 1.0:
        raise SystemExit("--mmr-lambda must be between 0.0 and 1.0.")
    settings = get_settings()
    settings.docs_dir.mkdir(exist_ok=True)
    settings.data_dir.mkdir(exist_ok=True)
    memory = SQLiteMemory(settings.memory_db, settings.memory_turns)
    memory.initialize()

    if args.reindex:
        if not reindex():
            return
        vectorstore = get_vectorstore(settings)
    else:
        vectorstore = get_vectorstore(settings)
        if not has_index(vectorstore):
            print("No document index exists yet.")
            print("Add files to docs/ and run: python app.py --reindex")
            return

    if args.evaluate:
        cases = load_evaluation_cases(args.evaluation_file)
        report = evaluate_retrieval(
            vectorstore,
            cases,
            settings,
            build_extension_filter(args.extension),
            args.mmr_lambda,
            args.hybrid,
            args.rerank,
        )
        print(format_evaluation_report(report))
        return

    pipeline = RagPipeline(settings, memory)
    async_pipeline = AsyncRagPipeline(settings, memory) if args.async_mode or args.async_stream else None
    metadata_filter = build_extension_filter(args.extension)
    print(f"\nReady. Session: {args.session}")
    if metadata_filter:
        print(f"Retrieval filter: {metadata_filter}")
    lambda_mult = args.mmr_lambda if args.mmr_lambda is not None else settings.mmr_lambda_mult
    print(f"MMR: top {settings.retrieval_k} from {settings.mmr_fetch_k} candidates (lambda={lambda_mult})")
    print(f"Retrieval: {'hybrid MMR + BM25' if args.hybrid else 'dense MMR'}")
    if args.rerank:
        print(f"Reranking: top {settings.retrieval_k} from {settings.rerank_fetch_k} candidates")
    if args.stream:
        print("Response mode: streaming")
    if args.async_mode:
        print("Response mode: asynchronous")
    if args.async_stream:
        print("Response mode: asynchronous streaming")
    print("Ask questions. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            return
        if not question:
            continue
        memory.save(args.session, "user", question)
        if args.stream:
            streamed_answer = pipeline.stream_answer(
                vectorstore,
                args.session,
                question,
                metadata_filter,
                args.mmr_lambda,
                args.hybrid,
                args.rerank,
            )
            print("\nAssistant: ", end="", flush=True)
            answer_parts: list[str] = []
            for token in streamed_answer.tokens:
                print(token, end="", flush=True)
                answer_parts.append(token)
            answer_text = "".join(answer_parts)
            print()
            sources = streamed_answer.sources
        elif args.async_stream:
            if async_pipeline is None:
                raise RuntimeError("Async pipeline was not initialized.")

            async def consume_async_stream() -> tuple[str, list[str]]:
                """Print one async answer stream and return its completed text and sources."""
                streamed_answer = await async_pipeline.stream_answer(
                    vectorstore,
                    args.session,
                    question,
                    metadata_filter,
                    args.mmr_lambda,
                    args.hybrid,
                    args.rerank,
                )
                print("\nAssistant: ", end="", flush=True)
                answer_parts: list[str] = []
                async for token in streamed_answer.tokens:
                    print(token, end="", flush=True)
                    answer_parts.append(token)
                print()
                return "".join(answer_parts), streamed_answer.sources

            answer_text, sources = asyncio.run(consume_async_stream())
        elif args.async_mode:
            if async_pipeline is None:
                raise RuntimeError("Async pipeline was not initialized.")
            answer = asyncio.run(
                async_pipeline.answer(
                    vectorstore,
                    args.session,
                    question,
                    metadata_filter,
                    args.mmr_lambda,
                    args.hybrid,
                    args.rerank,
                )
            )
            answer_text = answer.text
            sources = answer.sources
            print(f"\nAssistant: {answer_text}")
        else:
            answer = pipeline.answer(
                vectorstore,
                args.session,
                question,
                metadata_filter,
                args.mmr_lambda,
                args.hybrid,
                args.rerank,
            )
            answer_text = answer.text
            sources = answer.sources
            print(f"\nAssistant: {answer_text}")
        memory.save(args.session, "assistant", answer_text)
        print(f"Sources: {', '.join(sources)}\n")


if __name__ == "__main__":
    main()
