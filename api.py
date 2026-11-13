"""FastAPI service exposing complete and SSE-streaming RAG responses."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import AsyncIterator, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import Settings, get_settings
from loaders.loader_factory import LoaderFactory
from memory.sqlite_memory import SQLiteMemory
from rag.async_pipeline import AsyncRagPipeline
from utils.chunking import chunk_documents
from utils.metadata import add_document_metadata, build_retrieval_filter
from vectorstore.chroma_store import get_vectorstore, has_index


class ChatRequest(BaseModel):
    """Validate one API chat request and its retrieval options."""

    session: str = Field(default="default", min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=10_000)
    extension: str | None = None
    document_type: Literal["resume", "job_description", "other"] | None = None
    document_ids: list[str] = Field(default_factory=list)
    uploaded_only: bool = False
    mmr_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    hybrid: bool = False
    rerank: bool = False


class ChatResponse(BaseModel):
    """Return one complete answer, its sources, and clause-level citations."""

    answer: str
    sources: list[str]
    citations: list[dict] = Field(default_factory=list)


class CompareRequest(BaseModel):
    """Validate a request to compare two uploaded document versions."""

    document_id_a: str = Field(min_length=1)
    document_id_b: str = Field(min_length=1)
    topic: str | None = Field(default=None, max_length=500)


class CompareResponse(BaseModel):
    """Return a plain-language diff of two document versions."""

    report: str
    version_a: str
    version_b: str


class UploadResponse(BaseModel):
    """Describe a completed document upload and indexing operation."""

    uploaded_files: list[str]
    skipped_files: list[str]
    indexed_chunks: int
    documents: list["UploadedDocument"]


class UploadedDocument(BaseModel):
    """Expose an uploaded file through an opaque identifier and safe display name."""

    id: str
    name: str


class DeleteDocumentResponse(BaseModel):
    """Describe one uploaded document removed from local storage and the index."""

    id: str
    name: str


@dataclass(frozen=True)
class ApiServices:
    """Hold the process-wide collaborators created during application startup."""

    settings: Settings
    memory: SQLiteMemory
    vectorstore: object
    pipeline: AsyncRagPipeline
    indexed: bool


def create_app() -> FastAPI:
    """Create a FastAPI application without performing model calls at import time."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        settings.docs_dir.mkdir(exist_ok=True)
        settings.data_dir.mkdir(exist_ok=True)
        memory = SQLiteMemory(settings.memory_db, settings.memory_turns)
        memory.initialize()
        vectorstore = get_vectorstore(settings)
        application.state.services = ApiServices(
            settings=settings,
            memory=memory,
            vectorstore=vectorstore,
            pipeline=AsyncRagPipeline(settings, memory),
            indexed=has_index(vectorstore),
        )
        yield

    application = FastAPI(
        title="ContextVault API",
        version="1.0.0",
        description="RAG chat with citations, guardrails, version comparison, and SSE streaming.",
        lifespan=lifespan,
    )

    # Allow the static frontend (GitHub Pages / localhost) to call the API.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health")
    async def health(request: Request) -> dict[str, str]:
        """Report whether the application has a local document index ready to search."""
        services: ApiServices = request.app.state.services
        return {"status": "ready" if services.indexed else "index_missing"}

    @application.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        """Return one complete answer after asynchronous context preparation."""
        services = _ready_services(request)
        await asyncio.to_thread(services.memory.save, payload.session, "user", payload.question)
        answer = await services.pipeline.answer(
            services.vectorstore,
            payload.session,
            payload.question,
            _chat_retrieval_filter(payload, services),
            payload.mmr_lambda,
            payload.hybrid,
            payload.rerank,
        )
        await asyncio.to_thread(services.memory.save, payload.session, "assistant", answer.text)
        return ChatResponse(answer=answer.text, sources=answer.sources, citations=answer.citations)

    @application.post("/chat/stream")
    async def stream_chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        """Stream answer tokens as SSE events, then emit sources and completion state."""
        services = _ready_services(request)

        async def events() -> AsyncIterator[str]:
            try:
                await asyncio.to_thread(services.memory.save, payload.session, "user", payload.question)
                answer = await services.pipeline.stream_answer(
                    services.vectorstore,
                    payload.session,
                    payload.question,
                    _chat_retrieval_filter(payload, services),
                    payload.mmr_lambda,
                    payload.hybrid,
                    payload.rerank,
                )
                answer_parts: list[str] = []
                async for token in answer.tokens:
                    answer_parts.append(token)
                    yield _sse_event("token", {"text": token})
                answer_text = "".join(answer_parts)
                await asyncio.to_thread(services.memory.save, payload.session, "assistant", answer_text)
                yield _sse_event("sources", {"sources": answer.sources, "citations": answer.citations})
                yield _sse_event("done", {})
            except Exception:
                yield _sse_event("error", {"message": "Unable to generate an answer. Please try again."})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.post("/compare", response_model=CompareResponse)
    async def compare(payload: CompareRequest, request: Request) -> CompareResponse:
        """Compare two uploaded document versions and explain what changed."""
        from rag.compare import compare_documents

        services = _ready_services(request)
        source_by_id = _uploaded_sources_by_document_id(services.vectorstore)
        name_by_id = {d.id: d.name for d in _list_uploaded_documents(services.vectorstore)}
        for doc_id in (payload.document_id_a, payload.document_id_b):
            if doc_id not in source_by_id:
                raise HTTPException(status_code=400, detail="One or more selected documents are unavailable.")
        result = await asyncio.to_thread(
            compare_documents,
            services.vectorstore,
            services.settings,
            source_by_id[payload.document_id_a],
            name_by_id.get(payload.document_id_a, "Version A"),
            source_by_id[payload.document_id_b],
            name_by_id.get(payload.document_id_b, "Version B"),
            payload.topic,
        )
        return CompareResponse(report=result.report, version_a=result.source_a, version_b=result.source_b)

    @application.get("/documents", response_model=list[UploadedDocument])
    async def list_documents(request: Request) -> list[UploadedDocument]:
        """List documents available for precise retrieval in the web interface."""
        services: ApiServices = request.app.state.services
        return _list_uploaded_documents(services.vectorstore)

    @application.post("/documents", response_model=UploadResponse, status_code=201)
    async def upload_documents(
        request: Request,
        files: list[UploadFile] = File(...),
        document_type: Literal["resume", "job_description", "other"] = Form(...),
    ) -> UploadResponse:
        """Store and index supported documents while skipping byte-for-byte duplicates."""
        services: ApiServices = request.app.state.services
        if not files:
            raise HTTPException(status_code=400, detail="Upload at least one document.")
        uploaded_paths: list[Path] = []
        uploaded_documents: list[UploadedDocument] = []
        metadata_by_path: dict[Path, dict[str, str]] = {}
        skipped_files: list[str] = []
        known_hashes = await asyncio.to_thread(_existing_upload_hashes, services.settings.docs_dir / "uploaded")
        for upload in files:
            if not upload.filename:
                raise HTTPException(status_code=400, detail="Each uploaded file needs a filename.")
            filename = Path(upload.filename).name
            if not LoaderFactory().get_loader(Path(filename)):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
            content = await upload.read()
            if len(content) > 20 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File is larger than 20 MB: {filename}")
            content_hash = hashlib.sha256(content).hexdigest()
            if content_hash in known_hashes:
                skipped_files.append(filename)
                continue
            document_id = uuid4().hex
            destination = services.settings.docs_dir / "uploaded" / document_type / f"{document_id}_{filename}"
            await asyncio.to_thread(_write_upload, destination, content)
            uploaded_paths.append(destination)
            uploaded_documents.append(UploadedDocument(id=document_id, name=filename))
            metadata_by_path[destination] = {
                "document_id": document_id,
                "document_name": filename,
                "content_hash": content_hash,
            }
            known_hashes.add(content_hash)

        chunks = await asyncio.to_thread(
            _load_and_chunk_uploads,
            uploaded_paths,
            document_type,
            services.settings,
            metadata_by_path,
        )
        if uploaded_paths and not chunks:
            raise HTTPException(status_code=400, detail="No readable document content was found in the upload.")
        if chunks:
            await asyncio.to_thread(services.vectorstore.add_documents, chunks)
            request.app.state.services = replace(services, indexed=True)
        return UploadResponse(
            uploaded_files=[path.name.split("_", maxsplit=1)[-1] for path in uploaded_paths],
            skipped_files=skipped_files,
            indexed_chunks=len(chunks),
            documents=uploaded_documents,
        )

    @application.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
    async def delete_document(document_id: str, request: Request) -> DeleteDocumentResponse:
        """Remove one uploaded document from both the Chroma index and local disk."""
        services: ApiServices = request.app.state.services
        deleted = await asyncio.to_thread(
            _delete_uploaded_document,
            services.vectorstore,
            services.settings.docs_dir,
            document_id,
        )
        request.app.state.services = replace(services, indexed=has_index(services.vectorstore))
        return deleted

    return application


def _ready_services(request: Request) -> ApiServices:
    """Return initialized services or an actionable error when no index exists."""
    services: ApiServices = request.app.state.services
    if not services.indexed:
        raise HTTPException(status_code=503, detail="No document index exists. Run: python app.py --reindex")
    return services


def _chat_retrieval_filter(payload: ChatRequest, services: ApiServices) -> dict[str, object] | None:
    """Build a filter that enforces the document IDs supplied by the client."""
    sources = _sources_for_document_ids(services.vectorstore, payload.document_ids)
    return build_retrieval_filter(
        payload.extension,
        payload.document_type,
        payload.uploaded_only,
        sources,
    )


def _list_uploaded_documents(vectorstore: object) -> list[UploadedDocument]:
    """Return one safe display record for each uploaded source in the collection."""
    records = vectorstore.get(include=["metadatas"])
    documents: dict[str, UploadedDocument] = {}
    for metadata in records.get("metadatas", []):
        if not metadata:
            continue
        source = str(metadata.get("source", ""))
        if not _is_uploaded_source(source, metadata):
            continue
        document_id = str(metadata.get("document_id") or _document_id_from_source(source))
        document_name = str(metadata.get("document_name") or _document_name_from_source(source))
        documents[document_id] = UploadedDocument(id=document_id, name=document_name)
    return sorted(documents.values(), key=lambda document: document.name.casefold())


def _uploaded_sources_by_document_id(vectorstore: object) -> dict[str, str]:
    """Map each public uploaded-document ID to its internal source path."""
    records = vectorstore.get(include=["metadatas"])
    sources: dict[str, str] = {}
    for metadata in records.get("metadatas", []):
        if not metadata:
            continue
        source = str(metadata.get("source", ""))
        if _is_uploaded_source(source, metadata):
            document_id = str(metadata.get("document_id") or _document_id_from_source(source))
            sources[document_id] = source
    return sources


def _is_uploaded_source(source: str, metadata: dict[str, object]) -> bool:
    """Recognize an API-uploaded source without exposing or trusting its path."""
    return bool(metadata.get("document_id")) or "uploaded" in Path(source).parts


def _sources_for_document_ids(vectorstore: object, document_ids: list[str]) -> list[str] | None:
    """Resolve public document IDs to indexed source paths before querying Chroma."""
    if not document_ids:
        return None
    source_by_id = _uploaded_sources_by_document_id(vectorstore)
    missing_ids = set(document_ids) - source_by_id.keys()
    if missing_ids:
        raise HTTPException(status_code=400, detail="One or more selected documents are unavailable.")
    return [source_by_id[document_id] for document_id in document_ids]


def _delete_uploaded_document(vectorstore: object, docs_dir: Path, document_id: str) -> DeleteDocumentResponse:
    """Delete an uploaded document's chunks and its stored source file."""
    source_by_id = _uploaded_sources_by_document_id(vectorstore)
    source = source_by_id.get(document_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Document is unavailable.")
    document_path = Path(source)
    upload_directory = (docs_dir / "uploaded").resolve()
    resolved_path = document_path.resolve()
    if not resolved_path.is_relative_to(upload_directory):
        raise HTTPException(status_code=500, detail="Document storage path is invalid.")
    vectorstore.delete(where={"source": source})
    resolved_path.unlink(missing_ok=True)
    return DeleteDocumentResponse(id=document_id, name=_document_name_from_source(source))


def _document_id_from_source(source: str) -> str:
    """Recover the opaque UUID used by pre-existing uploaded documents."""
    return Path(source).name.partition("_")[0]


def _document_name_from_source(source: str) -> str:
    """Recover the original filename without exposing its local storage path."""
    filename = Path(source).name
    prefix, separator, original_name = filename.partition("_")
    return original_name if separator and _is_document_id(prefix) else filename


def _is_document_id(value: str) -> bool:
    """Recognize the UUID-like prefix generated for uploaded files."""
    return len(value) == 32 and all(character in "0123456789abcdef" for character in value.lower())


def _existing_upload_hashes(upload_directory: Path) -> set[str]:
    """Hash stored uploads so the same file is never indexed twice."""
    if not upload_directory.exists():
        return set()
    return {_file_hash(path) for path in upload_directory.rglob("*") if path.is_file()}


def _file_hash(path: Path) -> str:
    """Calculate a file digest without loading the entire stored file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sse_event(event: str, data: dict[str, object]) -> str:
    """Format one Server-Sent Event without allowing unescaped JSON payloads."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _write_upload(destination: Path, content: bytes) -> None:
    """Write one validated upload beneath the local document directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _load_and_chunk_uploads(
    paths: list[Path],
    document_type: str,
    settings: Settings,
    metadata_by_path: dict[Path, dict[str, str]] | None = None,
) -> list[object]:
    """Load uploaded files, attach scope metadata, and create retrievable chunks."""
    factory = LoaderFactory()
    documents = []
    for path in paths:
        loader = factory.get_loader(path)
        if loader is None:
            continue
        loaded_documents = loader.load(path)
        for document in loaded_documents:
            add_document_metadata(document, path)
            document.metadata["document_type"] = document_type
            document.metadata.update((metadata_by_path or {}).get(path, {}))
        documents.extend(loaded_documents)
    return chunk_documents(documents, settings)


app = create_app()
