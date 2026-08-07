# Local RAG Assistant

This repository is a ten-part learning series for building a production-minded local Retrieval-Augmented Generation (RAG) assistant. All **10 milestones** are implemented.

## Milestone 1 — Project refactoring

The working single-file assistant has been separated by responsibility without adding behavior. PDFs, TXT, and Markdown files are loaded; chunks are embedded in Chroma through Ollama; SQLite stores recent chat history; and the CLI answers questions from retrieved context.

### Folder tree

```text
app.py                 CLI composition root
config.py              typed application settings
loaders/               file-format adapters and selection
embeddings/            embedding-client construction
vectorstore/           Chroma persistence and baseline retrieval
memory/                SQLite persistence and prompt-message conversion
llm/                   chat-client construction and prompt template
rag/                   synchronous RAG orchestration
utils/                 chunking and source formatting
tests/                 offline unit tests
docs/                  your source documents
data/                  generated Chroma index and SQLite database
```

### Why these modules exist

- `app.py` is the composition root: it parses CLI input and wires the application together.
- `config.py` keeps environment-dependent choices in one typed place.
- `loaders` isolates external file formats, so another loader can be added without changing indexing.
- `embeddings`, `llm`, and `vectorstore` isolate framework clients from application flow.
- `memory` separates durable records from LangChain-specific message conversion.
- `rag/pipeline.py` owns the use case: retrieve, build context, prompt, and generate.
- `utils` contains small, reusable transformations rather than a catch-all service class.

### Execution flow

```text
docs/ -> LoaderFactory -> documents -> chunker -> Chroma
question + SQLite history -> RagPipeline -> Chroma retrieval -> prompt -> Ollama -> answer
```

`app.py` depends on the pipeline, while the pipeline composes retrieval, memory, prompts, and the LLM. File loaders and Chroma are adapters around third-party libraries. This keeps the central RAG workflow easy to test and makes later retrieval strategies replaceable.

### Run

Start Ollama and make sure these models are available: `qwen3.5:4b` and `qwen3-embedding:0.6b`.

```bash
./venv/bin/python app.py --reindex
./venv/bin/python app.py --session learning
```

Use `OLLAMA_URL` to point at a non-default Ollama server. `EMBEDDING_NUM_CTX` defaults to `1024`, a conservative context window for the local Qwen embedding runtime; increase it only after verifying your Ollama backend supports it. The refactor deliberately preserves the existing model names, chunk settings, top-4 similarity search, prompt intent, and SQLite history size.

### Test

```bash
./venv/bin/python -m unittest discover -s tests -v
./venv/bin/python -m compileall -q . -x 'venv|data'
```

The unit test is offline. End-to-end indexing and chat require a running Ollama server because embeddings and generation remain local model calls.

The chat client sets `reasoning=False`, so Qwen returns its final response directly rather than exposing its internal reasoning path. If a local runtime still returns no final text, the CLI displays a clear retry message instead of a blank answer.

### Concepts

RAG grounds a language model in retrieved local text. A loader produces `Document` objects, chunking breaks long documents into retrievable passages, embeddings map chunks and questions into vectors, and Chroma finds the closest chunks. The prompt supplies those chunks to the chat model. SQLite history provides conversational continuity but is separate from the document knowledge base.

## Milestone 2 — Metadata filtering

Every loaded document now records `source` and `extension` metadata. Chunks retain that metadata when they are stored in Chroma, which allows retrieval to pre-filter candidates before semantic similarity ranking. For example, `filter={"extension": ".pdf"}` prevents TXT and Markdown chunks from being considered.

Rebuild the index once to add the new metadata, then use the optional CLI filter:

```bash
./venv/bin/python app.py --reindex
./venv/bin/python app.py --extension .pdf
```

Use `--extension .md` or `--extension .txt` for the other supported source types. See [MILESTONE_02.md](MILESTONE_02.md) for the full lesson, execution flow, tests, and interview questions.

## Milestone 3 — MMR retrieval

Retrieval now uses Maximal Marginal Relevance (MMR), which starts with the 20 most relevant candidate chunks and selects 4 that balance relevance with diversity. This avoids answering from four nearly identical chunks. The default `lambda=0.5` gives relevance and diversity equal weight.

```bash
./venv/bin/python app.py
./venv/bin/python app.py --extension .pdf --mmr-lambda 0.3
```

Use a higher lambda, such as `0.8`, when exact relevance matters most. Use a lower value, such as `0.3`, to seek broader coverage. See [MILESTONE_03.md](MILESTONE_03.md) for the complete lesson.

## Milestone 4 — Streaming responses

The normal `invoke()` path waits for the complete answer. Use `--stream` to print visible response chunks immediately as Ollama produces them, while preserving the complete answer in SQLite memory after the stream ends.

```bash
./venv/bin/python app.py --stream
./venv/bin/python app.py --stream --extension .pdf --mmr-lambda 0.3
```

Streaming improves perceived responsiveness; it does not make model generation intrinsically faster. See [MILESTONE_04.md](MILESTONE_04.md) for the complete lesson.

## Milestone 5 — Async RAG

Use `--async` to load session history and run MMR retrieval concurrently, then await Ollama through `ainvoke()`. It is an alternative to `--stream`; the CLI keeps those modes separate for clarity.

```bash
./venv/bin/python app.py --async
./venv/bin/python app.py --async --extension .pdf --mmr-lambda 0.3
```

See [MILESTONE_05.md](MILESTONE_05.md) for the full async workflow and theory.

## Milestone 6 — Async streaming responses

Use `--async-stream` to prepare MMR retrieval and chat history concurrently, then print `ChatOllama.astream()` chunks as they arrive. It is mutually exclusive with `--stream` and `--async`, while preserving the completed response in SQLite after the stream finishes.

```bash
./venv/bin/python app.py --async-stream
./venv/bin/python app.py --async-stream --extension .pdf --mmr-lambda 0.3
```

See [MILESTONE_06.md](MILESTONE_06.md) for the full workflow and theory.

## Milestone 7 — Hybrid retrieval

Use `--hybrid` to fuse the existing dense MMR ranking with a local BM25 keyword ranking. It is particularly useful for exact technical terms, command names, and identifiers, while MMR continues to cover semantic intent. Hybrid retrieval works with every response mode and metadata filter; it does not require reindexing.

```bash
./venv/bin/python app.py --hybrid
./venv/bin/python app.py --async-stream --hybrid --extension .pdf
```

See [MILESTONE_07.md](MILESTONE_07.md) for the complete implementation guide.

## Milestone 8 — Cross-encoder reranking

Use `--rerank` to retrieve 12 candidates, score each question–chunk pair with a local cross-encoder, and pass only the strongest four chunks to the LLM. It can be combined with hybrid retrieval and every response mode.

Install the optional reranker once; its model downloads on the first reranked query:

```bash
./venv/bin/pip install sentence-transformers
./venv/bin/python app.py --async-stream --hybrid --rerank
```

See [MILESTONE_08.md](MILESTONE_08.md) for the full workflow and trade-offs.

## Milestone 9 — Retrieval evaluation

Use `--evaluate` to compare retrieved source paths against the question set in `eval/questions.json`. The report shows hit rate@k, mean reciprocal rank (MRR), and per-question evidence. Add questions that represent your actual documents before changing retrieval settings.

```bash
./venv/bin/python app.py --evaluate
./venv/bin/python app.py --evaluate --hybrid
./venv/bin/python app.py --evaluate --hybrid --rerank
```

See [MILESTONE_09.md](MILESTONE_09.md) for the evaluation workflow.

## Milestone 10 — FastAPI and Streamlit UI

The assistant is now available through a FastAPI service with document upload (`POST /documents`), complete JSON (`POST /chat`), and Server-Sent Event (`POST /chat/stream`) endpoints. The Streamlit UI accepts multiple Resume, Job Description, or Other documents, streams its answers, scopes search by category, and intentionally hides backend addresses and source paths.

Install the web dependencies, then use two terminals:

```bash
./venv/bin/pip install -r requirements.txt
./venv/bin/uvicorn api:app --reload
./venv/bin/streamlit run ui.py
```

Open the Streamlit URL printed in the terminal (normally `http://localhost:8501`). See [MILESTONE_10.md](MILESTONE_10.md) for API examples and the full architecture.

### Common mistakes

- Reindex after changing documents; Chroma does not watch files automatically.
- Keep `data/` local: it is generated state, not source code.
- Do not put retrieval logic in `app.py`; it belongs behind `vectorstore/retrieval.py`.
- Rebuild after a metadata-schema change; old Chroma chunks do not gain new fields automatically.
- Use `--extension` only for supported extensions; no results may mean the filter is too narrow.

### Production perspective

The structure applies single responsibility and dependency direction: format adapters, persistence adapters, and model clients do not control the use case. Large-scale teams would additionally use dependency injection containers, observability, background indexing, authentication, evaluation datasets, robust migrations, and managed/vector services. Those are deliberately out of scope for this no-new-features milestone.

### Interview checks

1. Why separate chunking from loaders? Loading translates a file; chunking is a retrieval policy that changes independently.
2. Why store chat memory in SQLite? It is durable, local, transactional, and sufficient for a single-user CLI.
3. Why have a retrieval module now? It gives MMR and hybrid retrieval a stable seam in later milestones.
