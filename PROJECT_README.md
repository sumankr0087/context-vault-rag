# DocuSense — Project Guide

## What this project does

DocuSense is an enterprise document and policy assistant. Upload PDF, TXT, or Markdown files, then ask questions in plain English (for example, *"Can I share client data externally?"*). The assistant retrieves the most relevant passages from your documents and answers **only** from them — with clause-level citations, compliance notes, and a guardrail that refuses to guess when the documents do not contain the answer.

It runs two ways from the same codebase:

- **Local / private** — Ollama for the LLM and embeddings, so your documents never leave your machine.
- **Cloud** — Groq for the LLM and FastEmbed (ONNX) for embeddings, so it deploys to a small free-tier host with no GPU and no local models.

## Live demo

- **App (frontend):** https://sumankr0087.github.io/context-vault-rag/
- **API (backend):** https://context-vault-rag.onrender.com

The demo runs on free tiers, so the backend sleeps when idle — the first request after a pause can take ~50 seconds to wake it. Uploaded documents are temporary (the free tier has ephemeral storage).

## Key features

- **Grounded answers** — the model answers only from your uploaded documents, never from general knowledge.
- **Clause-level citations** — every answer lists the exact excerpts used, each with its source filename, page number, and a snippet.
- **Hallucination guardrail** — if the documents do not cover the question, it replies with a clear "insufficient evidence" message instead of guessing.
- **Compliance notes** — on policy, rule, or obligation questions it adds a "⚠️ Compliance note" flagging limits and risks.
- **Hybrid search** — combines semantic (vector) retrieval with BM25 keyword matching.
- **MMR retrieval** — balances relevance and diversity so answers draw on varied passages.
- **Multi-document scope** — index many documents; skip byte-for-byte duplicate uploads.
- **Version comparison (API)** — the `/compare` endpoint summarizes what changed between two document versions, including compliance impact.
- **Optional cross-encoder reranking** — available for local use (needs PyTorch; disabled in the cloud demo to fit the 512MB free tier).
- **Two interfaces** — a modern responsive web app, plus a command-line interface.

## How it works

```text
Documents -> loaders -> chunks -> embeddings -> Chroma vector store
Question + chat history -> hybrid retrieval (+ optional reranking) -> LLM -> answer + citations
```

1. A loader reads PDF, TXT, or Markdown files.
2. Long text is split into overlapping chunks.
3. An embedding model (FastEmbed in the cloud, Ollama locally) turns each chunk into a vector.
4. Chroma stores and searches those vectors.
5. The best-matching chunks, plus recent chat history from SQLite, are sent to the LLM (Groq in the cloud, Ollama locally) with a strict grounding prompt.
6. The answer is returned with inline `[n]` citations and the list of source excerpts.

## Language and technologies

| Area | Technology |
| --- | --- |
| Main language (backend) | Python |
| Cloud LLM | Groq (`openai/gpt-oss-120b`, OpenAI-compatible API) |
| Cloud embeddings | FastEmbed (ONNX, `BAAI/bge-small-en-v1.5`) |
| Local LLM and embeddings | Ollama (`qwen3.5:4b`, `qwen3-embedding:0.6b`) |
| RAG framework | LangChain |
| Vector database | Chroma |
| Chat memory | SQLite |
| Backend API | FastAPI and Uvicorn |
| Web frontend | Next.js (React), TypeScript, Tailwind CSS |
| PDF reading | PyPDF |
| Keyword retrieval | BM25 |
| Optional reranking | Sentence Transformers cross-encoder (local only) |
| Frontend hosting | GitHub Pages (static export) |
| Backend hosting | Render (Docker) |
| Automated tests | Python `unittest` |

## Project structure

```text
api.py                 FastAPI upload, chat, and compare API
app.py                 Command-line application
ui.py                  Legacy Streamlit interface (optional)
config.py              Provider selection, paths, chunking, retrieval settings
loaders/               PDF, TXT, and Markdown loaders
embeddings/            Embedding client setup (Ollama / HuggingFace / FastEmbed)
vectorstore/           Chroma, MMR, BM25, and reranking retrieval
rag/                   RAG pipelines, prompt context, citations, compare
memory/                SQLite chat memory
llm/                   Chat client (Ollama / Groq / OpenAI) and prompts
utils/                 Chunking and metadata helpers
web/                   Next.js frontend (the deployed web app)
Dockerfile             Container image for the backend
render.yaml            Render Blueprint for one-click backend deploy
.github/workflows/     GitHub Pages deploy workflow for the frontend
docs/                  Source documents and uploaded files
data/                  Generated Chroma index and SQLite database
tests/                 Offline unit tests
```

## Run it locally

### Backend (Python)

Prerequisites: Python 3.11+, and [Ollama](https://ollama.com/) if you want to run fully local.

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api:app --reload
```

To use Groq instead of Ollama locally, set these before starting the API:

```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_key_here
export EMBEDDING_PROVIDER=fastembed
```

### Frontend (Next.js)

```bash
cd web
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`, upload documents, and ask questions.

### Command line

```bash
python app.py --reindex                     # build the index from docs/
python app.py --stream --hybrid             # ask with streaming + hybrid search
python app.py --async-stream --hybrid --rerank
```

| Option | Purpose |
| --- | --- |
| `--reindex` | Rebuild the document index from `docs/` |
| `--stream` | Stream an answer in the terminal |
| `--async-stream` | Asynchronously prepare context and stream the answer |
| `--hybrid` | Combine vector search with BM25 keyword search |
| `--rerank` | Use a cross-encoder to rerank retrieved chunks (needs `requirements-rerank.txt`) |
| `--extension .pdf` | Search only a particular file type |
| `--evaluate` | Run retrieval evaluation questions |

## Deploy it (free tier)

### Backend → Render

1. Push to GitHub.
2. On [Render](https://render.com): **New → Blueprint** → pick this repo. It reads `render.yaml` automatically (Docker, free plan).
3. When prompted, add your **`GROQ_API_KEY`** (all other env vars come from `render.yaml`).
4. Wait for the build (~5–10 min — it pre-downloads the embedding model). When it shows **Live**, copy the service URL.

The cloud image uses Groq + FastEmbed (no PyTorch) so it fits in a 512MB instance.

### Frontend → GitHub Pages

The workflow in `.github/workflows/pages.yml` builds `web/` and deploys to GitHub Pages on every push.

- It bakes in the backend URL via the `NEXT_PUBLIC_API_URL` Actions variable (defaults to the Render URL).
- To point at a different backend, add a repository Actions **variable** `NEXT_PUBLIC_API_URL`, or use the ⚙️ settings control in the app to set it per-browser.

## Configuration

Settings live in `config.py` and are overridable by environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama`, `groq`, or `openai` | `ollama` |
| `LLM_MODEL` | Chat model id | provider-specific |
| `GROQ_API_KEY` | Groq key (cloud LLM) | — |
| `EMBEDDING_PROVIDER` | `ollama`, `fastembed`, or `huggingface` | `ollama` |
| `FASTEMBED_MODEL` | FastEmbed model id | `BAAI/bge-small-en-v1.5` |
| `OLLAMA_URL` | Ollama address (local) | `http://localhost:11434` |
| `DATA_DIR` | Where Chroma index / SQLite live | `data` |

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall -q . -x 'venv|data'
```

## Notes

- The `data/` directory is generated local state — do not commit it.
- Answers come only from retrieved document context; the model says so when the documents lack enough relevant information.
- On the free-tier demo, the backend sleeps when idle and uploaded documents are not persisted.
