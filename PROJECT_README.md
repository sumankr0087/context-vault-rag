# ContextVault — Project Guide

## What this project does

ContextVault is a local document-question-answering application. Upload PDF, TXT, or Markdown files, then ask questions in a chat interface. The assistant retrieves the most relevant passages from your documents and uses them as context for its answer.

Your documents, vector index, conversation memory, and AI inference stay on your machine. The project is designed for personal notes, study material, technical documentation, reports, resumes, and other private files.

## How it works

```text
Documents -> loaders -> chunks -> embeddings -> Chroma vector store
Question + chat history -> hybrid retrieval -> optional reranking -> Ollama -> streamed answer
```

1. A loader reads PDF, TXT, or Markdown files.
2. Long text is split into overlapping chunks.
3. Ollama creates an embedding for each chunk.
4. Chroma stores and searches those embeddings locally.
5. The best matching chunks, plus recent chat history from SQLite, are sent to the local Ollama chat model.
6. The answer is streamed to the CLI or Streamlit interface.

## Language and technologies

| Area | Technology |
| --- | --- |
| Main language | Python |
| Local LLM and embeddings | Ollama with `qwen3.5:4b` and `qwen3-embedding:0.6b` |
| RAG framework | LangChain |
| Vector database | Chroma |
| Chat memory | SQLite |
| Backend API | FastAPI and Uvicorn |
| Web interface | Streamlit |
| PDF reading | PyPDF |
| Keyword retrieval | BM25 |
| Optional reranking | Sentence Transformers cross-encoder |
| Automated tests | Python `unittest` |

## Features

- Upload and index multiple PDF, TXT, and Markdown documents.
- Select the exact uploaded document or documents to search before asking a question.
- Skip byte-for-byte duplicate uploads so repeated files do not dilute retrieval quality.
- Ask questions through a browser chat interface or the command line.
- Receive streamed answers as the local model generates them.
- Display the filename(s) used as evidence below every web answer.
- Keep separate chat topics in the sidebar and reopen a previous topic.
- Store recent conversation context in SQLite.
- Use semantic MMR retrieval to balance relevance and diversity.
- Enable hybrid search to combine semantic retrieval with BM25 keyword matching.
- Optionally rerank retrieved chunks with a cross-encoder before answering.
- Filter CLI retrieval by file extension.
- Evaluate retrieval quality with hit rate and MRR against test questions.
- Run locally without sending your documents to a cloud AI service.

## Project structure

```text
api.py              FastAPI upload and chat API
app.py              Command-line application
ui.py               Streamlit browser interface
config.py           Models, paths, chunking, and retrieval settings
loaders/            PDF, TXT, and Markdown loaders
embeddings/         Embedding client setup
vectorstore/        Chroma, MMR, BM25, and reranking retrieval
rag/                RAG pipelines and prompt context
memory/             SQLite chat memory
llm/                Ollama chat client and prompts
utils/              Chunking and metadata helpers
docs/               Source documents and uploaded files
data/               Generated Chroma index and SQLite database
tests/              Offline unit tests
```

## Setup

### 1. Prerequisites

- Python 3.11 or later
- [Ollama](https://ollama.com/) installed and running

Download the local models used by the default configuration:

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For optional cross-encoder reranking:

```bash
pip install -r requirements-rerank.txt
```

## How to use it

### Browser interface

Start the API in one terminal:

```bash
uvicorn api:app --reload
```

Start the Streamlit interface in another terminal:

```bash
streamlit run ui.py
```

Open `http://localhost:8501`, upload documents, select the file or files to search, and ask a question. The assistant shows the source filenames below each answer. Re-uploading an identical file is safely skipped. Use **New chat** to begin another conversation; a topic is added to the sidebar after the first prompt.

### Command line

Place documents in `docs/`, then create or rebuild the index:

```bash
python app.py --reindex
```

Ask questions:

```bash
python app.py --session study
python app.py --stream --hybrid
python app.py --async-stream --hybrid --rerank
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--reindex` | Rebuild the document index from `docs/` |
| `--stream` | Stream an answer in the terminal |
| `--async` | Generate an answer using the asynchronous pipeline |
| `--async-stream` | Asynchronously prepare context and stream the answer |
| `--hybrid` | Combine vector search with BM25 keyword search |
| `--rerank` | Use a cross-encoder to rerank retrieved chunks |
| `--extension .pdf` | Search only a particular file type |
| `--mmr-lambda 0.3` | Change the relevance/diversity balance |
| `--evaluate` | Run retrieval evaluation questions |

## Configuration

Default settings are in `config.py`. You can change the chat model, embedding model, chunk size, overlap, number of retrieval results, and retrieval settings there.

Set `OLLAMA_URL` if Ollama is running on another address:

```bash
export OLLAMA_URL=http://localhost:11434
```

## Testing

Run the offline unit tests:

```bash
python -m unittest discover -s tests -v
```

Check Python syntax:

```bash
python -m compileall -q . -x 'venv|data'
```

## Notes

- The `data/` directory is generated local state. Do not commit it to source control.
- Run `--reindex` after adding or changing files in `docs/` when using the CLI.
- The web interface indexes uploaded documents immediately.
- This project answers from retrieved document context; it can still be wrong when the documents do not contain enough relevant information.
