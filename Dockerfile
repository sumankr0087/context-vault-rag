# ContextVault backend (FastAPI). Runs on Groq LLM + HuggingFace embeddings,
# so it needs no local Ollama. Works on Render / any Docker host.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud defaults: cloud LLM + serverless embeddings, writable dirs.
ENV LLM_PROVIDER=groq \
    LLM_MODEL=openai/gpt-oss-120b \
    EMBEDDING_PROVIDER=huggingface \
    HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    DATA_DIR=/app/data \
    HF_HOME=/app/.cache/huggingface

# Pre-download the embedding model so first request isn't slow / failing.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || true

EXPOSE 7860

# GROQ_API_KEY is provided at runtime (Render env / docker -e).
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
