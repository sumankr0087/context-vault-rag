# DocuSense backend (FastAPI). Runs on Groq LLM + FastEmbed (ONNX) embeddings,
# so it needs no local Ollama. Works on Hugging Face Spaces, Render, any Docker host.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the (small) embedding model into a world-readable cache so the
# first request is fast and memory stays low on constrained hosts.
RUN FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')" || true
RUN chmod -R 777 /app/.cache 2>/dev/null || true

# Cloud defaults: cloud LLM + lightweight ONNX embeddings (no PyTorch).
# Writable paths live under /tmp so the app works whether it runs as root
# (Render) or as a non-root user, and listens on the host's $PORT.
# The MALLOC_*/*_NUM_THREADS vars keep Linux RSS low (cap glibc arenas,
# return freed memory to the OS, keep math libs single-threaded).
ENV LLM_PROVIDER=groq \
    LLM_MODEL=openai/gpt-oss-120b \
    EMBEDDING_PROVIDER=fastembed \
    FASTEMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/tmp/hf \
    DATA_DIR=/tmp/data \
    DOCS_DIR=/tmp/data/docs \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=65536 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8080

# Listen on the platform-provided $PORT (Cloud Run sets 8080; Render sets its own).
# GROQ_API_KEY is provided at runtime (Cloud Run / Render env / docker -e).
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]
