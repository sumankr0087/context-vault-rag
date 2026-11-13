# DocuSense backend (FastAPI). Runs on Groq LLM + FastEmbed (ONNX) embeddings,
# so it needs no local Ollama. Works on Render / any Docker host.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud defaults: cloud LLM + lightweight ONNX embeddings (no PyTorch),
# so the service fits in a 512MB free instance.
ENV LLM_PROVIDER=groq \
    LLM_MODEL=openai/gpt-oss-120b \
    EMBEDDING_PROVIDER=fastembed \
    FASTEMBED_MODEL=BAAI/bge-small-en-v1.5 \
    DATA_DIR=/app/data \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# Pre-download the embedding model so the first request isn't slow / failing.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" || true

EXPOSE 7860

# GROQ_API_KEY is provided at runtime (Render env / docker -e).
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
