"""Resilient entrypoint for Cloud Run.

Importing the real app can fail at container startup for environment-specific
reasons (read-only filesystem, missing native libs, version issues). If that
happens, uvicorn would crash and the container would never open its port, so
Cloud Run reports an opaque "failed to listen on PORT" error.

This wrapper always produces an importable ``app``: the real one if it loads,
otherwise a minimal fallback that binds the port and serves the exact import
error on every route, so the failure is visible instead of hidden.
"""

import traceback

try:
    from api import app  # noqa: F401  (re-exported for uvicorn: server:app)
except Exception:  # pragma: no cover - diagnostic fallback
    _error = traceback.format_exc()
    print("APP IMPORT FAILED:\n" + _error, flush=True)

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get("/health")
    async def _health() -> dict:
        return {"status": "import_error", "error": _error[-3000:]}

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"])
    async def _catch_all(path: str) -> JSONResponse:
        return JSONResponse(status_code=500, content={"status": "import_error", "error": _error[-3000:]})
