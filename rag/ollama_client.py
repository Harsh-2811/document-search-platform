"""Thin Ollama HTTP client — embeddings and chat.

Phase 1.2/1.6. Deliberately stdlib-only: this is a handful of JSON POSTs, and
keeping it dependency-free means the package imports cleanly long before the
heavier RAG libraries land. LlamaIndex brings its own Ollama bindings in
Phase 2; this client stays for ingestion and for `verify_setup()`.

Verify a fresh Ollama install with:

    python -m rag.ollama_client
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Sequence

from rag.config import get_config

# Generation can be slow on CPU; embeddings should not be.
EMBED_TIMEOUT = 120
CHAT_TIMEOUT = 600

# Must match rag.retrieval.CONTEXT_WINDOW. Defined here rather than imported
# to keep this module free of the LlamaIndex import chain.
CONTEXT_WINDOW = 8192


class OllamaError(RuntimeError):
    """Ollama is unreachable, or returned something unusable."""


def _request(path: str, payload: dict | None = None, *, timeout: int) -> Any:
    base = get_config().ollama_base_url
    url = f"{base}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise OllamaError(f"{url} returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {base} ({exc.reason}). Is it running? "
            f"From inside Docker the host is http://host.docker.internal:11434."
        ) from exc


def list_models() -> list[str]:
    """Model names currently pulled, e.g. ['nomic-embed-text:latest']."""
    payload = _request("/api/tags", timeout=10)
    return [model["name"] for model in payload.get("models", [])]


def embed(texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
    """Embed one batch of texts, preserving input order."""
    if not texts:
        return []
    model = model or get_config().embed_model
    payload = _request(
        "/api/embed",
        {"model": model, "input": list(texts)},
        timeout=EMBED_TIMEOUT,
    )
    embeddings = payload.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise OllamaError(
            f"Expected {len(texts)} embeddings from {model}, got "
            f"{len(embeddings or [])}."
        )
    return embeddings


def chat(messages: list[dict[str, str]], *, model: str | None = None) -> str:
    """Single non-streaming chat completion. Streaming arrives in Phase 4."""
    model = model or get_config().chat_model
    payload = _request(
        "/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            # Pin the context width. Left to itself Ollama loads llama3.2:3b
            # at its advertised 128k, which allocates an ~18GB KV cache on a
            # 16GB host and drives the machine into swap. Keep this in step
            # with rag.retrieval.CONTEXT_WINDOW: a call that requests a
            # different width forces a full model reload.
            "options": {"num_ctx": CONTEXT_WINDOW},
        },
        timeout=CHAT_TIMEOUT,
    )
    return payload["message"]["content"]


def verify_setup() -> dict[str, Any]:
    """Check both models are pulled and the embedding width matches the schema.

    The dimension check is the one that bites: `Chunk.embedding` is a fixed
    768-wide column, so an embedding model of any other width fails at insert
    time, deep inside ingestion, long after the real mistake was made.
    """
    config = get_config()
    available = list_models()

    def is_pulled(name: str) -> bool:
        # `ollama pull llama3.1:8b` reports as 'llama3.1:8b'; a bare name
        # resolves to ':latest'.
        wanted = name if ":" in name else f"{name}:latest"
        return wanted in available or name in available

    report = {
        "base_url": config.ollama_base_url,
        "models_available": available,
        "embed_model": config.embed_model,
        "embed_model_pulled": is_pulled(config.embed_model),
        "chat_model": config.chat_model,
        "chat_model_pulled": is_pulled(config.chat_model),
        "expected_dimensions": config.embedding_dim,
        "actual_dimensions": None,
        "dimensions_match": False,
    }

    if report["embed_model_pulled"]:
        actual = len(embed(["dimension probe"])[0])
        report["actual_dimensions"] = actual
        report["dimensions_match"] = actual == config.embedding_dim

    return report


def _main() -> int:
    try:
        report = verify_setup()
    except OllamaError as exc:
        print(f"FAIL  {exc}")
        return 1

    ok = "ok"
    bad = "MISSING"
    print(f"Ollama          {report['base_url']}")
    print(f"models pulled   {', '.join(report['models_available']) or '(none)'}")
    print(
        f"embed model     {report['embed_model']} "
        f"[{ok if report['embed_model_pulled'] else bad}]"
    )
    print(
        f"chat model      {report['chat_model']} "
        f"[{ok if report['chat_model_pulled'] else bad}]"
    )
    print(
        f"dimensions      expected {report['expected_dimensions']}, "
        f"got {report['actual_dimensions']} "
        f"[{ok if report['dimensions_match'] else 'MISMATCH'}]"
    )

    healthy = (
        report["embed_model_pulled"]
        and report["chat_model_pulled"]
        and report["dimensions_match"]
    )
    print("\nStep 1.2:", "PASS" if healthy else "not satisfied yet")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(_main())
