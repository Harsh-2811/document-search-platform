"""D2 — OpenTelemetry tracing for every LLM call, exported to Arize Phoenix.

Auto-instrumentation, not manual spans: `LlamaIndexInstrumentor` patches
LlamaIndex at the framework level, so retrievals, embeddings, prompt templates
and LLM calls all produce spans without touching `rag/retrieval.py`. That
satisfies the brief's "tracing for all inference calls" without scattering
telemetry code through the pipeline.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Set once the instrumentors are attached. Django's autoreloader imports the
# app config twice in development, and instrumenting twice produces duplicate
# spans for every call.
_started = False


def setup_tracing() -> bool:
    """Attach the instrumentors. Returns True if tracing is now active.

    Never raises: a tracing failure must not take the API down with it. If
    Phoenix is unreachable the exporter retries in the background and the app
    keeps answering questions untraced.
    """
    global _started
    if _started:
        return True

    if os.environ.get("RAG_DISABLE_TRACING", "").strip().lower() in {"1", "true", "yes"}:
        logger.info("Tracing disabled via RAG_DISABLE_TRACING.")
        return False

    try:
        from phoenix.otel import register

        from rag.config import get_config

        endpoint = get_config().phoenix_endpoint

        # register() wires the OTLP exporter and sets the global tracer
        # provider. batch=True so exporting never blocks a request.
        tracer_provider = register(
            project_name=os.environ.get("PHOENIX_PROJECT_NAME", "document-search"),
            endpoint=f"{endpoint.rstrip('/')}/v1/traces",
            batch=True,
            set_global_tracer_provider=True,
        )

        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Tracing active -> %s", endpoint)

        # CrewAI is optional: the crew is off by default (RAG_USE_CREW), and
        # its instrumentor is a separate package that may not be installed.
        try:
            from openinference.instrumentation.crewai import CrewAIInstrumentor

            CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
            logger.info("CrewAI instrumentation active.")
        except ImportError:
            logger.info("CrewAI instrumentor not installed; skipping.")

        _started = True
        return True

    except Exception:
        logger.exception("Tracing setup failed; continuing untraced.")
        return False
