"""Framework-agnostic RAG logic: ingestion, retrieval, and agents.

Hard rule for this package: **nothing here may import Django.**

The RAG libraries (Docling, LlamaIndex, CrewAI, Ollama, Phoenix) are
async-native and slow; Django is synchronous and request-oriented. Keeping
them apart means this code can be driven from a management command, a Celery
worker, a notebook, or a plain script, and can be tested without a Django
settings module or a database connection. DRF views should only ever *call*
these functions.

Configuration comes from the process environment via `rag.config`, never from
`django.conf.settings`. `rag.test_django_free` enforces the rule.

Submodules are intentionally not imported here — importing `rag` must stay
cheap, so pulling in heavy optional dependencies is left to the caller.
"""
