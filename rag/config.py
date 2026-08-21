"""Environment-driven configuration for the RAG package.

Reads from `os.environ`. Django's settings module populates the process
environment from `.env` on startup (django-environ's `read_env` uses
`setdefault`), so a management command and a standalone script see the same
values — without this package importing Django to get them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import quote_plus


class ConfigError(RuntimeError):
    """A required environment variable is missing or unusable."""


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"{name} is not set. Copy .env.example to .env, or export it "
            f"before running this process."
        )
    return value


@dataclass(frozen=True)
class RagConfig:
    """Everything the RAG pipeline needs to reach its backing services."""

    # Ollama (Phase 1: embeddings, Phase 2+: chat/reasoning)
    ollama_base_url: str
    embed_model: str
    chat_model: str
    embedding_dim: int

    # PostgreSQL + PGVector (same database Django writes chunks into)
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: int

    # Arize Phoenix (Phase 5)
    phoenix_endpoint: str

    @property
    def database_url(self) -> str:
        """SQLAlchemy-style URL for LlamaIndex's PGVectorStore (Phase 2).

        Credentials are percent-encoded: passwords routinely contain `@` or
        `/`, which silently corrupt the URL otherwise.
        """
        user = quote_plus(self.db_user)
        password = quote_plus(self.db_password)
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def dsn(self) -> str:
        """libpq connection string for psycopg.

        Separate from `database_url` above: that one carries SQLAlchemy's
        `+psycopg` driver prefix, which libpq itself does not understand.
        """
        return (
            f"host={self.db_host} port={self.db_port} dbname={self.db_name} "
            f"user={self.db_user} password={self.db_password}"
        )

    @classmethod
    def from_env(cls) -> RagConfig:
        try:
            embedding_dim = int(os.environ.get("EMBEDDING_DIM", "768"))
        except ValueError as exc:
            raise ConfigError("EMBEDDING_DIM must be an integer") from exc

        try:
            db_port = int(os.environ.get("DB_PORT", "5432"))
        except ValueError as exc:
            raise ConfigError("DB_PORT must be an integer") from exc

        return cls(
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            chat_model=os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b"),
            embedding_dim=embedding_dim,
            db_name=_require("DB_NAME"),
            db_user=_require("DB_USER"),
            db_password=_require("DB_PASSWORD"),
            db_host=os.environ.get("DB_HOST", "localhost"),
            db_port=db_port,
            phoenix_endpoint=os.environ.get(
                "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
            ),
        )


@lru_cache(maxsize=1)
def get_config() -> RagConfig:
    """Process-wide config, built once on first use.

    Cached rather than module-level so importing `rag.config` never raises —
    tests and tooling can import the package without a full environment.
    """
    return RagConfig.from_env()
