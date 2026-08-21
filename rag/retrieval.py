"""Phase 2 / Block B — LlamaIndex over the PGVector store.

LlamaIndex reads the same `documents_chunk` table the Django `Chunk` model
writes to: one table, two accessors. Django owns writes during ingestion;
this module only reads.

Why a custom retriever rather than LlamaIndex's `PGVectorStore`: that class
owns its schema (`text`, `metadata_`, `node_id`) and expects to have created
the table itself. Ours is a Django-managed table with `content`, `metadata`
and a real foreign key to `documents_document`. Rather than distort either
schema to satisfy the other, this retriever reads the Django table directly
and hands LlamaIndex the `NodeWithScore` objects it wants. LlamaIndex still
orchestrates retrieval and synthesis; only the storage adapter is ours.

psycopg rather than the Django ORM keeps the package importable without
Django, which `rag.test_django_free` enforces.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TOP_K = 4

# Ollama on CPU is slow; a short default timeout produces confusing failures
# mid-generation rather than an honest wait.
LLM_TIMEOUT_SECONDS = 600.0

# Cap the context window. llama3.2:3b advertises 128k, and Ollama will happily
# load it at full width — which allocates an ~18GB KV cache on a 16GB host and
# sends the whole machine into swap (a two-word reply took 3m42s in that
# state). RAG prompts here are a handful of chunks; 8k is ample.
CONTEXT_WINDOW = 8192

# Keep the model resident in RAM indefinitely. The model files live on a SATA
# HDD (C: had no room), so an eviction costs minutes to reload — far worse
# than the ~2GB of RAM held. Without this, every call after a 5-minute idle
# pays a full reload and LlamaIndex times out. Ollama's keep_alive must carry
# a unit (e.g. "-1m"); the bare "-1" it used to accept is now a 400.
KEEP_ALIVE = "-1m"

# Number of model layers to offload to the GPU, or None to let Ollama decide.
#
# Set OLLAMA_NUM_GPU=0 to force CPU-only. That sounds backwards, but a GPU too
# small to hold the model is far worse than no GPU at all: Ollama offloads only
# the layers that fit and every token then round-trips across PCIe. Measured on
# this machine (GeForce GT 710, 2GB, holding 23%% of llama3.2:3b), a 344-token
# prompt took 46.5s at 7.8 tok/s; with num_gpu=0 the same prompt took 3.2s at
# 108 tok/s, and generation went 5.2 -> 18.8 tok/s. Ten times faster on the CPU.
_num_gpu_env = os.environ.get("OLLAMA_NUM_GPU", "").strip()
NUM_GPU = int(_num_gpu_env) if _num_gpu_env else None


@dataclass
class Answer:
    """What the API layer serializes back to the caller."""

    text: str
    sources: list[dict[str, Any]] = field(default_factory=list)


def configure_llama_index() -> None:
    """Point LlamaIndex's global Settings at our Ollama instance.

    Both models must match ingestion: querying with a different embedding
    model than the one that wrote the vectors returns plausible-looking
    nonsense rather than an error.
    """
    from llama_index.core import Settings
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama

    from rag.config import get_config

    config = get_config()
    Settings.embed_model = OllamaEmbedding(
        model_name=config.embed_model,
        base_url=config.ollama_base_url,
    )
    Settings.llm = Ollama(
        model=config.chat_model,
        base_url=config.ollama_base_url,
        request_timeout=LLM_TIMEOUT_SECONDS,
        keep_alive=KEEP_ALIVE,
        context_window=CONTEXT_WINDOW,
        # Reaches Ollama as options.num_gpu. Omitted entirely when unset so
        # machines with a capable GPU keep Ollama's own defaults.
        additional_kwargs=({"num_gpu": NUM_GPU} if NUM_GPU is not None else {}),
    )


def _build_retriever_class():
    """Define the retriever lazily so importing this module stays cheap."""
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    class ChunkRetriever(BaseRetriever):
        """Top-k cosine search over the Django-managed chunk table."""

        def __init__(self, top_k: int = DEFAULT_TOP_K) -> None:
            self.top_k = top_k
            # Every node this retriever has returned, in call order. The crew
            # may reformulate the question and search several times; reading
            # this afterwards gives the passages it actually saw, rather than
            # re-running retrieval on the original wording and guessing.
            self.seen_nodes: list = []
            super().__init__()

        def _retrieve(self, query_bundle) -> list[NodeWithScore]:
            import psycopg

            from rag.config import get_config
            from rag.ollama_client import embed

            vector = embed([query_bundle.query_str])[0]
            # Rendered as a literal and cast: avoids registering a pgvector
            # type adapter just to run one read query.
            literal = "[" + ",".join(str(v) for v in vector) + "]"

            sql = """
                SELECT c.content, c.metadata, c.chunk_index, d.filename,
                       c.embedding <=> %s::vector AS distance
                FROM documents_chunk c
                JOIN documents_document d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """
            with psycopg.connect(get_config().dsn) as conn:
                rows = conn.execute(sql, (literal, literal, self.top_k)).fetchall()

            nodes = []
            for content, metadata, chunk_index, filename, distance in rows:
                node_metadata = {
                    **(metadata or {}),
                    "filename": filename,
                    "chunk_index": chunk_index,
                }
                node = TextNode(
                    # Docling emits HTML entities into its Markdown; left as-is
                    # they reach both the prompt and the answer as "&amp;".
                    text=html.unescape(content),
                    metadata=node_metadata,
                    # Keep metadata out of the text handed to the LLM. By
                    # default LlamaIndex prepends it to each chunk, so the
                    # model sees "filename: x / chunk_index: 3" and imitates
                    # that shape — emitting "Document: ... chunk_index: ..."
                    # trailers instead of prose. The API returns the same
                    # information structurally in `sources`.
                    excluded_llm_metadata_keys=list(node_metadata),
                )
                # Cosine distance -> similarity, so higher means better and
                # LlamaIndex's score-based filters behave as expected.
                nodes.append(NodeWithScore(node=node, score=1.0 - float(distance)))
            self.seen_nodes.extend(nodes)
            return nodes

    return ChunkRetriever


def get_retriever(top_k: int = DEFAULT_TOP_K):
    """A LlamaIndex retriever over the ingested chunks."""
    configure_llama_index()
    return _build_retriever_class()(top_k=top_k)


# Kept inline for now; Block D3 moves prompts out of code into `prompts/`.
QA_PROMPT = """Answer the question using only the context below.

Rules:
- Use only the context. If it does not contain the answer, reply exactly:
  "The documents don't cover that."
- Quote figures, names and dates exactly as they appear.
- Answer in plain prose — one or two sentences, or a short bullet list if the
  answer is genuinely a list.
- Do NOT append citations, filenames, headings, chunk numbers, "Document:",
  "Source:" or any similar trailer. Source attribution is handled elsewhere;
  adding it here corrupts the answer.

Context:
---------------------
{context_str}
---------------------

Question: {query_str}
Answer:"""


def get_query_engine(top_k: int = DEFAULT_TOP_K):
    """B2 — retrieve top-k, stuff into one prompt, let Ollama answer."""
    from llama_index.core import get_response_synthesizer
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.query_engine import RetrieverQueryEngine

    retriever = get_retriever(top_k=top_k)
    synthesizer = get_response_synthesizer(
        # "compact" packs the chunks into as few LLM calls as possible.
        # "refine" would issue one call per chunk, which at ~30s per call on
        # CPU turns a 4-chunk answer into two minutes.
        response_mode="compact",
        text_qa_template=PromptTemplate(QA_PROMPT),
    )
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)


def answer_question(question: str, *, top_k: int = DEFAULT_TOP_K) -> Answer:
    """B2 — retrieve, ground, and answer. The DRF view calls this."""
    response = get_query_engine(top_k=top_k).query(question)

    sources = []
    seen = set()
    for node in response.source_nodes:
        filename = node.metadata.get("filename", "")
        # One entry per document: the API surfaces which documents backed the
        # answer, not every chunk that happened to be retrieved.
        if filename in seen:
            continue
        seen.add(filename)
        sources.append(
            {
                "filename": filename,
                "heading": node.metadata.get("heading", ""),
                "chunk_index": node.metadata.get("chunk_index"),
                "score": round(float(node.score), 4) if node.score is not None else None,
            }
        )

    return Answer(text=str(response).strip(), sources=sources)
