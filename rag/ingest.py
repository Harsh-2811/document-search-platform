"""Phase 1 / Block A — parse, chunk, and embed source documents.

Pipeline: PDF -> Docling -> clean Markdown -> overlapping chunks -> Ollama
embeddings. The caller (a Django management command) is what turns the
returned `TextChunk`s into `Chunk` rows; nothing here touches the ORM.

Eyeball a single PDF's parse output (A5):

    python -m rag.ingest data/sample_consulting_resume.pdf
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# ~300 tokens per chunk at roughly 4 chars/token, with enough overlap that a
# fact spanning a boundary stays retrievable from either side.
CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

# Chunks per Ollama embeddings request.
EMBED_BATCH_SIZE = 16


@dataclass
class TextChunk:
    """One embeddable slice of a document.

    Mirrors the fields of the `Chunk` model so the ingest command is a plain
    field-for-field mapping, with no translation logic in the Django layer.
    """

    chunk_index: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


def parse_pdf(path: Path | str) -> str:
    """A5 — run a PDF through Docling, returning structured Markdown.

    Markdown rather than raw text on purpose: Docling preserves headings,
    lists, and tables as markup, which survives chunking and gives the
    retriever real structure to cite instead of a wall of text.
    """
    # Imported lazily: Docling drags in torch, so a module-level import would
    # make `import rag.ingest` cost seconds and would break the Django-free
    # guard test on any machine where docling isn't installed yet.
    from docling.document_converter import DocumentConverter

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such PDF: {path}")

    converter = DocumentConverter()
    result = converter.convert(str(path))
    # Docling emits HTML-escaped Markdown, so an ampersand in a heading arrives
    # as "&amp;". Left alone it reaches both the LLM prompt and the cited
    # source line in the UI verbatim - "Breakroom &amp; Supplies". Unescape
    # once here, at the only place raw Docling output enters the pipeline.
    return html.unescape(result.document.export_to_markdown())


def chunk_text(
    text: str,
    *,
    source: str,
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[TextChunk]:
    """A6 — split Markdown into overlapping chunks, tagged with their heading.

    Splits on line boundaries and tracks the most recent Markdown heading, so
    every chunk carries the section it came from. That heading is what makes a
    retrieved chunk citable ("Experience", not "characters 4200-5400"), and it
    is stored in `metadata` for the API to hand back as a source reference.

    A hand-rolled splitter rather than Docling's HybridChunker on purpose: the
    latter pulls a tokenizer from HuggingFace on first use, and this deployment
    is on a slow link where an extra model download is a real cost.
    """
    lines = text.splitlines()
    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_len = 0
    heading = ""
    chunk_heading = ""

    def flush() -> None:
        nonlocal buffer, buffer_len, chunk_heading
        content = "\n".join(buffer).strip()
        if not content:
            buffer, buffer_len = [], 0
            chunk_heading = heading
            return
        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                content=content,
                metadata={"source": source, "heading": chunk_heading},
            )
        )
        # Carry the tail of this chunk into the next so a fact spanning the
        # boundary is retrievable from either side.
        tail: list[str] = []
        tail_len = 0
        for line in reversed(buffer):
            if tail_len >= overlap_chars:
                break
            tail.insert(0, line)
            tail_len += len(line) + 1
        buffer, buffer_len = tail, tail_len
        # The next chunk belongs to whatever section is in effect now. Set here
        # rather than on an empty-buffer check: the overlap tail leaves the
        # buffer non-empty, so such a check would never fire and every chunk
        # would inherit the first heading in the document.
        chunk_heading = heading

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            # A heading starts a new section: flush first so chunks don't
            # straddle two unrelated sections.
            if buffer_len:
                flush()
            heading = stripped.lstrip("#").strip()
            chunk_heading = heading

        buffer.append(line)
        buffer_len += len(line) + 1

        if buffer_len >= target_chars:
            flush()

    if buffer_len:
        flush()

    # Re-index: `flush` numbers as it goes, but the overlap carry-over means an
    # empty flush can be skipped, leaving gaps.
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index

    return chunks


def embed_chunks(
    chunks: Sequence[TextChunk], *, batch_size: int = EMBED_BATCH_SIZE
) -> list[TextChunk]:
    """A6 — populate `.embedding` for every chunk via Ollama.

    Batched to keep HTTP round-trips down. The width is checked against the
    configured dimension here rather than at insert time: a mismatch is a
    misconfigured model, and the error is far clearer now than as a Postgres
    "expected 768 dimensions" deep inside a bulk_create.
    """
    from rag.config import get_config
    from rag.ollama_client import embed

    expected = get_config().embedding_dim
    chunks = list(chunks)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed([chunk.content for chunk in batch])
        for chunk, vector in zip(batch, vectors):
            if len(vector) != expected:
                raise ValueError(
                    f"Embedding model returned {len(vector)} dimensions but "
                    f"Chunk.embedding is {expected}-wide. Check "
                    f"OLLAMA_EMBED_MODEL and EMBEDDING_DIM."
                )
            chunk.embedding = vector

    return chunks


def _main() -> int:
    """Parse one PDF and print the Markdown, so A5's output can be eyeballed."""
    import sys

    if len(sys.argv) != 2:
        print(f"usage: python -m rag.ingest <path-to.pdf>")
        return 2

    path = Path(sys.argv[1])
    markdown = parse_pdf(path)

    print(f"--- {path.name} ---")
    print(f"characters : {len(markdown)}")
    print(f"lines      : {markdown.count(chr(10)) + 1}")
    print("--- markdown ---")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
