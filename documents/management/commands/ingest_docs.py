"""A8 — ingest PDFs into the vector store.

    python manage.py ingest_docs                  # every PDF in data/
    python manage.py ingest_docs data/one.pdf     # a single file
    python manage.py ingest_docs --doc-type resume

The heavy lifting (parse, chunk, embed) lives in `rag/`, which knows nothing
about Django. This command is the seam: it drives those functions and maps the
returned `TextChunk`s onto `Chunk` rows. Ingestion runs here rather than in a
view because embedding a document takes far longer than any request should.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from documents.models import Chunk, Document

DEFAULT_SOURCE_DIR = Path("data")


class Command(BaseCommand):
    help = "Parse, chunk, embed, and index PDFs into the vector store."

    def add_arguments(self, parser):
        parser.add_argument(
            "paths",
            nargs="*",
            type=Path,
            help="PDF files to ingest. Defaults to every PDF in data/.",
        )
        parser.add_argument(
            "--doc-type",
            default="",
            help="Category stored on each Document, e.g. 'resume'.",
        )
        parser.add_argument(
            "--skip-existing",
            action="store_true",
            help="Leave already-ingested files alone instead of re-ingesting.",
        )

    def handle(self, *args, **options):
        paths = options["paths"] or sorted(DEFAULT_SOURCE_DIR.glob("*.pdf"))
        if not paths:
            raise CommandError(
                f"No PDFs found. Put files in {DEFAULT_SOURCE_DIR}/ or pass paths."
            )

        for path in paths:
            self.ingest_one(path, options["doc_type"], options["skip_existing"])

        total = Chunk.objects.count()
        self.stdout.write(
            self.style.SUCCESS(f"Done. {total} chunks across {Document.objects.count()} documents.")
        )

    def ingest_one(self, path: Path, doc_type: str, skip_existing: bool) -> None:
        # Imported lazily so `manage.py help` and unrelated commands don't pay
        # for Docling's torch import.
        from rag.ingest import chunk_text, embed_chunks, parse_pdf

        if not path.is_file():
            raise CommandError(f"No such file: {path}")

        if skip_existing and Document.objects.filter(filename=path.name).exists():
            self.stdout.write(f"{path.name}: already ingested, skipping")
            return

        self.stdout.write(f"{path.name}: parsing...")
        markdown = parse_pdf(path)

        chunks = chunk_text(markdown, source=path.name)
        if not chunks:
            self.stdout.write(self.style.WARNING(f"{path.name}: no text extracted, skipped"))
            return

        self.stdout.write(f"{path.name}: {len(chunks)} chunks, embedding...")
        chunks = embed_chunks(chunks)

        # One transaction per document: a failure part-way through leaves the
        # catalog consistent rather than half-indexed.
        with transaction.atomic():
            document, created = Document.objects.update_or_create(
                filename=path.name,
                defaults={"doc_type": doc_type, "total_chunks": len(chunks)},
            )
            # Re-ingestion replaces rather than appends, so chunk_index stays
            # unique per document and stale text cannot linger in results.
            if not created:
                document.chunks.all().delete()

            Chunk.objects.bulk_create(
                [
                    Chunk(
                        document=document,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        embedding=chunk.embedding,
                        metadata=chunk.metadata,
                    )
                    for chunk in chunks
                ],
                batch_size=100,
            )

        verb = "ingested" if created else "re-ingested"
        self.stdout.write(self.style.SUCCESS(f"{path.name}: {verb}, {len(chunks)} chunks"))
