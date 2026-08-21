from django.db import models
from pgvector.django import HnswIndex, VectorField

# Must match the output dimension of the Ollama embedding model
# (OLLAMA_EMBED_MODEL / EMBEDDING_DIM in .env — nomic-embed-text is 768).
# Deliberately a literal rather than read from the environment: a migration
# has to describe one fixed column shape, and an env-dependent value would
# make the same migration produce different schemas on different machines.
EMBEDDING_DIMENSIONS = 768


class Document(models.Model):
    """Catalog entry for one source file."""

    filename = models.CharField(max_length=255, unique=True)
    doc_type = models.CharField(
        max_length=50,
        blank=True,
        help_text="Category assigned at ingest, e.g. 'resume', 'floor_plan'.",
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    total_chunks = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.filename


class Chunk(models.Model):
    """One embedded slice of a document — the unit retrieval works on."""

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    chunk_index = models.PositiveIntegerField(
        help_text="Position within the document, starting at 0.",
    )
    content = models.TextField()
    embedding = VectorField(dimensions=EMBEDDING_DIMENSIONS)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Section heading, page number, and anything else retrieval "
        "should be able to cite.",
    )

    class Meta:
        ordering = ["document", "chunk_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["document", "chunk_index"],
                name="unique_chunk_index_per_document",
            ),
        ]
        indexes = [
            # ANN index for similarity search. `vector_cosine_ops` must match
            # the distance function used at query time (CosineDistance / <=>);
            # a mismatch silently falls back to a sequential scan.
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self):
        return f"{self.document.filename}#{self.chunk_index}"
