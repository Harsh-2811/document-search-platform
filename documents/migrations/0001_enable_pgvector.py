from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    """Enable the pgvector extension.

    Kept as its own first migration so every later migration that creates a
    VectorField column is guaranteed to run after the extension exists.
    """

    initial = True

    dependencies = []

    operations = [
        VectorExtension(),
    ]
