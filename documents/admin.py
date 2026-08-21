from django.contrib import admin

from .models import Chunk, Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("filename", "doc_type", "total_chunks", "uploaded_at")
    list_filter = ("doc_type",)
    search_fields = ("filename",)


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "chunk_index", "preview")
    list_filter = ("document__doc_type",)
    search_fields = ("content",)
    raw_id_fields = ("document",)
    # 768 floats in a textarea is unusable and easy to corrupt by accident.
    exclude = ("embedding",)

    @admin.display(description="content")
    def preview(self, obj):
        return obj.content[:120] + ("…" if len(obj.content) > 120 else "")
