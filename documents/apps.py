from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    name = "documents"

    def ready(self):
        # D2 — start tracing as the app comes up, so every LLM call made
        # through the API is instrumented. Done here rather than in settings
        # because `ready()` runs once the app registry is populated, which is
        # what the instrumentors expect.
        #
        # setup_tracing() swallows its own errors: an unreachable Phoenix must
        # not stop Django from serving.
        from rag.tracing import setup_tracing

        setup_tracing()
