"""C1 — the REST surface.

Kept deliberately thin: validate input, call one function in `rag/`, serialize
the result. All retrieval and generation logic lives in the framework-agnostic
`rag` package, which imports nothing from Django.
"""

import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import ChatRequestSerializer, ChatResponseSerializer

logger = logging.getLogger(__name__)


class ChatView(APIView):
    """POST a question, get a grounded answer plus its source documents.

    Answering runs inline and is slow on CPU-only hardware — tens of seconds
    for the plain engine, minutes via the agent crew. That is acceptable for
    an MVP and for OpenWebUI, which waits on the response, but it is the
    reason ingestion lives in a management command rather than a view.
    """

    def post(self, request):
        request_serializer = ChatRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        data = request_serializer.validated_data

        # Imported here, not at module scope: pulling in the RAG stack costs
        # seconds, and `manage.py` commands that merely import URLs shouldn't
        # pay for it.
        from rag.pipeline import answer_question

        kwargs = {}
        if data.get("top_k"):
            kwargs["top_k"] = data["top_k"]

        try:
            answer = answer_question(
                data["question"],
                list(data.get("history") or []),
                **kwargs,
            )
        except Exception:
            # rag.pipeline already falls back from the crew to the plain
            # engine; reaching here means both failed.
            logger.exception("Answering failed for question=%r", data["question"])
            return Response(
                {"detail": "Failed to generate an answer."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        response_serializer = ChatResponseSerializer(
            {"answer": answer.text, "sources": answer.sources}
        )
        return Response(response_serializer.data, status=status.HTTP_200_OK)
