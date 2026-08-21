"""
title: Document Search (RAG)
author: document-search-project
version: 0.1.0
description: Answers from the ingested PDF library via the Django/DRF RAG API.
requirements:
"""

# C4 — OpenWebUI custom Pipe.
#
# HOW TO INSTALL (this cannot be done from a config file):
#   1. Open http://127.0.0.1:3000
#   2. Workspace -> Functions -> "+"  (or Admin Panel -> Functions)
#   3. Paste this entire file, Save, then toggle it ON
#   4. In a new chat, pick "Document Search (RAG)" from the model dropdown
#
# The pipe forwards each question to POST /api/chat/ and returns the grounded
# answer with its sources appended.

from typing import List, Optional

import requests
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        api_url: str = Field(
            # host.docker.internal, not localhost: this runs inside the
            # OpenWebUI container, where localhost is that container.
            default="http://host.docker.internal:8000/api/chat/",
            description="URL of the Django chat endpoint.",
        )
        timeout_seconds: int = Field(
            # Generation is CPU-bound here (~70s warm, longer if the model has
            # to reload). A short timeout turns a slow answer into an error.
            default=600,
            description="How long to wait for an answer.",
        )
        top_k: int = Field(
            default=4,
            description="Chunks to retrieve per question.",
        )
        show_sources: bool = Field(
            default=True,
            description="Append the source documents beneath the answer.",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "document_search_rag"
        self.name = "Document Search (RAG)"
        self.valves = self.Valves()

    def pipe(self, body: dict) -> str:
        messages: List[dict] = body.get("messages", []) or []
        if not messages:
            return "No question received."

        question: Optional[str] = None
        for message in reversed(messages):
            if message.get("role") == "user":
                question = (message.get("content") or "").strip()
                break
        if not question:
            return "No question received."

        # Everything before the current question, so follow-ups resolve.
        # OpenWebUI messages can carry non-string content (images); keep only
        # the plain-text turns the API expects.
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages[:-1]
            if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str)
        ]

        try:
            response = requests.post(
                self.valves.api_url,
                json={
                    "question": question,
                    "history": history,
                    "top_k": self.valves.top_k,
                },
                timeout=self.valves.timeout_seconds,
            )
        except requests.exceptions.Timeout:
            return (
                f"The answer took longer than {self.valves.timeout_seconds}s. "
                "Generation runs on CPU here — try a simpler question, or "
                "raise `timeout_seconds` in this function's valves."
            )
        except requests.exceptions.RequestException as exc:
            return f"Could not reach the RAG API at {self.valves.api_url}: {exc}"

        if response.status_code != 200:
            return f"API returned HTTP {response.status_code}: {response.text[:400]}"

        data = response.json()
        answer = data.get("answer", "").strip() or "(empty answer)"

        if not self.valves.show_sources:
            return answer

        sources = data.get("sources") or []
        if not sources:
            return answer

        lines = [answer, "", "---", "**Sources**"]
        for source in sources:
            filename = source.get("filename", "?")
            heading = source.get("heading") or ""
            score = source.get("score")
            detail = f" — {heading}" if heading else ""
            detail += f" (score {score})" if score is not None else ""
            lines.append(f"- `{filename}`{detail}")
        return "\n".join(lines)
