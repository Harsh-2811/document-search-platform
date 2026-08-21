"""B7 — the one function the API calls.

The DRF view should import exactly this and nothing else from `rag/`. Which
strategy answers the question — the CrewAI crew or the plain LlamaIndex query
engine — is decided here, so swapping them never touches the API layer.

Default is the **plain engine**, with the crew available via
`RAG_USE_CREW=true`. That default is evidence-based, not a shortcut:

  plain engine — one generation, correct answers (see scripts/b3_smoketest.py)
  crew         — 201s per question, and answered "I don't know" about a
                 bulk discount while holding the very chunk that stated it

`llama3.2:3b` on CPU is not a strong enough tool-caller to drive a multi-agent
loop reliably, and three minutes per question is not demoable. The crew is
built, wired, and one env var away; when this runs on a machine with a real
GPU, flip the flag and re-measure.

When the crew *is* enabled it still falls back to the plain engine on any
exception, so the API always returns a grounded answer.
"""

from __future__ import annotations

import logging
import os

from rag.retrieval import Answer, DEFAULT_TOP_K

logger = logging.getLogger(__name__)

Message = dict[str, str]


def _use_crew() -> bool:
    return os.environ.get("RAG_USE_CREW", "false").strip().lower() in {
        "true",
        "1",
        "yes",
    }


def _with_history(question: str, history: list[Message] | None) -> str:
    """Fold prior turns into the question so follow-ups resolve.

    Inlined rather than run through a separate LLM "condense question" step:
    that would cost an extra generation per turn, which on this CPU host is
    another minute for a marginal gain.
    """
    if not history:
        return question

    lines = []
    for message in history[-4:]:
        role = message.get("role", "user")
        content = (message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        return question

    return (
        "Conversation so far:\n"
        + "\n".join(lines)
        + f"\n\nGiven that context, answer this question: {question}"
    )


def answer_question(
    question: str,
    history: list[Message] | None = None,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> Answer:
    """Answer a question against the ingested documents.

    `history` is an optional list of prior `{"role", "content"}` turns, used
    so follow-ups like "and what about shipping?" resolve against context.
    """
    query = _with_history(question, history)

    if _use_crew():
        try:
            from rag.agents import answer_question as crew_answer

            return crew_answer(query, top_k=top_k)
        except Exception:
            logger.exception("Crew failed; falling back to the plain query engine")

    from rag.retrieval import answer_question as plain_answer

    return plain_answer(query, top_k=top_k)
