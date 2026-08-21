"""Phase 3 / Block B — the agentic layer: retrieve, then answer.

A CrewAI crew replaces the single lookup in `rag.retrieval`. Two agents:
a researcher that decides what to search for and calls the retrieval tool,
and a writer that turns the retrieved passages into a grounded answer.

`answer_question` here mirrors the signature in `rag.retrieval`, so the API
can swap between them by changing one import, not the view.

Sizing note: every agent turn is a full LLM generation, and this host runs
`llama3.2:3b` on CPU with weights on a spinning disk. The crew is kept
deliberately small — two agents, low `max_iter`, one tool — because each
extra turn costs a minute or more.
"""

from __future__ import annotations

import os
from typing import Any

from rag.retrieval import Answer, DEFAULT_TOP_K

MAX_RETRIES = 2

# CrewAI phones home by default; on a slow link that stalls startup.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")


def build_retrieval_tool(top_k: int = DEFAULT_TOP_K):
    """B4 — expose the LlamaIndex retriever as a tool the crew can call.

    Returns `(tool, retriever)`. The retriever is handed back so the caller
    can read `retriever.seen_nodes` afterwards and report exactly which
    passages the crew looked at.

    Wraps a *retriever*, not a query engine, deliberately. Wrapping the query
    engine would make every tool call run its own synthesis step — a full LLM
    round-trip — so a retrieve-then-answer crew would pay two generations per
    question instead of one. On this CPU-only host that difference is minutes.
    """
    from crewai_tools import LlamaIndexTool
    from llama_index.core.tools import RetrieverTool, ToolMetadata

    from rag.retrieval import get_retriever

    retriever = get_retriever(top_k=top_k)
    retriever_tool = RetrieverTool(
        retriever=retriever,
        metadata=ToolMetadata(
            name="document_search",
            description=(
                "Search the indexed document library (a consulting resume, an "
                "office-supply product catalogue, a community newsletter, and "
                "a house floor plan). Input: a natural-language search query. "
                "Returns the most relevant passages verbatim, with the source "
                "filename. Use this for every factual claim — do not answer "
                "from your own knowledge."
            ),
        ),
    )
    return LlamaIndexTool.from_tool(retriever_tool), retriever


def get_retrieval_tool(top_k: int = DEFAULT_TOP_K):
    """Just the tool, for callers that don't need the retriever handle."""
    return build_retrieval_tool(top_k=top_k)[0]


def _build_llm():
    from crewai import LLM

    from rag.config import get_config
    from rag.retrieval import LLM_TIMEOUT_SECONDS

    config = get_config()
    # `ollama/` is a supported native provider, but CrewAI serves it through
    # its OpenAI-shaped completion handler — so only OpenAI-valid kwargs get
    # through. Passing Ollama-specific options here (keep_alive, num_ctx)
    # raises `Completions.create() got an unexpected keyword argument`.
    # Model residency is handled server-side instead; see KEEP_ALIVE in
    # rag.retrieval, which the plain path sets on the Ollama API directly.
    return LLM(
        model=f"ollama/{config.chat_model}",
        base_url=config.ollama_base_url,
        timeout=LLM_TIMEOUT_SECONDS,
    )


def build_crew(top_k: int = DEFAULT_TOP_K):
    """B5 — a two-agent crew: research the question, then answer it.

    Returns `(crew, retriever)`.
    """
    from crewai import Agent, Crew, Process, Task

    tool, retriever = build_retrieval_tool(top_k=top_k)
    llm = _build_llm()

    researcher = Agent(
        role="Document researcher",
        goal="Find the passages in the document library that answer: {question}",
        backstory=(
            "You are precise and literal. You never answer from memory — you "
            "search the library and report what the documents actually say, "
            "quoting the relevant lines and naming their source file."
        ),
        tools=[tool],
        llm=llm,
        allow_delegation=False,
        # Each iteration is a full generation on CPU; cap it hard.
        max_iter=3,
        verbose=False,
    )

    writer = Agent(
        role="Answer writer",
        goal="Write a short, correct answer grounded only in the retrieved passages",
        backstory=(
            "You turn retrieved passages into a direct answer. You quote "
            "figures exactly, name the source document, and if the passages "
            "do not contain the answer you say so plainly rather than guess."
        ),
        # Given the tool as well: if the researcher's hand-off loses the
        # passages, the writer can recover instead of answering "I don't know"
        # while the correct chunk sits one call away.
        tools=[tool],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
        verbose=False,
    )

    research_task = Task(
        description=(
            "Search the document library for information answering this "
            "question:\n\n{question}\n\n"
            "Call the document_search tool. Then COPY OUT the full text of "
            "every passage it returned, word for word, along with each "
            "passage's source filename. Do not summarise, shorten, or "
            "interpret them — your entire job is to reproduce the retrieved "
            "text so the next agent can read it."
        ),
        expected_output=(
            "The complete verbatim text of the retrieved passages, each "
            "preceded by its source filename."
        ),
        agent=researcher,
    )

    answer_task = Task(
        description=(
            "The context above contains passages copied verbatim out of the "
            "document library. Read them and answer this question:\n\n"
            "{question}\n\n"
            "The answer is almost certainly present in those passages — read "
            "them carefully before concluding otherwise. Quote figures "
            "exactly as they appear and name the source document. Only say "
            "you don't know if the passages genuinely do not address the "
            "question; if they are empty, call document_search yourself."
        ),
        expected_output="A short grounded answer naming its source document.",
        agent=writer,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, answer_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew, retriever


def _sources_from(retriever) -> list[dict[str, Any]]:
    """One entry per document the crew actually retrieved, best score first."""
    best: dict[str, dict[str, Any]] = {}
    for node in retriever.seen_nodes:
        filename = node.metadata.get("filename", "")
        score = float(node.score) if node.score is not None else 0.0
        if filename not in best or score > best[filename]["score"]:
            best[filename] = {
                "filename": filename,
                "heading": node.metadata.get("heading", ""),
                "chunk_index": node.metadata.get("chunk_index"),
                "score": round(score, 4),
            }
    return sorted(best.values(), key=lambda s: s["score"], reverse=True)


def answer_question(question: str, *, top_k: int = DEFAULT_TOP_K) -> Answer:
    """B5 — answer via the crew. Drop-in for `rag.retrieval.answer_question`."""
    crew, retriever = build_crew(top_k=top_k)
    result = crew.kickoff(inputs={"question": question})
    return Answer(text=str(result).strip(), sources=_sources_from(retriever))
