"""E2 — evaluate the RAG pipeline with RAGAs, judged by the local Ollama model.

Run inside the web container:

    docker exec docsearch-web python eval/run_ragas.py

Two phases, deliberately separated:

  1. **Generate** — ask every eval question, save answers + retrieved contexts
     to `eval/answers.json`.
  2. **Score** — run RAGAs metrics over that file.

They are split because generation is the expensive half (~10s per question on
CPU) and RAGAs is the fragile half: it defaults to OpenAI and has a history of
version churn. If scoring explodes, the generated answers survive and phase 2
can be re-run with `--score-only` without paying for generation again.

RAGAs is pointed at the same local models the pipeline uses, so the eval needs
no API key and no network.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Django has to be configured before anything imports the ORM.
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET = EVAL_DIR / "eval_set.json"
ANSWERS = EVAL_DIR / "answers.json"
RESULTS = EVAL_DIR / "results.json"
SUMMARY = EVAL_DIR / "RESULTS.md"


def generate() -> list[dict]:
    """Phase 1 — ask each question, capture the answer and its contexts."""
    from rag.retrieval import get_query_engine

    questions = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    engine = get_query_engine()

    rows = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"[{i}/{len(questions)}] {q}", flush=True)
        started = time.time()

        response = engine.query(q)
        # RAGAs scores the answer against the text actually retrieved, so the
        # node text is what matters here — not the source metadata the API
        # returns to callers.
        contexts = [n.get_content() for n in response.source_nodes]

        elapsed = time.time() - started
        answer = str(response).strip()
        print(f"      {elapsed:5.1f}s  {answer[:90]}", flush=True)

        rows.append(
            {
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
                "source_doc": item.get("source_doc", ""),
                "seconds": round(elapsed, 1),
            }
        )

    ANSWERS.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(rows)} answers -> {ANSWERS}")
    return rows


def score(rows: list[dict]) -> dict:
    """Phase 2 — RAGAs metrics, judged by the local model."""
    # Pinned to ragas 0.2.x. 0.4.x imports
    # `langchain_community.chat_models.vertexai`, which langchain-community 0.4
    # removed — it fails at import before any eval can run.
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import AnswerRelevancy, Faithfulness
    from ragas.run_config import RunConfig

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # num_gpu=0 for the same reason the pipeline sets it: this host's GPU is
    # too small to hold the model, and partial offload is ~10x slower.
    num_gpu = os.environ.get("OLLAMA_NUM_GPU", "").strip()
    extra = {"num_gpu": int(num_gpu)} if num_gpu else {}

    # num_ctx and keep_alive mirror rag/retrieval.py. Without keep_alive the
    # judge model is evicted after Ollama's 5-minute idle default, and the next
    # call pays a multi-minute reload from the SATA HDD the models live on --
    # which reads as a hang rather than a slow call.
    judge = LangchainLLMWrapper(
        ChatOllama(
            model=chat_model,
            base_url=base_url,
            temperature=0,
            num_ctx=8192,
            keep_alive="-1m",
            **extra,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=embed_model, base_url=base_url)
    )

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in rows
    ]

    # Warm both models before scoring. A cold load off the HDD takes ~90s,
    # which would otherwise land inside the first metric call and blow its
    # timeout -- turning a slow start into a scored NaN.
    print("Warming judge + embedding models...", flush=True)
    started = time.time()
    judge.langchain_llm.invoke("ok")
    embeddings.embeddings.embed_query("ok")
    print(f"  warm in {time.time() - started:.1f}s", flush=True)

    print("\nScoring with RAGAs (faithfulness + answer relevancy)...", flush=True)
    print("Each metric issues several judge calls PER question — expect minutes.\n", flush=True)

    # Scored one metric-sample at a time rather than through `evaluate()`.
    #
    # `evaluate()` returns everything or nothing: it holds all 16 results in
    # memory and hands them over at the end, so a process killed at question 5
    # loses every judgement made so far -- which is exactly what happened
    # (SIGKILL at 8 minutes in, zero output kept). Driving the metrics
    # directly costs the parallelism we already gave up with max_workers=1,
    # and buys a checkpoint after every score.
    #
    # timeout=180: a warm judge call takes ~10s, so this is generous. A long
    # timeout does not rescue a stuck call, it only hides it -- an earlier run
    # sat for 75 minutes because one wedged call had 900s and two retries.
    run_config = RunConfig(timeout=180, max_workers=1, max_retries=1)

    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
    }
    for metric in metrics.values():
        metric.init(run_config)

    per_question: list[dict] = []
    for index, (row, sample) in enumerate(zip(rows, samples), 1):
        scored = {"question": row["question"]}
        for name, metric in metrics.items():
            started = time.time()
            try:
                value = asyncio.run(
                    asyncio.wait_for(
                        metric.single_turn_ascore(sample),
                        timeout=run_config.timeout,
                    )
                )
                # A judge that fails to produce parseable JSON yields NaN
                # rather than raising; treat it the same as a failure.
                scored[name] = None if value != value else round(float(value), 4)
            except Exception as exc:
                scored[name] = None
                scored[f"{name}_error"] = type(exc).__name__

            note = scored[name] if scored[name] is not None else scored.get(f"{name}_error", "n/a")
            print(
                f"[{index}/{len(rows)}] {name:17s} {str(note):10s} "
                f"({time.time() - started:5.1f}s)",
                flush=True,
            )

        per_question.append(scored)
        # Checkpoint after every question, so an interrupted run keeps
        # everything it has already paid for.
        _checkpoint(per_question)

    return per_question


def _checkpoint(per_question: list[dict]) -> None:
    """Write partial scores to disk so progress survives a kill."""
    RESULTS.write_text(
        json.dumps({"per_question": per_question}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_summary(rows: list[dict], per_q: list[dict]) -> None:
    """Persist scores as JSON plus a readable table (E3)."""
    # Coverage is reported alongside every mean. A local 3b judge does not
    # score all 8 reliably -- some calls time out, and some come back as JSON
    # the parser rejects -- so RAGAs records those as NaN and carries on. A
    # mean over 3 of 8 questions is a different claim than a mean over 8, and
    # printing it without the denominator would overstate what was measured.
    scores = {}
    coverage = {}
    for key in ("faithfulness", "answer_relevancy"):
        try:
            vals = [
                r[key] for r in per_q
                if isinstance(r.get(key), (int, float)) and r[key] == r[key]
            ]
            scores[key] = round(sum(vals) / len(vals), 4) if vals else None
            coverage[key] = f"{len(vals)}/{len(rows)}"
        except Exception:
            scores[key] = None
            coverage[key] = f"0/{len(rows)}"

    RESULTS.write_text(
        json.dumps(
            {"aggregate": scores, "coverage": coverage, "per_question": per_q},
            indent=2, default=str, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# RAGAs Evaluation Results",
        "",
        f"**Questions:** {len(rows)} · **Judge:** local `llama3.2:3b` via Ollama · "
        f"**Embeddings:** `nomic-embed-text`",
        "",
        "## Aggregate",
        "",
        "| Metric | Score | Scored |",
        "|---|---|---|",
    ]
    for k, v in scores.items():
        lines.append(
            f"| {k} | {v if v is not None else 'n/a'} | {coverage.get(k, '?')} |"
        )

    lines += ["", "## Per question", "", "| Question | Faithfulness | Answer relevancy | Gen time |", "|---|---|---|---|"]
    for index, r in enumerate(rows):
        # A checkpointed run can hold fewer scores than there are questions;
        # anything not reached shows as n/a rather than shifting the table.
        row = per_q[index] if index < len(per_q) else {}
        fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "n/a"
        lines.append(
            f"| {r['question'][:60]} | {fmt(row.get('faithfulness'))} | "
            f"{fmt(row.get('answer_relevancy'))} | {r['seconds']}s |"
        )

    if any(v != f"{len(rows)}/{len(rows)}" for v in coverage.values()):
        lines += [
            "",
            "## Why some questions show n/a",
            "",
            "The judge is `llama3.2:3b` running on CPU — the same small local model",
            "the pipeline uses. Two things go wrong at that size, and RAGAs records",
            "both as NaN rather than failing the run:",
            "",
            "- **Timeouts.** Faithfulness decomposes an answer into individual claims",
            "  and judges each one, so a long-context question is many sequential",
            "  CPU generations, not one call.",
            "- **Unparseable output.** RAGAs asks for strict JSON; a 3b model does not",
            "  always produce it.",
            "",
            "Both are limits of the *judge*, not of the pipeline being judged. A",
            "larger judge model would score more of the set; the answers themselves",
            "are unchanged either way, and are recorded in `answers.json`.",
        ]

    lines += [
        "",
        "> Scores are recorded, not tuned. The brief asks that evaluation runs and",
        "> produces metrics; improving them is a separate exercise.",
        "",
    ]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULTS} and {SUMMARY}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true",
                        help="Reuse eval/answers.json instead of regenerating.")
    parser.add_argument("--generate-only", action="store_true",
                        help="Stop after generating answers.")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Rebuild RESULTS.md from an existing results.json.")
    args = parser.parse_args()

    if args.summarize_only:
        # Rewriting the report shouldn't cost another scoring run: judging all
        # 8 questions takes ~20 minutes on this host.
        if not (RESULTS.is_file() and ANSWERS.is_file()):
            print(f"Need both {RESULTS} and {ANSWERS}.")
            return 1
        rows = json.loads(ANSWERS.read_text(encoding="utf-8"))
        saved = json.loads(RESULTS.read_text(encoding="utf-8"))
        write_summary(rows, saved.get("per_question", []))
        return 0

    if args.score_only:
        if not ANSWERS.is_file():
            print(f"No {ANSWERS} — run without --score-only first.")
            return 1
        rows = json.loads(ANSWERS.read_text(encoding="utf-8"))
        print(f"Reusing {len(rows)} saved answers.")
    else:
        rows = generate()
        if args.generate_only:
            return 0

    try:
        per_question = score(rows)
    except Exception as exc:
        # Generation is already saved, so this is recoverable: fix the config
        # and re-run with --score-only. Anything already scored is in
        # results.json thanks to the per-question checkpoint.
        print(f"\nRAGAs scoring failed: {type(exc).__name__}: {exc}")
        print(f"Answers are safe in {ANSWERS}; re-run with --score-only after fixing.")
        print(f"Partial scores, if any, are in {RESULTS} — see --summarize-only.")
        return 1

    write_summary(rows, per_question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
