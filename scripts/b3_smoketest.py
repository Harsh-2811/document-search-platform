"""B3 — baseline end-to-end answer test.

Run from inside the web container:
    docker compose exec -T web python /app/scripts/b3_smoketest.py
"""
from __future__ import annotations

from rag.retrieval import answer_question

# Same three questions A9 used to verify retrieval. The bar for B3 is the
# same right document at the top, plus a text answer grounded in it.
QUESTIONS = [
    "Where did the candidate get their MBA?",
    "How many bedrooms does the home plan have?",
    "What is the return policy?",
]


def main() -> int:
    fails = 0
    for q in QUESTIONS:
        a = answer_question(q)
        top = a.sources[0] if a.sources else None
        print(f"Q: {q}")
        print(f"A: {a.text}")
        if top is not None:
            print(
                "TOP SOURCE: "
                f"{top['filename']} chunk#{top['chunk_index']} "
                f"score={top['score']}"
            )
        else:
            print("TOP SOURCE: (none)")
        print("---")
        if top is None:
            fails += 1
    print(f"questions answered: {len(QUESTIONS) - fails}/{len(QUESTIONS)}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
