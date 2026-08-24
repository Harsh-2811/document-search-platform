# RAGAs Evaluation Results

**Questions:** 8 · **Judge:** local `llama3.2:3b` via Ollama · **Embeddings:** `nomic-embed-text`

## Aggregate

| Metric | Score | Scored |
|---|---|---|
| faithfulness | 1.0 | 3/8 |
| answer_relevancy | 0.6779 | 1/8 |

## Per question

| Question | Faithfulness | Answer relevancy | Gen time |
|---|---|---|---|
| What bulk discounts does the supplier offer? | 1.000 | n/a | 26.2s |
| What is the return policy on items? | 1.000 | n/a | 3.7s |
| How many years of experience does the candidate have? | n/a | n/a | 4.4s |
| Which product category had the highest revenue share, and wh | 1.000 | 0.678 | 6.7s |
| How were sales split across channels? | n/a | n/a | 4.0s |
| Which destinations are recommended for autumn travel? | n/a | n/a | 10.0s |
| What is the total built-up area of the Aspenwood Residence? | n/a | n/a | 12.1s |
| Free shipping applies above what order value, and until when | n/a | n/a | 10.0s |

## Why some questions show n/a

The judge is `llama3.2:3b` running on CPU — the same small local model
the pipeline uses. Two things go wrong at that size, and RAGAs records
both as NaN rather than failing the run:

- **Timeouts.** Faithfulness decomposes an answer into individual claims
  and judges each one, so a long-context question is many sequential
  CPU generations, not one call.
- **Unparseable output.** RAGAs asks for strict JSON; a 3b model does not
  always produce it.

Both are limits of the *judge*, not of the pipeline being judged. A
larger judge model would score more of the set; the answers themselves
are unchanged either way, and are recorded in `answers.json`.

> Scores are recorded, not tuned. The brief asks that evaluation runs and
> produces metrics; improving them is a separate exercise.
