# Document Search Platform

Ask questions in plain English about a folder of PDFs and get an answer that
cites the documents it came from. Everything runs locally — no OpenAI key, no
data leaving the machine.

```
"What bulk discounts does the supplier offer?"

  -> "10% off orders over $500, and 15% off orders over $1,000."
     Sources: product_catalog.pdf — "Breakroom & Supplies" (score 0.669)
```

## How it works

Two phases. **Ingestion** happens offline in a management command; **answering**
happens per request.

```
INGEST   data/*.pdf
            |  Docling            parse PDF -> Markdown (keeps headings/tables)
            |  chunk              split on headings, carry heading in metadata
            |  nomic-embed-text   768-dim vector per chunk
            v
         Postgres 17 + pgvector      documents_document / documents_chunk
                                     HNSW index, cosine distance
ANSWER   question
            |  nomic-embed-text   embed the question
            |  pgvector           top-k nearest chunks (<=> cosine)
            |  llama3.2:3b        stuff chunks into one prompt, generate
            v
         {"answer": "...", "sources": [...]}      -> traced to Phoenix
```

The retrieved chunks are the *only* material the model is given, which is what
keeps answers grounded and makes the `sources` list meaningful.

### Design notes worth knowing

- `rag/` **never imports Django.** Django owns the schema and writes chunks;
`rag/` reads the same table over psycopg. `rag/test_django_free.py` enforces
the boundary. This is why the retriever is custom rather than LlamaIndex's
`PGVectorStore`, which insists on owning its own schema.
- **Ingestion is a command, not a view.** Embedding a document takes far longer
than any HTTP request should.
- **Ollama runs natively on the host, not in compose.** The Ollama image bundles
~5GB of CUDA/ROCm runtimes this host can't use, and models live on `D:`.
Containers reach it via `host.docker.internal`.
- **CPU beats a small GPU here.** See `OLLAMA_NUM_GPU` below.



## Stack


| Piece                      | What it does                                      | Where              |
| -------------------------- | ------------------------------------------------- | ------------------ |
| Django 6.1 + DRF           | REST API                                          | `:8000`            |
| Postgres 17 + pgvector 0.8 | chunks + vector search                            | `:5433`            |
| Ollama                     | `llama3.2:3b` chat, `nomic-embed-text` embeddings | `:11434` (host)    |
| Docling                    | PDF -> Markdown                                   | in `rag/ingest.py` |
| LlamaIndex                 | retrieval + synthesis orchestration               | `rag/retrieval.py` |
| CrewAI                     | multi-agent path (built, off by default)          | `rag/agents.py`    |
| OpenWebUI                  | chat frontend                                     | `:3000`            |
| Arize Phoenix              | LLM tracing UI                                    | `:6006`            |
| RAGAs                      | offline evaluation                                | `eval/`            |




## Setup

**Prerequisites:** Docker Desktop, and Ollama installed on the host.

```bash
# 1. Pull the models (on the host, not in a container)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Configure
cp .env.example .env          # then edit it — see the comments in that file

# 3. Start everything
docker compose up -d

# 4. Create the schema
docker compose exec web python manage.py migrate

# 5. Add documents and index them
docker compose exec web python manage.py ingest_docs
```

Open **[http://localhost:3000](http://localhost:3000)**, pick **Document Search (RAG)** from the model
dropdown, and ask a question.

> Commands below are written as `python manage.py ...`. Run them inside the
> container with `docker compose exec web python manage.py ...`.



## Commands



### `ingest_docs` — parse, chunk, embed, index

```bash
python manage.py ingest_docs                          # every PDF in data/
python manage.py ingest_docs data/resume.pdf          # one file
python manage.py ingest_docs --doc-type resume        # tag what you ingest
python manage.py ingest_docs --skip-existing          # only new files
```


| Param             | Type           | Default                  | Description                                                 |
| ----------------- | -------------- | ------------------------ | ----------------------------------------------------------- |
| `paths`           | positional, 0+ | every `*.pdf` in `data/` | PDF files to ingest.                                        |
| `--doc-type`      | string         | `""`                     | Category stored on each `Document`, e.g. `resume`.          |
| `--skip-existing` | flag           | off                      | Leave already-ingested files alone instead of re-ingesting. |


Re-ingesting a file **replaces** its chunks rather than appending, so stale text
can't linger in results. One transaction per document: a mid-way failure leaves
the catalog consistent, not half-indexed.

> `--doc-type` applies to *every* file in the run. To label files differently,
> run the command once per type.



### `fetch_drive` — download a shared PDF from Google Drive

Paste the share link. Nothing to configure first.

```bash
python manage.py fetch_drive https://drive.google.com/file/d/1AbCdEf.../view
python manage.py fetch_drive 1AbCdEf... --name resume        # bare id also works
python manage.py fetch_drive <url> --name resume --overwrite
```


| Param         | Type                 | Default             | Description                                                                           |
| ------------- | -------------------- | ------------------- | ------------------------------------------------------------------------------------- |
| `url`         | positional, required | —                   | Drive share link, or a bare file id.                                                  |
| `--name`      | string               | whatever Drive says | Save as `<name>.pdf` instead of Drive's own name. The `.pdf` is added if you omit it. |
| `--overwrite` | flag                 | off                 | Replace the file if it already exists.                                                |


The filename comes from Drive's `Content-Disposition` header, so the saved file
keeps the name it has in Drive. That name is untrusted input from a public URL,
so it's sanitised before use — path separators, traversal segments, and control
characters are stripped, and the result is forced to `.pdf`.

Accepted link shapes:

```
https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
https://drive.google.com/open?id=<FILE_ID>
https://drive.google.com/uc?export=download&id=<FILE_ID>
<FILE_ID>
```

The file must be shared as **"Anyone with the link" (Viewer)** and be a real
uploaded PDF, not a Google Doc. Drive answers most failures with HTTP 200 and an
HTML body, so the download is verified to start with `%PDF` before it's kept —
and it's streamed to a `.part` file first, so a failed fetch never leaves
something `ingest_docs` would try to parse.

Fetching is separate from ingesting on purpose: download, look at the PDF, then
ingest. Chaining them would turn a bad download into bad chunks silently.

### Standard Django commands

```bash
python manage.py migrate                 # create/update the schema
python manage.py createsuperuser         # then browse /admin to inspect chunks
python manage.py shell
```



## API



### `POST /api/chat/`

```bash
curl -s -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the return policy?"}'
```


| Field      | Type                      | Required | Description                                                                      |
| ---------- | ------------------------- | -------- | -------------------------------------------------------------------------------- |
| `question` | string, 1–2000 chars      | yes      | The question to answer.                                                          |
| `history`  | list of `{role, content}` | no       | Prior turns, for pronoun resolution. `role` is `user`, `assistant`, or `system`. |
| `top_k`    | int, 1–20                 | no       | Chunks to retrieve. Default 4.                                                   |


Returns `200` with `{"answer": "...", "sources": [{filename, heading, chunk_index, score}]}`.  
`400` on invalid input, `503` if generation fails.



## Evaluation

```bash
docker exec docsearch-web python -u eval/run_ragas.py                  # generate + score
docker exec docsearch-web python -u eval/run_ragas.py --generate-only  # answers only
docker exec docsearch-web python -u eval/run_ragas.py --score-only     # reuse answers.json
docker exec docsearch-web python -u eval/run_ragas.py --summarize-only # rebuild the report
```


| Param              | Description                                                                                                 |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| `--generate-only`  | Ask every question in `eval/eval_set.json`, save answers + retrieved contexts to `eval/answers.json`, stop. |
| `--score-only`     | Skip generation and score the saved `eval/answers.json`.                                                    |
| `--summarize-only` | Rebuild `RESULTS.md` from an existing `results.json`. No judging, so it's instant.                          |


Scored with RAGAs **faithfulness** (is the answer supported by the retrieved
text?) and **answer relevancy**, judged by the same local `llama3.2:3b` — no API
key, no network. Results land in `eval/results.json` and `eval/RESULTS.md`.

Three separate phases, because each one fails differently and all three are slow:

- **Generation** is the expensive half (~10s per question on CPU).
- **Scoring** is the fragile half — RAGAs defaults to OpenAI and has a history of
version churn, so a config problem here must not cost the answers.
- **Summarising** is neither, so it shouldn't require re-running the other two.

Scoring checkpoints to `results.json` after every question. A run interrupted at
question 5 keeps the first four judgements rather than losing twenty minutes of
CPU time, and `--summarize-only` will turn whatever survived into a report.

**Expect gaps in the scores.** `llama3.2:3b` is a weak judge: RAGAs demands
strict JSON and a 3b model doesn't always produce it, and faithfulness
decomposes each answer into individual claims to judge separately, so long
questions time out. Both are limits of the judge, not of the pipeline being
judged — every mean is reported with its denominator so a partial result can't
read as a complete one.

## Observability

Every retrieval, embedding, and LLM call is traced. Open
**[http://localhost:6006](http://localhost:6006)** → project `document-search` for the span tree.

This is the fastest way to tell retrieval cost from generation cost — a slow
`Ollama.chat` span is the model, a slow `ChunkRetriever` span is the database or
the embedding call.

## Configuration

Everything lives in `.env` (copy from `.env.example`, which documents each
value). The ones that change behaviour rather than addresses:


| Variable             | Default            | Description                                                                                                              |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `OLLAMA_CHAT_MODEL`  | `llama3.2:3b`      | Model that writes answers.                                                                                               |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Must match what was used at ingest — querying with a different embedding model returns plausible nonsense, not an error. |
| `EMBEDDING_DIM`      | `768`              | Must match the embedding model and the migration.                                                                        |
| `RAG_USE_CREW`       | `false`            | `true` routes answering through the CrewAI crew instead of the plain engine.                                             |
| `OLLAMA_NUM_GPU`     | unset              | Layers to offload to the GPU. **Set to** `0` **to force CPU-only.**                                                      |


`OLLAMA_NUM_GPU=0` **is not a typo.** A GPU too small to hold the model is worse
than no GPU: Ollama offloads only the layers that fit and every token then
crosses PCIe. Measured here (GT 710, 2GB, holding 23% of `llama3.2:3b`), prompt
processing ran at 7.8 tok/s; forced to CPU the same prompt ran at 108 tok/s.
Roughly ten times faster on the CPU. Leave it unset on a machine with a real GPU.

`RAG_USE_CREW=false` **is also deliberate.** The crew is built and wired, but on
this CPU host it took ~200s per question and answered "I don't know" about a
bulk discount while holding the chunk that stated it — `llama3.2:3b` isn't a
strong enough tool-caller to drive a multi-agent loop. It falls back to the plain
engine on any error, so the API always returns a grounded answer. On GPU
hardware, flip the flag and re-measure.

## Troubleshooting


| Symptom                                                      | Cause                                                                                                                                                                                                                        |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Every request takes ~70s                                     | Partial GPU offload. Set `OLLAMA_NUM_GPU=0`.                                                                                                                                                                                 |
| One request takes ~90s, then fast                            | Cold model load off the HDD. `ollama ps` returning nothing is the tell. Not a regression.                                                                                                                                    |
| `400 DisallowedHost`                                         | `host.docker.internal` missing from `DJANGO_ALLOWED_HOSTS`.                                                                                                                                                                  |
| Answers contain `&`                                          | Docling HTML-escapes its Markdown. Handled in `rag/ingest.py`; older rows need a backfill.                                                                                                                                   |
| OpenWebUI stuck `unhealthy`                                  | It's downloading sentence-transformers. `RAG_EMBEDDING_ENGINE=ollama` + `HF_HUB_OFFLINE=1` prevent it.                                                                                                                       |
| Answers cite the wrong document                              | Embedding model changed since ingest. Re-ingest.                                                                                                                                                                             |
| `No module named 'langchain_community.chat_models.vertexai'` | ragas resolved to 0.4.x, which imports a module langchain-community 0.4 removed. `requirements.txt` pins 0.2.15 — rebuild the image rather than `pip install` into a running container, or the next `compose up` reverts it. |




## Layout

```
config/         Django settings, URLs
documents/      models (Document, Chunk), DRF view, serializers, commands
rag/            framework-agnostic pipeline — imports no Django
  ingest.py       parse, chunk, embed
  retrieval.py    custom pgvector retriever + LlamaIndex query engine
  pipeline.py     the single entry point the API calls
  agents.py       CrewAI crew (opt-in)
  drive_download.py
  tracing.py      Phoenix/OpenTelemetry setup
prompts/        qa_prompt.txt, agents.yaml — editable without touching code
eval/           RAGAs eval set + runner + results
openwebui/      the custom Pipe that connects OpenWebUI to this API
data/           source PDFs
scripts/        b3_smoketest.py — quick end-to-end check
```



