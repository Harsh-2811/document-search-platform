# Document Search Platform

Ask questions in plain English about a folder of PDFs and get an answer that
cites the documents it came from. Everything runs locally — no OpenAI key, no
data leaving the machine.

```
"What bulk discounts does the supplier offer?"

  -> "10% off orders over $500, and 15% off orders over $1,000."
     Sources: product_catalog.pdf — "Breakroom & Supplies" (score 0.669)
```

## What's implemented

- **Ingestion pipeline** — Docling PDF parsing, heading-aware chunking, Ollama
embeddings, pgvector indexing, all behind one management command.
- **Google Drive fetch** — paste a share link, get a verified PDF in `data/`.
- **Vector search** — Postgres 17 + pgvector with an HNSW cosine index.
- **RAG answering** — LlamaIndex retrieval + synthesis over a custom retriever.
- **REST API** — `POST /api/chat/` returning an answer plus its source documents.
- **Multi-turn conversations** — prior turns resolve pronouns without an extra LLM call.
- **Multi-agent path** — a CrewAI two-agent crew, selectable by env var.
- **Chat frontend** — OpenWebUI with a custom Pipe, installed at DB level.
- **Tracing** — Arize Phoenix capturing every retrieval, embedding, and LLM call.
- **Externalized prompts** — plain files, editable without touching code.
- **Evaluation** — a RAGAs harness with a checkpointing runner and saved results.



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
| CrewAI                     | multi-agent path, selectable by env var           | `rag/agents.py`    |
| OpenWebUI                  | chat frontend                                     | `:3000`            |
| Arize Phoenix              | LLM tracing UI                                    | `:6006`            |


---



# Setup



## Prerequisites


| Requirement                    | Notes                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| **Docker Desktop**             | WSL2 backend on Windows. Four containers run here.                                                           |
| **Ollama**, installed natively | **Not** a compose service — see the design note above.                                                       |
| **~8 GB free RAM**             | The model holds ~3 GB, the Docker VM ~3.5 GB. Memory pressure kills the VM and takes all containers with it. |
| **~15 GB free disk**           | The web image is ~4.2 GB; models are another ~3 GB.                                                          |




## 1. Pull the models

On the **host**, not in a container:

```bash
ollama pull llama3.2:3b        # chat / answer generation
ollama pull nomic-embed-text   # embeddings, 768-dim
```

Verify both are present:

```bash
ollama list
```



## 2. Make Ollama reachable from Docker

**This step is required, and skipping it makes every request fail with** `503`**.**

Ollama binds to `127.0.0.1` by default. Containers reach the host through the
Docker gateway, not loopback, so a loopback-only Ollama **refuses every container
connection** — the API then fails at its first step (embedding the question) with
`Connection refused`.

Set `OLLAMA_HOST` **persistently**, so it survives an Ollama or machine restart:

```powershell
# Windows (PowerShell) — User scope, not just this session
[Environment]::SetEnvironmentVariable('OLLAMA_HOST','0.0.0.0','User')
```

```bash
# Linux / macOS — add to your shell profile or systemd unit
export OLLAMA_HOST=0.0.0.0
```

Then restart Ollama and confirm it is no longer loopback-only:

```powershell
netstat -ano | Select-String ":11434"
# want:  TCP  0.0.0.0:11434  LISTENING
# not:   TCP  127.0.0.1:11434  LISTENING
```

> `0.0.0.0` exposes Ollama to your whole network. That is the standard fix and is
> fine on a trusted network; bind to the Docker gateway address specifically if
> you need it tighter.



## 3. Configure

```bash
cp .env.example .env
```

Edit `.env` — every value is documented inline there. At minimum set
`DJANGO_SECRET_KEY` and `DB_PASSWORD`. See [Configuration](#configuration) below
for the full reference.

## 4. Start the stack

```bash
docker compose up -d
docker compose ps          # all four should be Up; db and openwebui healthy
```



## 5. Create the schema

```bash
docker compose exec web python manage.py migrate
```

This enables the `vector` extension, creates `documents_document` and
`documents_chunk`, and builds the HNSW index.

## 6. Add documents and index them

Drop PDFs into `data/`, or pull them from Drive:

```bash
docker compose exec web python manage.py fetch_drive <drive-share-url>
docker compose exec web python manage.py ingest_docs
```



## 7. Verify

```bash
# API answers a question
curl -s -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the return policy?"}'

# end-to-end smoke test (3 questions, checks the right document ranks first)
docker compose exec web python scripts/b3_smoketest.py
```

Then open **[http://localhost:3000](http://localhost:3000)**, pick
**Document Search (RAG)** from the model dropdown, and ask a question.

> Commands below are written as `python manage.py ...`. Run them inside the
> container with `docker compose exec web python manage.py ...`.

---



# Commands



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

---



# API

## Interactive documentation

The full OpenAPI 3.0 specification is at [openapi.yaml](openapi.yaml), served
live by the app:

| URL                                                              | What it is                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------ |
| **[http://localhost:8000/api/docs/](http://localhost:8000/api/docs/)**     | Swagger UI — browse the schema and send test requests. |

The spec documents every request field with its validation bounds, all three
response codes with real captured examples, and the error shapes — including the
fact that errors inside `history` are keyed by index string rather than returned
as an array.

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

Multi-turn example — the pronoun resolves from prior turns:

```bash
curl -s -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question":"And how many bathrooms does it have?",
    "history":[
      {"role":"user","content":"How many bedrooms does the home plan have?"},
      {"role":"assistant","content":"The home plan has 3 bedrooms."}
    ]
  }'
```

---



# Configuration

Everything lives in `.env` (copy from `.env.example`, which documents each
value). `rag/` reads `os.environ` directly and never imports Django settings, so
a management command and a standalone script see identical configuration.

### Django


| Variable               | Default                                                | Description                                                           |
| ---------------------- | ------------------------------------------------------ | --------------------------------------------------------------------- |
| `DJANGO_SECRET_KEY`    | —                                                      | **Set this.** Generate a fresh one per environment.                   |
| `DJANGO_DEBUG`         | `True`                                                 | Turn off outside local development.                                   |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1,0.0.0.0,host.docker.internal,web` | `host.docker.internal` **is required** or container calls return 400. |




### Database


| Variable      | Default      | Description                                                 |
| ------------- | ------------ | ----------------------------------------------------------- |
| `DB_NAME`     | `documentdb` | Database name.                                              |
| `DB_USER`     | `postgres`   | Database user.                                              |
| `DB_PASSWORD` | —            | **Set this.**                                               |
| `DB_HOST`     | `localhost`  | Compose overrides to `db` inside the network.               |
| `DB_PORT`     | `5433`       | Host port. The container always listens on 5432 internally. |




### Ollama and the RAG pipeline


| Variable                     | Default                  | Description                                                                                                              |
| ---------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `OLLAMA_BASE_URL`            | `http://localhost:11434` | Compose overrides to `http://host.docker.internal:11434`.                                                                |
| `OLLAMA_CHAT_MODEL`          | `llama3.2:3b`            | Model that writes answers.                                                                                               |
| `OLLAMA_EMBED_MODEL`         | `nomic-embed-text`       | Must match what was used at ingest — querying with a different embedding model returns plausible nonsense, not an error. |
| `EMBEDDING_DIM`              | `768`                    | Must match the embedding model and the migration.                                                                        |
| `RAG_USE_CREW`               | `false`                  | `true` routes answering through the CrewAI crew instead of the plain engine.                                             |
| `OLLAMA_NUM_GPU`             | unset                    | Layers to offload to the GPU. **Set to** `0` **to force CPU-only.**                                                      |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006`  | Compose overrides to `http://phoenix:6006`.                                                                              |




### Two settings that look wrong and aren't

`OLLAMA_NUM_GPU=0` **is not a typo.** A GPU too small to hold the model is worse
than no GPU: Ollama offloads only the layers that fit and every token then
crosses PCIe. Measured here (GT 710, 2GB, holding 23% of `llama3.2:3b`), prompt
processing ran at 7.8 tok/s; forced to CPU the same prompt ran at 108 tok/s —
roughly ten times faster, and end-to-end answers went from ~70s to 4–11s. Leave
it unset on a machine with a GPU that fits the whole model.

`RAG_USE_CREW=false` **is also deliberate.** The crew is built and wired, but the
plain engine is faster on CPU-only hardware. Either path returns a grounded
answer with sources, and the crew falls back to the plain engine on any error, so
the API always responds.

### Tuning constants

These live in `rag/retrieval.py` rather than `.env`, because they are properties
of the host rather than of a deployment:


| Constant              | Value   | Why                                                                                                    |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------ |
| `CONTEXT_WINDOW`      | `8192`  | `llama3.2:3b` advertises 128k; loading it there allocates an ~18GB KV cache on a 16GB host.            |
| `KEEP_ALIVE`          | `"-1m"` | Keeps the model resident. Models live on a SATA HDD, so an eviction costs minutes to reload.           |
| `LLM_TIMEOUT_SECONDS` | `600.0` | CPU generation is slow; a short timeout produces confusing mid-generation failures rather than a wait. |
| `DEFAULT_TOP_K`       | `4`     | Chunks retrieved per question.                                                                         |




### Prompts

Prompts are files, editable without touching Python:

```
prompts/qa_prompt.txt     the answer-synthesis template
prompts/agents.yaml       CrewAI agent roles, goals, backstories, tasks
```

They are read once per process (`rag/prompts.py`, `lru_cache`d), so an edit takes
effect on the next restart:

```bash
docker compose restart web
```

---



# Deployment



## Services

`docker-compose.yml` defines four services. Ollama runs natively on the host.


| Service     | Container             | Port(s)        | Depends on   |
| ----------- | --------------------- | -------------- | ------------ |
| `db`        | `docsearch-db`        | `5433:5432`    | —            |
| `web`       | `docsearch-web`       | `8000:8000`    | `db` healthy |
| `openwebui` | `docsearch-openwebui` | `3000:8080`    | —            |
| `phoenix`   | `docsearch-phoenix`   | `6006`, `4317` | —            |


All four use `restart: unless-stopped`. `db` has a `pg_isready` healthcheck and
`web` waits for it, so migrations never race a cold database.

## Volumes


| Volume      | Holds                                         | Losing it means              |
| ----------- | --------------------------------------------- | ---------------------------- |
| `pgdata`    | Postgres data — documents, chunks, embeddings | Re-ingesting everything      |
| `openwebui` | OpenWebUI's SQLite, incl. the installed Pipe  | Reinstalling the Pipe        |
| `phoenix`   | Collected traces                              | Losing trace history         |
| `hfcache`   | Docling's HuggingFace models                  | Re-downloading on next parse |


`hfcache` matters more than it looks: without it, Docling re-downloads its
layout and table models every time the container is recreated.

## Everyday operations

```bash
docker compose up -d                    # start
docker compose down                     # stop, keep volumes
docker compose ps                       # status
docker compose logs -f web              # follow API logs
docker compose restart web              # reload after a prompt or .env change
```



## Rebuilding the image

Required after any change to `requirements.txt` or the `Dockerfile`:

```bash
docker compose build web
docker compose up -d web
```

The rebuild takes roughly 15–30 minutes — the dependency tree (torch, docling,
crewai, llama-index) is ~4.2 GB.

> Install dependencies by rebuilding, **not** by `pip install` into a running
> container. A container-level install is wiped the next time the container is
> recreated, which reverts pinned versions silently.



## Backup and restore

```bash
# Back up the vector store
docker exec docsearch-db pg_dump -U postgres documentdb > backup.sql

# Restore
docker exec -i docsearch-db psql -U postgres documentdb < backup.sql
```

Source PDFs live in `data/` on the host and are covered by ordinary file backups.
Embeddings can always be regenerated from them with `ingest_docs`.

## Resource notes

The stack is memory-sensitive. On a 16 GB host, the model (~3 GB) plus the Docker
VM (~3.5 GB) plus a browser and an IDE is enough to exhaust RAM — and when
Windows reclaims memory, WSL is the largest target, so the Docker VM is killed
and every container goes down at once (all exiting `137`). If containers die
together for no obvious reason, check free memory before anything else.

Disk fills mainly through the build cache. `docker builder prune -a -f` reclaims
it without touching any image.

---



# Observability

Every retrieval, embedding, and LLM call is traced. Open
**[http://localhost:6006](http://localhost:6006)** → project `document-search`
for the span tree.

A single question produces spans like:

```
retriever  ChunkRetriever._retrieve            4133ms
chain      TokenTextSplitter.split_text           1ms
llm        Ollama.chat                        86886ms
chain      CompactAndRefine.synthesize        86894ms
chain      RetrieverQueryEngine.query         91030ms
```

This is the fastest way to tell retrieval cost from generation cost — a slow
`Ollama.chat` span is the model, a slow `ChunkRetriever` span is the database or
the embedding call.

Tracing is wired in `rag/tracing.py` and started from `DocumentsConfig.ready()`.
It swallows its own errors by design, so an unreachable Phoenix can slow nothing
and take nothing down.

---



# Evaluation

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

**Eval set:** 8 questions across 5 documents, ground truth read from the actual
chunks. All 8 generated answers were verified correct by hand.

**Latest scores:** faithfulness **1.0** (3 of 8 scored), answer relevancy
**0.6779** (1 of 8 scored). Every mean is written alongside its denominator,
since a local 3b judge cannot always produce the strict JSON RAGAs requires and
those questions are recorded as `n/a` rather than guessed at.

Three separate phases, because each fails differently and all three are slow:

- **Generation** is the expensive half (~10s per question on CPU).
- **Scoring** is the fragile half — RAGAs defaults to OpenAI and has a history of
version churn, so a config problem here must not cost the answers.
- **Summarising** is neither, so it doesn't require re-running the other two.

Scoring checkpoints to `results.json` after every question, so a run interrupted
at question 5 keeps the first four judgements, and `--summarize-only` turns
whatever survived into a report.

---



# Troubleshooting


| Symptom                                                                                     | Cause and fix                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Every request returns `503`, logs show `Connection refused` to `host.docker.internal:11434` | Ollama is bound to loopback. Set `OLLAMA_HOST=0.0.0.0` persistently and restart Ollama. Confirm with `netstat -ano                                                                                                           |
| All containers exit `137` at once                                                           | Host out of RAM — WSL is the largest reclaim target. Free memory, then `docker compose up -d`.                                                                                                                               |
| Every request takes ~70s                                                                    | Partial GPU offload. Set `OLLAMA_NUM_GPU=0`.                                                                                                                                                                                 |
| One request takes ~90s, then fast                                                           | Cold model load off the HDD. `ollama ps` returning nothing is the tell. Not a regression.                                                                                                                                    |
| `400 DisallowedHost`                                                                        | `host.docker.internal` missing from `DJANGO_ALLOWED_HOSTS`.                                                                                                                                                                  |
| Answers contain `&`                                                                         | Docling HTML-escapes its Markdown. Handled in `rag/ingest.py`; older rows need a backfill.                                                                                                                                   |
| OpenWebUI stuck `unhealthy`                                                                 | It's downloading sentence-transformers. `RAG_EMBEDDING_ENGINE=ollama` + `HF_HUB_OFFLINE=1` prevent it.                                                                                                                       |
| Answers cite the wrong document                                                             | Embedding model changed since ingest. Re-ingest.                                                                                                                                                                             |
| `No module named 'langchain_community.chat_models.vertexai'`                                | ragas resolved to 0.4.x, which imports a module langchain-community 0.4 removed. `requirements.txt` pins 0.2.15 — rebuild the image rather than `pip install` into a running container, or the next `compose up` reverts it. |
| Empty reply on `localhost:8000`                                                             | Docker sometimes binds the published port IPv6-only. Use `127.0.0.1` instead.                                                                                                                                                |


