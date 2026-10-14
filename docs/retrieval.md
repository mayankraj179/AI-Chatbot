# Ingestion and retrieval

## Pipeline

```
content/<path>  ->  parse_file()  ->  chunk_document()  ->  EmbeddingProvider.embed_documents()  ->  documents + chunks (Postgres/pgvector)
```

- **Parsers** (`backend/app/ingestion/parsers.py`): PDF (per page), DOCX (Heading 1–6 / Title styles, plus tables),
  XLSX (one section per sheet; rows keyed by the header row), Markdown (`#` headings, fenced code ignored),
  HTML (`h1`–`h6`, scripts and styles dropped), plain text. Library errors become `ParseError`; files over 25 MB are refused.
- **Chunking** (`chunking.py`): chunks never cross a heading section. Markdown tables are split by rows only, and every
  piece repeats the header and separator lines, so a header never ends up apart from its rows (a "?" in a header cell
  does not trigger a sentence split). Other text uses paragraphs, then sentences, then hard slices to fill
  chunks up to `retrieval.chunk_max_chars` (default 1200) with `chunk_overlap_chars` (default 150) overlap.
  The embedded text is `title > heading path` followed by the chunk, so a chunk carries its context.
- **Embeddings** (`providers/embeddings.py`), chosen per assistant in YAML:
  - `fastembed` (default): local ONNX model, e.g. `BAAI/bge-small-en-v1.5` (384-dim). No key, no quota. Queries get the
    BGE retrieval instruction prefix.
  - `gemini`: `gemini-embedding-001` with `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types (needs `GEMINI_API_KEY`).
  Chunks are tagged with `embedding_model`; search only compares vectors of the assistant's current model.

## Access control in SQL

Each knowledge source in YAML has `access_labels`. They are copied onto every chunk, and the search query
(`services/retrieval.py`) filters inside the `WHERE` clause, never afterwards:

```sql
WHERE c.tenant_id = :tenant_id AND c.assistant_id = :assistant_id
  AND c.embedding_model = :embedding_model
  AND c.access_labels && :principal_labels          -- any shared label grants access
  AND 1 - (c.embedding <=> :query) >= :min_similarity
ORDER BY c.embedding <=> :query LIMIT :top_k
```

A principal's labels come from the token: `user:<sub>`, `tenant:<tenant>`, `role:<realm role>`, `group:<group>`.

If nothing passes the filter, `/chat` answers "I don't know…" **without calling the model**, with `grounded: false`.
Otherwise the numbered chunks go to the model as CONTEXT, and the response carries `grounded: true` and
structured `citations` (`number`, `title`, `source_uri`, `heading_path`, `page`, `score`, `snippet`, ids).
`POST /search` returns the same citations without a model call.

## How source changes reach embeddings

Run ingestion after content or YAML changes (from `backend/`):

```bash
.venv/bin/python -m app.ingestion                # all assistants with retrieval enabled
.venv/bin/python -m app.ingestion --assistant leave_helper
```

It is idempotent and versioned. Every file is parsed first; its fingerprint is
`sha256(parsed text + title + access labels + chunk settings)`, stored with the embedding model id, so saving a file
without changing its text is not a change.

| Change | Effect |
|---|---|
| parsed text, labels and model unchanged | skipped (no embedding calls) |
| text edited, labels or chunk settings changed, or embedding model switched | new chunks are embedded **first**, then one transaction inserts them as version N+1 and flags version N `retired_at` |
| file deleted (or source removed from YAML) | the document and its chunks are flagged `retired_at`; a file that comes back becomes a new version |
| file fails to parse, or the swap transaction fails | reported in `failed`; the previous version stays active |

Nothing is ever deleted: search reads only rows with `retired_at IS NULL`, and retired versions stay for audit.
A forced failure inside the swap transaction is covered by `test_failed_swap_keeps_previous_version_active`.

Ingestion only reads the source folders; the read-only rule applies to sources. Nothing runs on a schedule
yet: re-run the command (or a cron/CI job) after content changes.

## min_similarity calibration

`min_similarity` is set per assistant and per embedding model from real scores:

```bash
.venv/bin/python -m app.ingestion.calibrate --queries ../eval/calibration/leave_helper.yaml
```

The tool embeds each relevant and unrelated query, takes the best chunk score across all of the assistant's chunks,
and suggests the midpoint between the lowest relevant score and the highest unrelated score.

| Assistant | Embedding model | Relevant | Unrelated | min_similarity |
|---|---|---|---|---|
| leave_helper | fastembed:BAAI/bge-small-en-v1.5 | 0.6203 – 0.8822 (11) | 0.3524 – 0.6318 (10) | **0.62** |
| finance_helper | fastembed:BAAI/bge-small-en-v1.5 | 0.7077 – 0.8447 (5) | 0.3168 – 0.4594 (5) | **0.584** |

Measured live on 2026-10-14 against the sample content (`content/`).

- **finance_helper** separates cleanly; the threshold is the midpoint.
- **leave_helper does not separate cleanly.** "How do I book a meeting room?" scores 0.6318, because it is close to
  the travel policy's "Booking" section, which is above the weakest relevant query, "Do unused casual leaves carry
  forward?" at 0.6203. "Company policy on remote work equipment" scores 0.6123. The cut-off 0.62 keeps all 11
  relevant queries and blocks 9 of 10 unrelated ones. The one that gets through reaches the model with irrelevant
  context, and the grounding rules (plus Phase 3 refusal detection) must answer "I don't know". A higher cut-off
  (0.632) would block it but refuse a real policy question.

Re-run the calibration whenever the content, chunking or embedding model changes. A Gemini embedding calibration has not been run
(no API key in this environment).
