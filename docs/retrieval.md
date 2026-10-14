# Ingestion and retrieval

## Pipeline

```
content/<path>  ->  parse_file()  ->  chunk_document()  ->  EmbeddingProvider.embed_documents()  ->  documents + chunks (Postgres/pgvector)
```

- **Parsers** (`backend/app/ingestion/parsers.py`): PDF (per page), DOCX (Heading 1–6 / Title styles, plus tables),
  XLSX (one section per sheet; rows keyed by the header row), Markdown (`#` headings, fenced code ignored),
  HTML (`h1`–`h6`, scripts and styles dropped), plain text. Library errors become `ParseError`; files over 25 MB are refused.
- **Chunking** (`chunking.py`): chunks never cross a heading section. Paragraphs, then sentences, then hard slices fill
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

It is idempotent. Each file's fingerprint is `sha256(bytes + access labels + chunk settings)`, stored alongside the
embedding model id:

| Change | Effect |
|---|---|
| file unchanged | skipped (no embedding calls) |
| file edited, labels or chunk settings changed, or embedding model switched | document re-parsed, its chunks replaced in one transaction |
| file deleted (or source removed from YAML) | its document and chunks are deleted from the index |
| file fails to parse | reported in `failed`; any previously indexed version is kept |

Ingestion only reads the source folders; the read-only rule applies to sources. Nothing runs on a schedule
yet: re-run the command (or a cron/CI job) after content changes.

## min_similarity calibration

`min_similarity` is set per assistant and per embedding model from real scores:

```bash
.venv/bin/python -m app.ingestion.calibrate --queries ../eval/calibration/leave_helper.yaml
```

The tool embeds each relevant and unrelated query, takes the best chunk score across all of the assistant's chunks,
and suggests the midpoint between the lowest relevant score and the highest unrelated score.

| Assistant | Embedding model | Relevant (11 queries) | Unrelated (10 queries) | min_similarity |
|---|---|---|---|---|
| leave_helper | fastembed:BAAI/bge-small-en-v1.5 | 0.6145 – 0.8820 | 0.3928 – 0.5960 | **0.605** |

Measured live on 2026-10-14 against the sample content. The gap is narrow (0.0185). The weakest relevant query was
"When is Founders' Day?" (0.6145, a short table row), and the strongest unrelated one was "What is the company's
policy on remote work equipment?" (0.5960, a policy question in a different domain). Re-run the calibration
whenever the content, chunking or embedding model changes. A Gemini embedding calibration has not been run
(no API key in this environment).
