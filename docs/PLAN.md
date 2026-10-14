# AI-Chatbot - Plan

Status labels used in every report: **live-verified** (real call against a real
service), **mock-tested** (automated test with fakes), **not verified**.

## 1. Understanding

A config-driven, retrieval-grounded, strictly read-only company chat assistant
platform. One YAML file (+ content + credentials in env vars) defines an assistant.
Everything else is shared: Keycloak sign-in, permission-aware pgvector search
(access control inside the SQL), guardrails, streaming (SSE), structured citations,
charts, conversation history and one audit row per turn. Answers come only from
retrieved documents or read-only tools/connectors; otherwise "I don't know".

Per-turn pipeline: sign-in check -> input guardrails / greeting short-circuit ->
permission-filtered vector search with relevance cut-off -> optional read-only tools
-> model call -> output guardrails -> streamed reply + citations -> persist
messages + one audit row.

The reference repo (mayankraj179/Generic-AI-Assistant) is readable from this
environment. It is used for structure and behaviour only; no secrets, keys,
company names or real documents are copied. Code is rewritten for this repo,
trimmed to what each phase needs.

## 2. Phases (one PR per phase, stop for "go" after Phase 1)

| Phase | Scope | Acceptance |
|---|---|---|
| 0 | Layout `backend/ ui/ infra/ docs/ scripts/`; docker-compose (Postgres 16 + pgvector, Keycloak); dev realm export; CI (ruff, pytest, gitleaks); README | Clean clone -> running stack with documented commands |
| 1 | Config loader; JWT validation; principal; schema + Alembic; conversations with owner checks; Gemini provider; `/health`, `/assistants`, `/sessions`, `/sessions/{id}/messages`, `/chat`, `/chat/stream` | Real Keycloak tokens via curl; no token 401; other user's conversation 404 |
| 2-9 | Per spec (ingestion/retrieval, guardrails, UI, providers/charts, sources, observability, eval/security, REST connectors) | Per spec |

## 3. Design decisions (Phase 0/1)

- **Python 3.12** via `uv`-managed interpreter; `backend/pyproject.toml`, no lockfile
  manager beyond pip/uv. Dependencies Phase 0/1: fastapi, uvicorn, pydantic,
  pydantic-settings, pyyaml, sqlalchemy[asyncio], asyncpg, alembic, pgvector, pyjwt[crypto]>=2.15,
  httpx (JWKS fetch + tests), google-genai. Dev: pytest, pytest-asyncio, ruff.
- **Keycloak realm** `ai-chatbot-dev`: public client `ai-chatbot-api` (standard flow,
  PKCE S256 required, redirect `http://localhost:*` and `http://127.0.0.1:*`,
  audience mapper so `aud` contains the client id), access token 3600 s. Direct
  access grant enabled on the dev client **only for curl/CLI testing** (no UI
  uses it). Users: `dev_user` (normal) and `dev_admin` (has realm role
  `chatbot-admin`); a second normal user `dev_user2` so the cross-user 404 check
  uses real tokens. All passwords are dev-only, documented in README.
- **JWT**: JWKS fetched once and cached (PyJWT `PyJWKClient` with cache + lifespan);
  validation is local. Checks: signature, `iss`, `aud`, `exp` (required), `sub`
  required, algorithm allow-list from env (default `RS256`; `none`/HS* always
  rejected). `RecursionError`/`TypeError`/`ValueError` during decode -> 401.
- **Principal**: `tenant_id`, `principal_id` (`sub`), namespaced labels
  (`user:<sub>`, `role:<realm role>`, `group:<group>`, `tenant:<id>`), permissions.
  A principal with zero labels is refused (403). Tenant comes from a `tenant_id`
  claim if present, else `local-development` (dev realm has no tenant mapper).
- **Permissions**: every authenticated user gets `chat`/`conversation:*`;
  `admin` permission only from realm role `chatbot-admin`. Admin routes arrive
  in Phase 2 but the check is built in Phase 1.
- **Schema (Phase 1)**: `conversations` (tenant, assistant, owner, title),
  `messages` (role, content, citations JSONB, seq). `documents`, `chunks`
  (dimensionless `vector`, `embedding_model` tag) and `turn_audit` added in the
  phase that uses them, via new migrations.
- **Owner checks**: every conversation read/write filters by
  `tenant_id AND principal_id AND assistant_id` in the SQL; not found and not
  owned are both 404.
- **Phase 1 chat**: retrieval does not exist yet, and the product rules forbid
  answering from model knowledge. So in Phase 1 `/chat` runs the real Gemini call
  with the assistant's system prompt plus an explicit "no documents were found ->
  say you don't know" context, persists both messages and returns
  `grounded: false`, `citations: []`. This proves auth, provider, persistence and
  streaming end-to-end without violating the grounding rule. Real grounding lands
  in Phase 2/3.
- **Streaming**: SSE events `delta`, `done`, `error` (`chart` added in Phase 5).
  Nothing persisted on a mid-stream failure.
- **Tests** use their own database (`chatbot_test`), created by the test fixture,
  never the dev DB. Provider calls are faked in default pytest; live checks are
  separate, opt-in and counted against quota.

## 4. Assumptions (tell me if any is wrong)

1. Docker + Docker Compose are available in my environment (checked: Docker
   Compose v5.4 works). Live verification runs against the compose stack here.
2. Gemini chat model: a current Flash model confirmed from Google's model list
   at implementation time (I will read the docs and list models with the key,
   not guess the name).
3. Phase 1 PR is stacked on the Phase 0 branch (base = Phase 0 branch) so both
   can be reviewed before you say "go".
4. Live Gemini budget for Phase 1: at most ~5 chat requests total (one /chat,
   one /chat/stream, plus a model-list call which is not a chat request).

## 5. Blocking questions

1. **Gemini API key.** No secrets exist in this session. Phase 1 live
   verification of `/chat` needs `GEMINI_API_KEY`. Options: skip (Phase 1 provider
   becomes mock-tested only), this session only, or saved for future sessions.
   **Answer (2026-10-14): skip.** The Gemini provider is mock-tested only in Phase 1.
2. **Empty repo / base branch.** A PR needs a base branch. May I push one bootstrap
   commit straight to `main` containing only `.gitignore`, `.env.example` and this
   `docs/PLAN.md`, then open Phase 0 and Phase 1 as PRs against it?
   **Answer (2026-10-14): yes.** The bootstrap commit is on `main`; Phase 1 is stacked on Phase 0.

## 6. Open risks

- Free Gemini quota (~20 chat requests/day) limits live checks; batched, no retries.
- Keycloak imports the realm only into an empty volume; later realm changes go
  through the admin API (documented in README).
