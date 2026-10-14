# AI-Chatbot

A config-driven, retrieval-grounded, **read-only** company chat assistant platform.
A new assistant is one YAML file plus its content and credentials; sign-in,
permission-aware search, safety checks, streaming, citations, charts and logging are
shared by every assistant. Plan and status: [`docs/PLAN.md`](docs/PLAN.md).

## Layout

| Path | What |
|---|---|
| `backend/` | Python 3.12 FastAPI service (`app/`), tests (`tests/`) |
| `ui/` | React 18 + Vite + TypeScript frontends (Phase 4) |
| `infra/` | `docker-compose.yml` (Postgres 16 + pgvector, Keycloak), dev realm, DB init |
| `docs/` | Plan and design notes |
| `scripts/` | Developer helpers (e.g. `get_token.sh`) |

## Prerequisites

- Docker with Docker Compose v2
- Python 3.12 (e.g. `uv python install 3.12`)
- `curl`; optionally [`gitleaks`](https://github.com/gitleaks/gitleaks) for local secret scans

## Quick start (clean clone)

```bash
git clone https://github.com/mayankraj179/AI-Chatbot.git
cd AI-Chatbot
cp .env.example backend/.env           # local dev values; never commit .env

# 1. Infrastructure: Postgres+pgvector on :5432, Keycloak on :8080
docker compose -f infra/docker-compose.yml up -d
docker compose -f infra/docker-compose.yml ps   # wait until both are "healthy" (~1 min)

# 2. Backend on :8000
cd backend
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000

# 3. Check it
curl http://localhost:8000/health           # {"status":"ok"}
```

## Dev identity provider (Keycloak)

- Admin console: http://localhost:8080 (`admin` / `admin`, dev only)
- Realm `ai-chatbot-dev`, public client `ai-chatbot-api` (Authorization Code + PKCE
  S256, redirect `http://localhost:*` / `http://127.0.0.1:*`, 3600 s access tokens,
  audience mapper so `aud` = `ai-chatbot-api`).
- Dev users (password = `<username>_password`, dev only):

| User | Realm roles | Purpose |
|---|---|---|
| `dev_user` | - | normal user |
| `dev_user2` | - | second user, for cross-user isolation checks |
| `dev_admin` | `chatbot-admin` | may call `/admin/*` |

The UI signs in with Authorization Code + PKCE. For curl/CLI testing only, the dev
client also allows the password grant:

```bash
TOKEN=$(scripts/get_token.sh dev_user)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/assistants
```

### Changing the dev realm

Keycloak imports `infra/keycloak/realm-export.json` **only into an empty volume**.
Editing the file does nothing to a running realm. Either change the live realm via
the admin console / admin REST API (keeps data), or, if losing all Keycloak state is
acceptable, recreate the volume: `docker compose -f infra/docker-compose.yml down`
then `docker volume rm ai-chatbot_keycloak_data` and `up -d` again.

## Databases

`infra/postgres/init.sql` runs on the first start of an empty volume: it enables
`vector` and creates a separate `chatbot_test` database. Tests never use the dev
database `chatbot`.

## Development checks

```bash
cd backend
ruff check . && ruff format --check .
pytest -q
cd .. && gitleaks detect --source . --redact   # run before every PR
```

CI (`.github/workflows/ci.yml`) runs ruff, pytest and gitleaks on every PR.

## Secrets

Secrets live only in environment variables (`backend/.env`, ignored by git) or the
deployment's secret store. `.env.example` lists every variable with empty
placeholders. Assistant YAML files may only reference secrets by env var name.
