#!/usr/bin/env bash
# Prints a dev access token from the local Keycloak (password grant, CLI testing only).
# Usage: scripts/get_token.sh [username]   (default: dev_user; password is <username>_password)
set -euo pipefail
user="${1:-dev_user}"
curl -fsS -X POST \
  "${KEYCLOAK_URL:-http://localhost:8080}/realms/ai-chatbot-dev/protocol/openid-connect/token" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode "username=${user}" \
  --data-urlencode "password=${user}_password" \
  -d 'grant_type=password' -d 'client_id=ai-chatbot-api' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
