from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.chat.service import NO_CONTEXT
from app.providers.base import ProviderError
from tests.conftest import FakeProvider, TokenFactory, auth, random_uuid

ASSISTANT = "leave_helper"


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_protected_routes_need_a_token(client: TestClient) -> None:
    body = {"assistant_id": ASSISTANT, "message": "hi"}
    assert client.get("/assistants").status_code == 401
    assert client.post("/sessions", json={"assistant_id": ASSISTANT}).status_code == 401
    assert client.get(f"/sessions/{random_uuid()}/messages").status_code == 401
    assert client.post("/chat", json=body).status_code == 401
    assert client.post("/chat/stream", json=body).status_code == 401


def test_list_assistants_filtered_by_tenant(client: TestClient, tokens: TokenFactory) -> None:
    mine = client.get("/assistants", headers=auth(tokens.make("a")))
    assert [a["assistant_id"] for a in mine.json()] == [ASSISTANT]
    other_tenant = client.get("/assistants", headers=auth(tokens.make("a", tenant_id="acme")))
    assert other_tenant.json() == []


def test_chat_persists_turn_and_returns_ungrounded_answer(
    client: TestClient, tokens: TokenFactory, provider: FakeProvider
) -> None:
    headers = auth(tokens.make("alice"))
    response = client.post(
        "/chat",
        json={"assistant_id": ASSISTANT, "message": "How many casual leaves?"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["grounded"] is False and data["citations"] == []
    assert NO_CONTEXT in provider.requests[0].user_message
    assert "Never use general" in provider.requests[0].system_prompt

    messages = client.get(f"/sessions/{data['session_id']}/messages", headers=headers).json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == data["answer"]

    follow_up = client.post(
        "/chat",
        json={
            "assistant_id": ASSISTANT,
            "message": "And earned?",
            "session_id": data["session_id"],
        },
        headers=headers,
    )
    assert follow_up.json()["session_id"] == data["session_id"]
    assert [t.role for t in provider.requests[1].history] == ["user", "assistant"]
    assert len(client.get(f"/sessions/{data['session_id']}/messages", headers=headers).json()) == 4


def test_other_users_conversation_is_404(client: TestClient, tokens: TokenFactory) -> None:
    alice, bob = auth(tokens.make("alice")), auth(tokens.make("bob"))
    created = client.post("/sessions", json={"assistant_id": ASSISTANT}, headers=alice)
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    assert client.get(f"/sessions/{session_id}/messages", headers=alice).json() == []
    assert client.get(f"/sessions/{session_id}/messages", headers=bob).status_code == 404
    write = {"assistant_id": ASSISTANT, "message": "hi there", "session_id": session_id}
    assert client.post("/chat", json=write, headers=bob).status_code == 404
    assert client.post("/chat/stream", json=write, headers=bob).status_code == 404
    # Same user id in another tenant is a different principal.
    other_tenant = auth(tokens.make("alice", tenant_id="acme"))
    assert client.get(f"/sessions/{session_id}/messages", headers=other_tenant).status_code == 404
    # Nothing was written by the rejected attempts.
    assert client.get(f"/sessions/{session_id}/messages", headers=alice).json() == []


def test_unknown_and_malformed_ids_are_404(client: TestClient, tokens: TokenFactory) -> None:
    headers = auth(tokens.make("alice"))
    assert client.get(f"/sessions/{random_uuid()}/messages", headers=headers).status_code == 404
    assert client.get("/sessions/not-a-uuid/messages", headers=headers).status_code == 404
    assert (
        client.post("/sessions", json={"assistant_id": "nope"}, headers=headers).status_code == 404
    )
    chat = client.post("/chat", json={"assistant_id": "nope", "message": "x"}, headers=headers)
    assert chat.status_code == 404


def test_input_length_and_empty_rejected(client: TestClient, tokens: TokenFactory) -> None:
    headers = auth(tokens.make("alice"))
    too_long = {"assistant_id": ASSISTANT, "message": "x" * 4001}
    assert client.post("/chat", json=too_long, headers=headers).status_code == 400
    empty = {"assistant_id": ASSISTANT, "message": "   "}
    assert client.post("/chat", json=empty, headers=headers).status_code == 400


def test_provider_error_is_502_and_persists_nothing(
    client: TestClient, tokens: TokenFactory, provider: FakeProvider
) -> None:
    headers = auth(tokens.make("alice"))
    provider.fail_with = ProviderError("QUOTA_EXCEEDED", "quota", provider="fake")
    response = client.post(
        "/chat", json={"assistant_id": ASSISTANT, "message": "q"}, headers=headers
    )
    assert response.status_code == 502
    assert response.json()["detail"]["kind"] == "QUOTA_EXCEEDED"


def test_stream_emits_deltas_then_done_and_persists(
    client: TestClient, tokens: TokenFactory
) -> None:
    headers = auth(tokens.make("alice"))
    response = client.post(
        "/chat/stream", json={"assistant_id": ASSISTANT, "message": "q"}, headers=headers
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [e for e, _ in events] == ["delta", "delta", "done"]
    done = events[-1][1]
    assert done["grounded"] is False and done["citations"] == []
    messages = client.get(f"/sessions/{done['session_id']}/messages", headers=headers).json()
    assert messages[1]["content"] == "".join(d["text"] for e, d in events if e == "delta")


def test_mid_stream_failure_emits_error_and_persists_nothing(
    client: TestClient, tokens: TokenFactory, provider: FakeProvider
) -> None:
    headers = auth(tokens.make("alice"))
    session_id = client.post("/sessions", json={"assistant_id": ASSISTANT}, headers=headers).json()[
        "session_id"
    ]
    provider.fail_mid_stream = True
    response = client.post(
        "/chat/stream",
        json={"assistant_id": ASSISTANT, "message": "q", "session_id": session_id},
        headers=headers,
    )
    events = parse_sse(response.text)
    assert [e for e, _ in events] == ["delta", "error"]
    assert events[-1][1]["kind"] == "OVERLOADED"
    assert client.get(f"/sessions/{session_id}/messages", headers=headers).json() == []
