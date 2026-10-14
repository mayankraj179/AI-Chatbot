# Guardrails, grounding and tools

Order of a turn (`backend/app/chat/service.py`):

1. Sign-in check and conversation owner check (404 for someone else's session).
2. Length check (`guardrails.max_input_chars`).
3. **Greeting handler** (`chat/greeting.py`): if the *whole* message is a plain greeting ("hi", "good morning team",
   "namaste"…), return a fixed reply built from `display_name`. No search, no provider call, `outcome: "greeting"`.
   "hi, how many casual leaves do I get?" is not a greeting and goes through retrieval.
4. **Input guardrails** (`guardrails/input.py`), run on NFKC-normalised text with zero-width characters removed:
   - `prompt_injection_screening` (default on): patterns such as "ignore previous instructions", "reveal your
     system prompt", "you are now…", "developer mode", fake `<system>` tags, "answer from general knowledge".
   - `input_pii_block` (default off): email, phone, national-id shapes, Luhn-valid card numbers.
   A hit returns **400** `{"detail": {"reason": "prompt_injection" | "input_pii", "message": "..."}}`.
   Nothing is retrieved, sent to a model or saved.
5. Permission-filtered retrieval (see `docs/retrieval.md`).
6. **Zero chunks:** without tools, or with no arithmetic/date intent in the message, the answer is the fixed
   "I don't know…" text with no provider call (`outcome: "no_context"`). With tool intent, the model is called with
   tools only; if it answers without calling a tool, the answer is replaced by "I don't know…" (a tool-enabled
   assistant with zero chunks may only answer through a tool).
7. Model call, with tool rounds (up to 3) when `enabled_tools` is set.
8. **Output guardrails** on the full text (`guardrails/output.py`):
   - stray citation markers (`【4:0†source】`, `[1]`, `[1, 2]`, `[source …]`, `(source 3)`) are stripped, including
     across streamed chunks (`MarkerStripper` holds back an unfinished marker);
   - `output_pii_block` (default off) replaces the answer with a refusal (`outcome: "output_pii"`);
   - **refusal detection**: "I don't know", "not mentioned in the context", … → `grounded: false`, no citations,
     `outcome: "refused"`.

`grounded` is true only when retrieved chunks were used. Tool answers without documents have `grounded: false`,
`citations: []`, `outcome: "tool_answer"`, so the UI can label them "No document sources".

Every response (and the SSE `done` event) carries `outcome` and `tools_used`. In streaming, `done.answer` is the
authoritative final text after output checks; nothing is saved on a mid-stream failure.

## Tools (read-only)

| Tool | Arguments | Notes |
|---|---|---|
| `current_datetime` | `timezone` (IANA, default UTC) | date, time, weekday |
| `calculate` | `expression` | hand-written recursive-descent parser (`tools/calculator.py`), never `eval`; `+ - * / % ^ ( )`, max 200 chars, nesting 32, exponent 100, magnitude 1e15 |

Tools are declared to the provider as function declarations; the application executes them itself (the SDK's
automatic function calling is disabled). A call to a tool that is not in the assistant's `enabled_tools` is not
executed and returns an error result to the model.
