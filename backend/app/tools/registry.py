"""Read-only tools the model may call. Each returns a JSON-serialisable result."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.providers.base import ToolSpec
from app.tools.calculator import CalculationError, calculate, format_number


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    run: Callable[[dict[str, Any]], dict[str, Any]]


def _current_datetime(args: dict[str, Any], now: Callable[[], datetime] | None = None) -> dict:
    name = str(args.get("timezone") or "UTC")
    try:
        zone = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return {"error": f"unknown timezone '{name[:64]}'"}
    current = (now or (lambda: datetime.now(UTC)))().astimezone(zone)
    return {
        "timezone": name,
        "iso": current.isoformat(timespec="seconds"),
        "date": current.date().isoformat(),
        "weekday": current.strftime("%A"),
        "time": current.strftime("%H:%M"),
    }


def _calculate(args: dict[str, Any]) -> dict:
    expression = str(args.get("expression", ""))
    try:
        return {"expression": expression, "result": format_number(calculate(expression))}
    except CalculationError as exc:
        return {"expression": expression[:200], "error": str(exc)}


TOOLS: dict[str, Tool] = {
    "current_datetime": Tool(
        ToolSpec(
            name="current_datetime",
            description="Returns the current date, time and weekday in an IANA timezone.",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone such as 'Asia/Kolkata'. Defaults to UTC.",
                    }
                },
            },
        ),
        _current_datetime,
    ),
    "calculate": Tool(
        ToolSpec(
            name="calculate",
            description=(
                "Evaluates an arithmetic expression with + - * / % ^ and parentheses, "
                "e.g. '(96.5 - 42.3) / 42.3 * 100'."
            ),
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        ),
        _calculate,
    ),
}


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool '{name[:64]}'"}
    return tool.run(args if isinstance(args, dict) else {})


_CALC_INTENT = re.compile(
    r"\d\s*[-+*/x×÷^%]\s*\d|\b(calculate|compute|sum|total|average|percent(age)?|plus|minus|"
    r"times|multipl\w*|divid\w*|difference|growth|increase|decrease|ratio|how much more)\b"
)
_DATETIME_INTENT = re.compile(
    r"\b(today|tomorrow|yesterday|now|date|time|day is it|weekday|current(ly)?|this (week|month|"
    r"year)|next (week|month)|days? (until|left|till))\b"
)


def may_need_tool(message: str, enabled_tools: list[str]) -> bool:
    """Cheap pre-check before a model call when retrieval found nothing.

    A tool-enabled assistant with zero chunks may only answer through a tool, so a
    message with no arithmetic or date/time intent is refused without a provider call.
    """
    text = message.lower()
    if "calculate" in enabled_tools and _CALC_INTENT.search(text):
        return True
    return "current_datetime" in enabled_tools and bool(_DATETIME_INTENT.search(text))
