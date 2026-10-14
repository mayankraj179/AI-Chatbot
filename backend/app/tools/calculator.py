"""Arithmetic for the ``calculate`` tool: a hand-written recursive-descent parser.

Grammar (``^`` is right-associative, unary minus binds tighter than ``^`` operands):
    expr   := term (('+' | '-') term)*
    term   := power (('*' | '/' | '%') power)*
    power  := unary ('^' power)?
    unary  := ('-' | '+') unary | atom
    atom   := NUMBER | '(' expr ')'
Never uses eval.
"""

from __future__ import annotations

import math
import re

MAX_EXPRESSION_CHARS = 200
MAX_DEPTH = 32
MAX_EXPONENT = 100
MAX_MAGNITUDE = 1e15

_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d*)?|\.\d+)|(\*\*|[-+*/%^()]))")


class CalculationError(ValueError):
    pass


def _tokenize(expression: str) -> list[str]:
    expression = expression.replace(",", "").replace("×", "*").replace("÷", "/")
    tokens: list[str] = []
    position = 0
    while position < len(expression):
        if expression[position:].strip() == "":
            break
        match = _TOKEN.match(expression, position)
        if not match:
            raise CalculationError(f"unexpected character {expression[position]!r}")
        tokens.append("^" if match.group(2) == "**" else (match.group(1) or match.group(2)))
        position = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0
        self.depth = 0

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise CalculationError("unexpected end of expression")
        self.position += 1
        return token

    def expr(self) -> float:
        value = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            right = self.term()
            value = value + right if op == "+" else value - right
        return value

    def term(self) -> float:
        value = self.power()
        while self.peek() in ("*", "/", "%"):
            op = self.take()
            right = self.power()
            if op == "*":
                value *= right
            elif right == 0:
                raise CalculationError("division by zero")
            elif op == "/":
                value /= right
            else:
                value %= right
        return value

    def power(self) -> float:
        base = self.unary()
        if self.peek() == "^":
            self.take()
            exponent = self.power()
            if abs(exponent) > MAX_EXPONENT:
                raise CalculationError("exponent too large")
            if base == 0 and exponent < 0:
                raise CalculationError("division by zero")
            if base < 0 and not float(exponent).is_integer():
                raise CalculationError("fractional power of a negative number")
            return base**exponent
        return base

    def unary(self) -> float:
        if self.peek() in ("-", "+"):
            op = self.take()
            self._enter()
            value = self.unary()
            self.depth -= 1
            return -value if op == "-" else value
        return self.atom()

    def atom(self) -> float:
        item = self.take()
        if item == "(":
            self._enter()
            value = self.expr()
            self.depth -= 1
            if self.peek() != ")":
                raise CalculationError("missing ')'")
            self.take()
            return value
        if item[0].isdigit() or item[0] == ".":
            return float(item)
        raise CalculationError(f"unexpected {item!r}")

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalculationError("expression is nested too deeply")


def calculate(expression: str) -> float:
    if not expression or not expression.strip():
        raise CalculationError("empty expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculationError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    parser = _Parser(_tokenize(expression))
    value = parser.expr()
    if parser.peek() is not None:
        raise CalculationError(f"unexpected {parser.peek()!r}")
    if math.isnan(value) or math.isinf(value) or abs(value) > MAX_MAGNITUDE:
        raise CalculationError("result out of range")
    return value


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"
