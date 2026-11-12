"""CDSL reader: source text -> AST, and AST -> source text.

AST node types
--------------
- ``Symbol(name)``   identifiers and operators: ``x``, ``total-cost``, ``>=``, ``+``
- ``Keyword(name)``  ``:type``, ``:op``, ``:env`` (name stored without the colon)
- ``str`` / ``int`` / ``float`` / ``bool`` / ``None`` literals
- ``list``           a ``( ... )`` form
- ``Vector``         a ``[ ... ]`` literal
- ``MapLit``         a ``{ k v ... }`` literal, stored as a list of (key, value) pairs

Commas are whitespace (Clojure convention) and ``;`` starts a line comment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class CDSLSyntaxError(ValueError):
    """Raised when text is not a well-formed CDSL program."""


@dataclass(frozen=True)
class Symbol:
    name: str

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


@dataclass(frozen=True)
class Keyword:
    name: str

    def __repr__(self) -> str:
        return f"Keyword({self.name!r})"


class Vector(list):
    """``[a b c]`` literal."""


class MapLit(list):
    """``{k1 v1 k2 v2}`` literal, kept as an ordered list of (key, value) pairs.

    A literal with an odd number of elements still parses: the unpaired last
    element is kept in ``dangling``. Small models often write ``:env {a b c}``
    as a plain list of names, and the runtime accepts that in ``:env`` position.
    Evaluating such a literal anywhere else is a runtime error.
    """
    dangling: object = None


_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {")", "]", "}"}
_ATOM_END = set("()[]{}\";,") | {" ", "\t", "\r", "\n"}
_LITERALS = {"true": True, "false": False, "nil": None, "null": None, "None": None}


def _tokenize(text: str) -> list[tuple[str, object, int]]:
    """Split text into (kind, value, offset) tokens. kind: open|close|str|atom."""
    tokens: list[tuple[str, object, int]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n,":
            i += 1
        elif c == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif c in _OPEN:
            tokens.append(("open", c, i))
            i += 1
        elif c in _CLOSE:
            tokens.append(("close", c, i))
            i += 1
        elif c == '"':
            start, i, buf = i, i + 1, []
            while True:
                if i >= n:
                    raise CDSLSyntaxError(f"unterminated string starting at offset {start}")
                ch = text[i]
                if ch == "\\" and i + 1 < n:
                    nxt = text[i + 1]
                    buf.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt, "\\" + nxt))
                    i += 2
                elif ch == '"':
                    i += 1
                    break
                else:
                    buf.append(ch)
                    i += 1
            tokens.append(("str", "".join(buf), start))
        else:
            start = i
            while i < n and text[i] not in _ATOM_END:
                i += 1
            tokens.append(("atom", text[start:i], start))
    return tokens


def _atom(text: str) -> object:
    if text in _LITERALS:
        return _LITERALS[text]
    if _NUMBER.match(text):
        if re.match(r"^[+-]?\d+$", text):
            return int(text)
        return float(text)
    if text.startswith(":") and len(text) > 1:
        return Keyword(text[1:])
    return Symbol(text)


def _read(tokens: list, pos: int) -> tuple[object, int]:
    kind, value, offset = tokens[pos]
    if kind == "str":
        return value, pos + 1
    if kind == "atom":
        return _atom(value), pos + 1
    if kind == "close":
        raise CDSLSyntaxError(f"unexpected '{value}' at offset {offset} (extra closing bracket?)")
    # kind == "open"
    closer = _OPEN[value]
    items: list = []
    pos += 1
    while True:
        if pos >= len(tokens):
            raise CDSLSyntaxError(
                f"unbalanced brackets: '{value}' opened at offset {offset} is never closed"
            )
        k, v, o = tokens[pos]
        if k == "close":
            if v != closer:
                raise CDSLSyntaxError(
                    f"mismatched brackets: '{value}' at offset {offset} closed by '{v}' at offset {o}"
                )
            pos += 1
            break
        item, pos = _read(tokens, pos)
        items.append(item)
    if value == "[":
        return Vector(items), pos
    if value == "{":
        m = MapLit(zip(items[0::2], items[1::2]))
        if len(items) % 2:
            m.dangling = items[-1]
        return m, pos
    return items, pos


def parse_all(text: str) -> list:
    """Parse every top-level form in ``text``."""
    tokens = _tokenize(text)
    forms, pos = [], 0
    while pos < len(tokens):
        form, pos = _read(tokens, pos)
        forms.append(form)
    return forms


def parse(text: str) -> object:
    """Parse exactly one form."""
    forms = parse_all(text)
    if not forms:
        raise CDSLSyntaxError("empty program")
    if len(forms) > 1:
        raise CDSLSyntaxError(f"expected one form, found {len(forms)} (extra closing parenthesis?)")
    return forms[0]


def parse_program(text: str) -> list:
    """Parse text that must be a single ``(program ...)`` form."""
    form = parse(text)
    if not (isinstance(form, list) and form and form[0] == Symbol("program")):
        raise CDSLSyntaxError("top-level form must be (program ...)")
    return form


def extract_program_text(raw: str) -> str:
    """Pull the first balanced ``(program ...)`` form out of raw LLM output.

    Tolerates prose and markdown fences around the program. Brackets inside
    strings and comments are ignored while matching.
    """
    text = re.sub(r"```[a-zA-Z]*", "", raw)
    start = text.find("(program")
    if start < 0:
        start = text.find("(")
    if start < 0:
        raise CDSLSyntaxError("no s-expression found in compiler output")
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise CDSLSyntaxError("unbalanced parentheses: the program is never closed")


def to_source(node: object, max_len: int | None = None) -> str:
    """Render an AST node back to compact CDSL source."""
    if isinstance(node, Symbol):
        s = node.name
    elif isinstance(node, Keyword):
        s = ":" + node.name
    elif isinstance(node, bool):
        s = "true" if node else "false"
    elif node is None:
        s = "nil"
    elif isinstance(node, str):
        s = '"' + node.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    elif isinstance(node, MapLit):
        parts = [f"{to_source(k)} {to_source(v)}" for k, v in node]
        if node.dangling is not None:
            parts.append(to_source(node.dangling))
        s = "{" + " ".join(parts) + "}"
    elif isinstance(node, Vector):
        s = "[" + " ".join(to_source(x) for x in node) + "]"
    elif isinstance(node, list):
        s = "(" + " ".join(to_source(x) for x in node) + ")"
    else:
        s = repr(node)
    if max_len is not None and len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s
