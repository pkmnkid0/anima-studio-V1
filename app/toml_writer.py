"""
Minimal, dependency-free TOML serializer.

We only ever need to write the shapes this app produces (nested dicts of
str / int / float / bool / list / dict, plus lists of dicts for arrays of
tables), so a hand-rolled writer avoids depending on a third-party toml
package. Correctness is checked in tests/test_toml_writer.py by round
tripping through Python's stdlib `tomllib` parser.
"""
from __future__ import annotations

import math


def _format_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return _format_string(value)
    raise TypeError(f"Unsupported scalar type for TOML: {type(value)!r}")


def _format_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _format_inline_array(items: list) -> str:
    parts = [_format_value_inline(v) for v in items]
    return "[" + ", ".join(parts) + "]"


def _format_value_inline(value) -> str:
    if isinstance(value, dict):
        parts = [f"{_format_key(k)} = {_format_value_inline(v)}" for k, v in value.items()]
        return "{ " + ", ".join(parts) + " }"
    if isinstance(value, (list, tuple)):
        return _format_inline_array(list(value))
    return _format_scalar(value)


def _format_key(key: str) -> str:
    # Bare keys: letters, digits, underscore, dash. Quote anything else.
    if key and all(c.isalnum() or c in "_-" for c in key):
        return key
    return _format_string(key)


def _write_table(lines: list[str], table: dict, prefix: tuple[str, ...]) -> None:
    """Write a table's scalar/array-of-scalar keys first, then recurse
    into subtables (and arrays-of-tables), matching conventional TOML
    layout so the file stays human-readable."""
    scalar_items = []
    array_of_tables_items = []
    subtable_items = []

    for key, value in table.items():
        if isinstance(value, dict):
            subtable_items.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            array_of_tables_items.append((key, value))
        else:
            scalar_items.append((key, value))

    # A table header is only worth emitting when the table itself has
    # scalar keys directly on it. A table that exists purely to hold
    # subtables (e.g. "general_args" holding only "general_args.args")
    # doesn't need its own [general_args] line - TOML implies it from
    # the child header - and skipping it keeps the file readable.
    if prefix and scalar_items:
        header = ".".join(_format_key(p) for p in prefix)
        lines.append(f"[{header}]")

    for key, value in scalar_items:
        if isinstance(value, list):
            lines.append(f"{_format_key(key)} = {_format_inline_array(value)}")
        else:
            lines.append(f"{_format_key(key)} = {_format_scalar(value)}")

    if scalar_items and (subtable_items or array_of_tables_items):
        lines.append("")

    for i, (key, value) in enumerate(subtable_items):
        _write_table(lines, value, prefix + (key,))
        if i < len(subtable_items) - 1 or array_of_tables_items:
            lines.append("")

    for i, (key, value) in enumerate(array_of_tables_items):
        header = ".".join(_format_key(p) for p in prefix + (key,))
        for entry in value:
            lines.append(f"[[{header}]]")
            sub_lines: list[str] = []
            _write_table_body(sub_lines, entry)
            lines.extend(sub_lines)
            lines.append("")


def _write_table_body(lines: list[str], table: dict) -> None:
    """Write only the scalar keys of a table with no header - used for
    the body of an [[array.of.tables]] entry."""
    scalar_items = []
    nested_items = []
    for key, value in table.items():
        if isinstance(value, dict):
            nested_items.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            nested_items.append((key, value))
        else:
            scalar_items.append((key, value))
    for key, value in scalar_items:
        if isinstance(value, list):
            lines.append(f"{_format_key(key)} = {_format_inline_array(value)}")
        else:
            lines.append(f"{_format_key(key)} = {_format_scalar(value)}")
    # Nested tables inside an inline array-of-tables entry are rare in
    # our data; fall back to inline representation to stay valid TOML.
    for key, value in nested_items:
        lines.append(f"{_format_key(key)} = {_format_value_inline(value)}")


def dumps(data: dict) -> str:
    """Serialize a nested dict to a TOML document string."""
    if not isinstance(data, dict):
        raise TypeError("Top-level TOML value must be a dict")
    lines: list[str] = []
    _write_table(lines, data, ())
    text = "\n".join(lines).rstrip() + "\n"
    return text
