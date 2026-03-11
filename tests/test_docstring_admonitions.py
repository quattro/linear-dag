from __future__ import annotations

import ast
import re

from pathlib import Path

ADMONITION_PATTERN = re.compile(r"^(?P<indent>[ \t]*)!!!\s+(?P<kind>\w+)(?:\s+.*)?$")


def _find_sequential_same_type_admonitions(docstring: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    lines = docstring.splitlines()
    i = 0
    while i < len(lines):
        current_match = ADMONITION_PATTERN.match(lines[i])
        if current_match is None:
            i += 1
            continue

        indent = current_match.group("indent")
        admonition_type = current_match.group("kind")
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if not line.strip():
                j += 1
                continue

            next_match = ADMONITION_PATTERN.match(line)
            if next_match is not None and next_match.group("indent") == indent:
                break

            if line.startswith(indent + "    "):
                j += 1
                continue

            break

        if j < len(lines):
            next_match = ADMONITION_PATTERN.match(lines[j])
            if (
                next_match is not None
                and next_match.group("indent") == indent
                and next_match.group("kind") == admonition_type
            ):
                matches.append((i + 1, admonition_type))

        i = max(j, i + 1)

    return matches


def test_public_docstrings_do_not_repeat_same_admonition_type_sequentially() -> None:
    violations: list[str] = []

    for path in sorted(Path("src").rglob("*.py")):
        module = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(module):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            docstring = ast.get_docstring(node, clean=False)
            if docstring is None:
                continue

            node_name = getattr(node, "name", "<module>")
            for line_number, admonition_type in _find_sequential_same_type_admonitions(docstring):
                violations.append(
                    (
                        f"{path}:{getattr(node, 'lineno', 1)}:{node_name}: "
                        f"docstring line {line_number} repeats !!! {admonition_type}"
                    )
                )

    assert not violations, "Sequential admonition blocks of the same type should be merged:\n" + "\n".join(violations)
