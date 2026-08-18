"""Enforce descriptive documentation for every function in the source package."""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_every_source_function_has_descriptive_docstring() -> None:
    """Check that every function explains its steps in a nontrivial docstring."""

    missing = []
    for path in (PROJECT_ROOT / "src" / "fe2pef_retina").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node) or ""
                if "What happens" not in docstring or len(docstring.split()) < 15:
                    missing.append(f"{path.name}:{node.lineno}:{node.name}")
    assert not missing, "Functions missing descriptive docstrings: " + ", ".join(missing)
