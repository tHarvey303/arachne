"""The results reader, plots and viewer must not import JAX or Equinox.

Only ``arachne/results/writer.py`` may (it evaluates the forward model).  This is a static AST
scan of the module sources; importing ``arachne`` itself still pulls in JAX, which is accepted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "arachne"
RESULTS_MODULES = (
    "__init__",
    "schema",
    "io",
    "reader",
    "regions",
    "display",
    "testing",
    "metadata",
    "provenance",
    "cli",
    "__main__",
)
FORBIDDEN = ("jax", "jaxlib", "equinox")


def _files() -> list[Path]:
    files = [SRC / "results" / f"{m}.py" for m in RESULTS_MODULES]
    for sub in ("viz", "gui"):
        files += sorted((SRC / sub).glob("*.py"))
    return [f for f in files if f.exists()]


def _forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # relative imports: the writer is the only JAX module in results/
                names = ["writer"] if (node.module or "").startswith("writer") else []
                names += [f"writer:{a.name}" for a in node.names if a.name == "writer"]
            else:
                names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root in FORBIDDEN or root.startswith("writer") or name == "arachne.results.writer":
                bad.append(f"{path.name}:{node.lineno} imports {name}")
    return bad


def test_scan_finds_results_modules():
    """The scan covers the P0 results modules."""
    names = {f.name for f in _files()}
    assert {"schema.py", "reader.py", "regions.py", "display.py", "testing.py"} <= names


@pytest.mark.parametrize("path", _files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_jax_imports(path):
    """No scanned module imports jax, jaxlib, equinox or the writer."""
    assert _forbidden_imports(path) == []


def test_scanner_catches_jax(tmp_path):
    """The scanner itself flags JAX and Equinox imports."""
    f = tmp_path / "m.py"
    f.write_text("import numpy\nfrom jax import numpy as jnp\nimport equinox as eqx\n")
    assert len(_forbidden_imports(f)) == 2
