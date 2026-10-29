"""Provenance for results bundles: versions, git state, host and job (no JAX import)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io import default_provenance

__all__ = ["collect_provenance"]


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def collect_provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect provenance for the root attributes of a bundle.

    Git information is taken from the repository that contains the arachne source (fail-safe:
    empty when git or the repository is unavailable).  Versions come from
    ``importlib.metadata``, so nothing is imported.

    Args:
        extra: Entries that override or extend the collected ones.

    Returns:
        Dict with ``created_utc``, ``arachne_version``, ``git_commit``, ``git_dirty``,
        ``command_line``, ``hostname``, ``python_version``, ``jax_version``,
        ``numpy_version``, ``slurm_job_id`` and any ``extra`` keys.
    """
    prov = default_provenance()
    src = Path(__file__).resolve().parent
    commit = _git(["rev-parse", "HEAD"], src)
    if commit:
        prov["git_commit"] = commit
        status = _git(["status", "--porcelain", "--untracked-files=no"], src)
        prov["git_dirty"] = bool(status)
    prov["command_line"] = shlex.join(sys.argv) if sys.argv else ""
    prov["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")
    prov.update(extra or {})
    return prov
