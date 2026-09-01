"""Build public slim fixtures before collection so parity tests do not skip.

A clean clone used to skip ~56 tests because private ``fixtures/*.npz`` were
absent. ``make infer`` / this hook generate those files from the public
m1-demo-v1 slim checkpoint instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "gen_public_fixtures.py"
_NEEDED = (
    ROOT / "fixtures" / "personacore_parity.npz",
    ROOT / "fixtures" / "personacore_generate_fixture.npz",
    ROOT / "fixtures" / "personacore_eos_fixture.npz",
    ROOT / "fixtures" / "val_windows.npz",
)


def pytest_configure(config):
    if getattr(config.option, "collectonly", False):
        return
    if all(p.is_file() and p.stat().st_size > 0 for p in _NEEDED):
        return
    proc = subprocess.run(
        [sys.executable, str(GEN)],
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "public fixtures were not generated from m1-demo-v1. "
            "Run `make infer` (needs a CPU PyTorch wheel for the PersonaCore oracle)."
        )
