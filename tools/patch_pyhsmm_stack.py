#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path

def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(path.read_text())
        print(f"[backup] {bak}")

def replace_in_file(path: Path, pattern: str, repl: str, flags=0) -> int:
    txt = path.read_text()
    new, n = re.subn(pattern, repl, txt, flags=flags)
    if n:
        backup(path)
        path.write_text(new)
        print(f"[patch] {path}  ({n} replacements)")
    return n

def patch_logsumexp(root: Path) -> None:
    # replace: from scipy.misc import logsumexp -> from scipy.special import logsumexp
    pattern = r"from scipy\.misc import logsumexp"
    repl = "from scipy.special import logsumexp"
    for p in root.rglob("*.py"):
        if "site-packages" not in str(p):
            continue
        if "pybasicbayes" not in str(p):
            continue
        replace_in_file(p, pattern, repl)

def inner1d_replacement() -> str:
    return (
        "import numpy as np\n"
        "def inner1d(a, b):\n"
        "    \"\"\"Replacement for deprecated numpy.core.umath_tests.inner1d.\n"
        "    Computes inner product along last axis.\"\"\"\n"
        "    a = np.asarray(a)\n"
        "    b = np.asarray(b)\n"
        "    return np.einsum('...i,...i->...', a, b)\n"
    )

def patch_inner1d_specific(venv: Path) -> None:
    # Files known to import umath_tests.inner1d
    targets = [
        venv / "lib/python3.10/site-packages/pybasicbayes/util/stats.py",
        venv / "lib/python3.10/site-packages/pybasicbayes/distributions/gaussian.py",
        venv / "lib/python3.10/site-packages/pyhsmm/util/stats.py",
    ]
    pat = r"^from numpy\.core\.umath_tests import inner1d\s*$"
    repl = inner1d_replacement()
    for f in targets:
        if f.exists():
            replace_in_file(f, pat, repl, flags=re.MULTILINE)
        else:
            print(f"[skip] not found: {f}")

def patch_np_Inf_if_present(venv: Path) -> None:
    # Only if someone later installs numpy>=2; harmless otherwise.
    f = venv / "lib/python3.10/site-packages/pybasicbayes/util/stats.py"
    if f.exists():
        replace_in_file(f, r"np\.Inf", "np.inf")

def main() -> None:
    venv = Path(os.environ.get("VIRTUAL_ENV", ""))
    if not venv:
        raise SystemExit("VIRTUAL_ENV not set. Activate .venv_hsmm first.")
    site = venv / "lib/python3.10/site-packages"
    if not site.exists():
        raise SystemExit(f"Expected site-packages not found: {site}")

    patch_logsumexp(site)
    patch_inner1d_specific(venv)
    patch_np_Inf_if_present(venv)

    print("[ok] patching done")

if __name__ == "__main__":
    main()