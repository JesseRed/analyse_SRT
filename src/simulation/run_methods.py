"""Run registered chunking methods on simulated data (parallel over files).

Writes the standard output contract (summary.csv / trials.csv / errors.csv / meta.json)
to <results>/<family>/<method>/<sequence>/ so that ``evaluate`` (or any other tool that
reads the chunking outputs) can be applied unchanged. Null-model validation is switched
off (n_null_runs=0): calibration is assessed empirically on the null family A.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_METHODS = ["community_network", "change_point_pelt", "rational_chunking", "hcrp_lm", "hsmm_chunking"]


def _method_params(method: str, config_path: Optional[Path]) -> Dict[str, Any]:
    """Parameters from methods_config.json, restricted to what run_analysis accepts."""
    import inspect

    from ..chunking.methods import get_runner

    params: Dict[str, Any] = {}
    if config_path and config_path.exists():
        params.update(json.loads(config_path.read_text(encoding="utf-8")).get(method, {}))
    params["n_null_runs"] = 0
    accepted = inspect.signature(get_runner(method)).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        unknown = sorted(k for k in params if k not in accepted)
        if unknown:
            print(f"[warn] {method}: ignoring config keys not accepted by run_analysis: {unknown}")
        params = {k: v for k, v in params.items() if k in accepted}
    return params


def _run_one(task: Tuple[str, str, str, Dict[str, Any], int]) -> Dict[str, Any]:
    from ..chunking._base import result_to_summary_row, result_to_trials_df
    from ..chunking.methods import get_runner
    from ..chunking.run import _augment_summary_row_meta, _augment_trials_df_meta

    method, file_path, seq, params, seed = task
    params = dict(params)
    params["random_state"] = seed
    try:
        result = get_runner(method)(file_path, sequence_type=seq, **params)
        return {
            "summary": _augment_summary_row_meta(result_to_summary_row(result)),
            "trials": _augment_trials_df_meta(result_to_trials_df(result)),
        }
    except Exception as exc:  # recorded in errors.csv, like run_batch
        return {"error": {"source_file": file_path, "error": f"{type(exc).__name__}: {exc}",
                          "trace": traceback.format_exc(limit=2)}}


def method_available(method: str) -> Optional[str]:
    """Return an error message if the method cannot be imported in this environment."""
    try:
        from ..chunking.methods import get_runner

        get_runner(method)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def run_methods(
    sim_dir: str | Path,
    results_dir: str | Path,
    families: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    sequences: Optional[List[str]] = None,
    config_path: Optional[str | Path] = "methods_config.json",
    jobs: int = 8,
    seed: int = 42,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    sim_dir, results_dir = Path(sim_dir), Path(results_dir)
    cfg = Path(config_path) if config_path else None
    methods = methods or DEFAULT_METHODS
    sequences = sequences or ["blue", "green", "yellow"]
    fam_dirs = [d for d in sorted(sim_dir.iterdir()) if (d / "data").is_dir()]
    if families:
        fam_dirs = [d for d in fam_dirs if d.name in families]
    report: Dict[str, Any] = {}
    for method in methods:
        problem = method_available(method)
        if problem:
            print(f"[skip] {method} not available here: {problem}")
            report[method] = {"skipped": problem}
            continue
        params = _method_params(method, cfg)
        for fam in fam_dirs:
            files = sorted((fam / "data").glob("*.csv"))
            if limit:
                files = files[:limit]
            for seq in sequences:
                out = results_dir / fam.name / method / seq
                out.mkdir(parents=True, exist_ok=True)
                tasks = [(method, str(f), seq, params, seed + i) for i, f in enumerate(files, start=1)]
                t0 = time.time()
                if jobs > 1:
                    with ProcessPoolExecutor(max_workers=jobs) as ex:
                        results = list(ex.map(_run_one, tasks, chunksize=2))
                else:
                    results = [_run_one(t) for t in tasks]
                summaries = [r["summary"] for r in results if "summary" in r]
                trials = [r["trials"] for r in results if "trials" in r]
                errors = [r["error"] for r in results if "error" in r]
                pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
                (pd.concat(trials, ignore_index=True) if trials else pd.DataFrame()).to_csv(out / "trials.csv", index=False)
                pd.DataFrame(errors, columns=["source_file", "error", "trace"]).to_csv(out / "errors.csv", index=False)
                meta = {
                    "method_name": method, "family": fam.name, "sequence_type": seq,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "n_files_total": len(files), "n_files_success": len(summaries),
                    "n_files_failed": len(errors), "elapsed_seconds": time.time() - t0,
                    "parameters": params,
                }
                (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"[{fam.name}] {method}/{seq}: ok={len(summaries)} failed={len(errors)} "
                      f"({time.time() - t0:.0f}s)")
                report.setdefault(method, {})[f"{fam.name}/{seq}"] = meta
    return report
