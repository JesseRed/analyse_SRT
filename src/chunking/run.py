"""Run chunking analysis: single file, batch, or benchmark across methods."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ._base import ChunkingResult, result_to_summary_row, result_to_trials_df
from .methods import get_runner, list_methods


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parents[2],
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def run_single_file(
    method_name: str,
    filepath: str | Path,
    output_dir: str | Path,
    sequence_type: str = "blue",
    **method_params: Any,
) -> ChunkingResult:
    """
    Run one chunking method on one file and write results to output_dir.

    Creates output_dir if needed; writes meta.json, parameters.json,
    summary.csv, trials.csv, errors.csv (empty if success).
    """
    filepath = Path(filepath)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    runner = get_runner(method_name)
    result = runner(
        filepath,
        sequence_type=sequence_type,
        **method_params,
    )

    summary_row = result_to_summary_row(result)
    trials_df = result_to_trials_df(result)

    meta = {
        "method_name": result.method_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_file": str(filepath),
        "sequence_type": sequence_type,
        "n_blocks": result.n_blocks,
    }
    (out_path / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (out_path / "parameters.json").write_text(
        json.dumps(result.parameters, indent=2), encoding="utf-8"
    )
    pd.DataFrame([summary_row]).to_csv(out_path / "summary.csv", index=False)
    trials_df.to_csv(out_path / "trials.csv", index=False)
    pd.DataFrame(columns=["source_file", "error"]).to_csv(
        out_path / "errors.csv", index=False
    )

    return result


def run_batch(
    method_name: str,
    input_dir: str | Path = "SRT",
    output_dir: str | Path = "outputs",
    pattern: str = "*.csv",
    sequence_type: str = "blue",
    limit: int | None = None,
    random_state: int | None = 42,
    progress_log: bool = True,
    **method_params: Any,
) -> dict[str, Any]:
    """
    Run one chunking method on all matching files. Writes to output_dir:
    meta.json, parameters.json, summary.csv, trials.csv, errors.csv, progress.log.
    """
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob(pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No files found: {input_path / pattern}")

    runner = get_runner(method_name)
    summary_rows = []
    trial_frames = []
    error_rows = []
    progress_log_path = out_path / "progress.log" if progress_log else None
    if progress_log_path and progress_log_path.exists():
        progress_log_path.unlink()

    total_files = len(files)
    start_time = time.time()

    def log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}\n"
        if progress_log_path:
            progress_log_path.open("a", encoding="utf-8").write(line)
        print(msg)

    # Inject per-file random_state if method accepts it
    for idx, file_path in enumerate(files, start=1):
        params = dict(method_params)
        if "random_state" in params and params["random_state"] is not None:
            params["random_state"] = params["random_state"] + idx

        try:
            result = runner(
                file_path,
                sequence_type=sequence_type,
                **params,
            )
            summary_rows.append(result_to_summary_row(result))
            trial_frames.append(result_to_trials_df(result))
            status = "ok"
        except Exception as exc:
            error_rows.append({"source_file": str(file_path), "error": str(exc)})
            status = "failed"

        elapsed = time.time() - start_time
        eta = (total_files - idx) / (idx / elapsed) if idx else 0
        log(
            f"[{idx}/{total_files}] {status} file='{file_path.name}' "
            f"success={len(summary_rows)} failed={len(error_rows)} "
            f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
        )

    summary_df = pd.DataFrame(summary_rows)
    trial_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    errors_df = pd.DataFrame(error_rows)

    full_params = {
        "method_name": method_name,
        "input_dir": str(input_path),
        "pattern": pattern,
        "sequence_type": sequence_type,
        "limit": limit,
        "random_state": random_state,
        "n_files_total": len(files),
        "n_files_success": len(summary_df),
        "n_files_failed": len(errors_df),
        **method_params,
    }

    meta = {
        "method_name": method_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(input_path),
        "pattern": pattern,
        "sequence_type": sequence_type,
        "n_files_total": len(files),
        "n_files_success": len(summary_df),
        "n_files_failed": len(errors_df),
        "elapsed_seconds": time.time() - start_time,
    }

    (out_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_path / "parameters.json").write_text(
        json.dumps(full_params, indent=2), encoding="utf-8"
    )
    summary_df.to_csv(out_path / "summary.csv", index=False)
    trial_df.to_csv(out_path / "trials.csv", index=False)
    errors_df.to_csv(out_path / "errors.csv", index=False)
    if progress_log_path:
        log(
            f"Batch finished success={len(summary_df)} failed={len(errors_df)} "
            f"elapsed={_format_seconds(time.time() - start_time)}"
        )

    return {
        "summary_path": str(out_path / "summary.csv"),
        "trials_path": str(out_path / "trials.csv"),
        "errors_path": str(out_path / "errors.csv"),
        "meta_path": str(out_path / "meta.json"),
        "parameters_path": str(out_path / "parameters.json"),
        "n_files_total": len(files),
        "n_files_success": len(summary_df),
        "n_files_failed": len(errors_df),
    }


def run_benchmark(
    input_dir: str | Path = "SRT",
    output_dir: str | Path = "outputs",
    pattern: str = "*.csv",
    sequence_type: str = "blue",
    method_list: list[str] | None = None,
    limit: int | None = None,
    random_state: int | None = 42,
    combine_summary: bool = True,
    **method_params: Any,
) -> dict[str, Any]:
    """
    Run all (or selected) chunking methods on the same file set.

    Writes outputs to output_dir/<method_name>/ for each method.
    If combine_summary is True, also writes output_dir/benchmark_summary.csv
    with one row per (source_file, method) and all summary metrics (long format).
    """
    input_path = Path(input_dir)
    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    methods = method_list if method_list is not None else list_methods()
    if not methods:
        raise ValueError("No chunking methods registered.")

    all_summary_rows = []
    run_results = {}

    for method_name in methods:
        method_out = base_out / method_name
        result = run_batch(
            method_name=method_name,
            input_dir=input_path,
            output_dir=method_out,
            pattern=pattern,
            sequence_type=sequence_type,
            limit=limit,
            random_state=random_state,
            progress_log=True,
            **method_params,
        )
        run_results[method_name] = result

        if combine_summary:
            summary_df = pd.read_csv(result["summary_path"])
            all_summary_rows.append(summary_df)

    if combine_summary and all_summary_rows:
        combined = pd.concat(all_summary_rows, ignore_index=True)
        combined_path = base_out / "benchmark_summary.csv"
        combined.to_csv(combined_path, index=False)

    return {
        "output_dir": str(base_out),
        "methods": methods,
        "run_results": run_results,
        "benchmark_summary_path": str(base_out / "benchmark_summary.csv")
        if combine_summary and all_summary_rows
        else None,
    }
