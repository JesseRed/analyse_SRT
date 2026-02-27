"""Run chunking analysis: single file, batch, or benchmark across methods."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import logging
import sys
import traceback

from ._base import ChunkingResult, result_to_summary_row, result_to_trials_df
from .methods import get_runner, list_methods


def setup_logging(output_dir: Path, log_name: str = "run.log") -> logging.Logger:
    """Configure dual logging to console and a file in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_name

    logger = logging.getLogger("chunking")
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicates in interactive environments
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "unknown"


def _parse_source_file_meta(filepath: str | Path) -> dict[str, Any]:
    """
    Parse participant/session metadata from source file names.

    Expected stem pattern (common in this project):
        <prefix>_<YYYYMMDD>_FRA_<day>[_fertig]
    """
    stem = Path(filepath).stem
    out: dict[str, Any] = {
        "participant_id": stem,
        "session": "",
        "day_index": None,
    }

    match = re.match(
        r"^(?P<prefix>.+?)_(?P<date>\d{8})_FRA_(?P<day>\d+)(?:_fertig)?$",
        stem,
    )
    if not match:
        return out

    prefix = str(match.group("prefix"))
    day_idx = int(match.group("day"))
    parts = prefix.split("_")
    if (
        len(parts) >= 4
        and parts[0].isalpha()
        and parts[1].isalpha()
        and parts[2].isdigit()
    ):
        participant_id = "_".join(parts[3:]) or prefix
    else:
        participant_id = prefix

    out["participant_id"] = participant_id
    out["session"] = f"FRA_{day_idx}"
    out["day_index"] = day_idx
    return out


def _augment_summary_row_meta(summary_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary_row)
    source_file = str(row.get("source_file", ""))
    row.update(_parse_source_file_meta(source_file))
    return row


def _augment_trials_df_meta(trials_df: pd.DataFrame) -> pd.DataFrame:
    df = trials_df.copy()
    if "source_file" not in df.columns or df.empty:
        return df

    source_values = df["source_file"].astype(str)
    unique_sources = source_values.unique().tolist()
    meta_map = {src: _parse_source_file_meta(src) for src in unique_sources}
    df["participant_id"] = source_values.map(
        lambda s: str(meta_map.get(s, {}).get("participant_id", ""))
    )
    df["session"] = source_values.map(lambda s: str(meta_map.get(s, {}).get("session", "")))
    df["day_index"] = source_values.map(
        lambda s: meta_map.get(s, {}).get("day_index", None)
    )
    return df


def _write_validation_artifact(
    output_dir: Path,
    source_file: str,
    validation: dict[str, Any] | None,
) -> None:
    if not validation:
        return
    stem = _sanitize_name(Path(source_file).stem)
    target = output_dir / "artifacts" / stem
    target.mkdir(parents=True, exist_ok=True)
    (target / "validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )


def _write_trials_artifact(
    output_dir: Path,
    source_file: str,
    trials_df: pd.DataFrame,
) -> None:
    """Save participant-specific trials CSV for later subject-wise analysis."""
    stem = _sanitize_name(Path(source_file).stem)
    target = output_dir / "artifacts" / stem
    target.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(target / "trials.csv", index=False)


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
) -> ChunkingResult | None:
    """
    Run one chunking method on one file and write results to output_dir.

    Creates output_dir if needed; writes meta.json, parameters.json,
    summary.csv, trials.csv, errors.csv (empty if success), and run.log.
    """
    filepath = Path(filepath)
    out_path = Path(output_dir)
    logger = setup_logging(out_path, "run.log")

    try:
        logger.info(f"Starting analysis: method={method_name} file={filepath.name}")
        runner = get_runner(method_name)
        result = runner(
            filepath,
            sequence_type=sequence_type,
            **method_params,
        )

        summary_row = _augment_summary_row_meta(result_to_summary_row(result))
        trials_df = _augment_trials_df_meta(result_to_trials_df(result))

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
        if result.validation:
            (out_path / "validation.json").write_text(
                json.dumps(result.validation, indent=2),
                encoding="utf-8",
            )
            _write_validation_artifact(
                output_dir=out_path,
                source_file=str(filepath),
                validation=result.validation,
            )

        # Subject-wise trials (new requirement for better longitudinal analysis)
        if method_name in {"hcrp_lm", "hsmm_chunking", "community_network", "change_point_pelt", "rational_chunking"}:
            _write_trials_artifact(
                output_dir=out_path,
                source_file=str(filepath),
                trials_df=trials_df,
            )

        logger.info("Analysis successfully completed.")
        return result

    except Exception as exc:
        logger.error(f"Analysis failed: {exc}")
        logger.error(traceback.format_exc())
        pd.DataFrame([{"source_file": str(filepath), "error": str(exc)}]).to_csv(
            out_path / "errors.csv", index=False
        )
        return None


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
    total_files = len(files)
    start_time = time.time()
    logger = setup_logging(out_path, "progress.log")

    # Inject per-file random_state if method accepts it
    for idx, file_path in enumerate(files, start=1):
        params = dict(method_params)
        if "random_state" in params and params["random_state"] is not None:
            params["random_state"] = params["random_state"] + idx

        status = "failed"
        try:
            result = runner(
                file_path,
                sequence_type=sequence_type,
                **params,
            )
            summary_rows.append(_augment_summary_row_meta(result_to_summary_row(result)))
            trial_frames.append(_augment_trials_df_meta(result_to_trials_df(result)))
            _write_validation_artifact(
                output_dir=out_path,
                source_file=result.source_file,
                validation=result.validation,
            )
            # Subject-wise trials (new requirement for better longitudinal analysis)
            if method_name in {"hcrp_lm", "hsmm_chunking", "community_network", "change_point_pelt", "rational_chunking"}:
                _write_trials_artifact(
                    output_dir=out_path,
                    source_file=result.source_file,
                    trials_df=trial_frames[-1],
                )
            status = "ok"
        except Exception as exc:
            error_rows.append({"source_file": str(file_path), "error": str(exc)})
            logger.error(f"Failed file='{file_path.name}': {exc}")
            logger.debug(traceback.format_exc())

        elapsed = time.time() - start_time
        avg_speed = idx / elapsed if elapsed > 0 else 0
        eta = (total_files - idx) / avg_speed if avg_speed > 0 else 0
        logger.info(
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

    meta_path = out_path / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_path / "parameters.json").write_text(
        json.dumps(full_params, indent=2), encoding="utf-8"
    )
    summary_df.to_csv(out_path / "summary.csv", index=False)
    trial_df.to_csv(out_path / "trials.csv", index=False)
    errors_df.to_csv(out_path / "errors.csv", index=False)
    
    logger.info(
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


def merge_sequence_summaries(method_output_dir: str | Path) -> str:
    """
    Merge blue/green/yellow summary.csv files and compute structured-vs-yellow deltas.

    Expected structure:
        <method_output_dir>/blue/summary.csv
        <method_output_dir>/green/summary.csv
        <method_output_dir>/yellow/summary.csv
    """
    base = Path(method_output_dir)
    # 1. Look for sequences
    seq_paths = {seq: base / seq / "summary.csv" for seq in ("blue", "green", "yellow")}
    active_seqs = [seq for seq, path in seq_paths.items() if path.exists()]

    # 2. Fallback: Check if summary.csv is in the root (for flat structures)
    direct_sum = base / "summary.csv"
    if not active_seqs and direct_sum.exists():
        # Try to infer sequence from file content or use 'default'
        try:
            df = pd.read_csv(direct_sum)
            seq = df["sequence_type"].iloc[0] if "sequence_type" in df.columns and not df.empty else "unknown"
        except Exception:
            seq = "unknown"
        active_seqs = [seq]
        seq_paths = {seq: direct_sum}

    if not active_seqs:
        raise FileNotFoundError(
            f"No summary.csv found in subdirectories or root of {base}."
        )

    keys = ["source_file", "participant_id", "session", "day_index"]
    merged: pd.DataFrame | None = None
    from pandas.errors import EmptyDataError
    for seq in active_seqs:
        try:
            df = pd.read_csv(seq_paths[seq])
        except (EmptyDataError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        for key in keys:
            if key not in df.columns:
                df[key] = pd.NA
        keep_cols = keys
        metric_cols = [c for c in df.columns if c not in keep_cols + ["sequence_type", "method"]]
        seq_df = df[keep_cols + metric_cols].copy()
        # Only rename if we have multiple sequences or it's a known sequence
        if len(active_seqs) > 1 or seq != "unknown":
            rename_map = {col: f"{col}_{seq}" for col in metric_cols}
            seq_df = seq_df.rename(columns=rename_map)
        
        if merged is None:
            merged = seq_df
        else:
            merged = merged.merge(seq_df, on=keep_cols, how="outer")

    if merged is None:
        raise ValueError(f"No summary rows found in {base}.")

    if "mean_n_chunks_blue" in merged.columns and "mean_n_chunks_yellow" in merged.columns:
        merged["delta_mean_n_chunks_blue_vs_yellow"] = (
            merged["mean_n_chunks_blue"] - merged["mean_n_chunks_yellow"]
        )
    if "mean_n_chunks_green" in merged.columns and "mean_n_chunks_yellow" in merged.columns:
        merged["delta_mean_n_chunks_green_vs_yellow"] = (
            merged["mean_n_chunks_green"] - merged["mean_n_chunks_yellow"]
        )

    out_path = base / "combined_summary.csv"
    merged.to_csv(out_path, index=False)
    return str(out_path)


def run_benchmark(
    input_dir: str | Path = "SRT",
    output_dir: str | Path = "outputs",
    pattern: str = "*.csv",
    sequence_type: str = "blue",
    method_list: list[str] | None = None,
    limit: int | None = None,
    random_state: int | None = 42,
    combine_summary: bool = True,
    per_method_params: dict[str, dict[str, Any]] | None = None,
    **common_params: Any,
) -> dict[str, Any]:
    """
    Run all (or selected) chunking methods on the same file set.

    Writes outputs to output_dir/<method_name>/ for each method.
    If combine_summary is True, also writes output_dir/benchmark_summary.csv
    with one row per (source_file, method) and all summary metrics (long format).
    """
    input_path = Path(input_dir)
    base_out = Path(output_dir)
    logger = setup_logging(base_out, "benchmark.log")

    methods = method_list if method_list is not None else list_methods()
    if not methods:
        raise ValueError("No chunking methods registered.")

    logger.info(f"Starting benchmark: input_dir={input_path} methods={methods}")
    all_summary_rows = []
    run_results = {}

    for method_name in methods:
        method_out = base_out / method_name
        
        # Merge common params with method-specific config
        params = dict(common_params)
        if per_method_params and method_name in per_method_params:
            params.update(per_method_params[method_name])
            
        try:
            logger.info(f"Running method: {method_name}")
            result = run_batch(
                method_name=method_name,
                input_dir=input_path,
                output_dir=method_out,
                pattern=pattern,
                sequence_type=sequence_type,
                limit=limit,
                random_state=random_state,
                progress_log=True,
                **params,
            )
            run_results[method_name] = result

            if combine_summary:
                summary_df = pd.read_csv(result["summary_path"])
                all_summary_rows.append(summary_df)
        except Exception as exc:
            logger.error(f"Failed to run method '{method_name}': {exc}")
            logger.error(traceback.format_exc())
            run_results[method_name] = {"error": str(exc)}

    if combine_summary and all_summary_rows:
        combined = pd.concat(all_summary_rows, ignore_index=True)
        combined_path = base_out / "benchmark_summary.csv"
        combined.to_csv(combined_path, index=False)
        logger.info(f"Combined benchmark summary written to {combined_path}")

    logger.info("Benchmark completed.")
    return {
        "output_dir": str(base_out),
        "methods": methods,
        "run_results": run_results,
        "benchmark_summary_path": str(base_out / "benchmark_summary.csv")
        if combine_summary and all_summary_rows
        else None,
    }
