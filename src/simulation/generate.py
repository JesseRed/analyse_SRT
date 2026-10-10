"""Generate a simulation tier: data files, ground truth and manifest per family."""

from __future__ import annotations

import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..chunking._data import extract_ikis, load_srt_file
from .calibrate import load_calibration
from .config import SimulationConfig, load_config
from .generator import simulate_session, window_modal_boundaries
from .io import file_stem, participant_uid, write_srt_csv
from .scenarios import build_cells


def _seed_sequence(base_seed: int, family: str, cell_index: int, rep: int, pid: int, part: int) -> np.random.SeedSequence:
    # crc32 instead of hash(): stable across processes and Python versions.
    return np.random.SeedSequence(base_seed, spawn_key=(zlib.crc32(family.encode()), cell_index, rep, pid, part))


def _generate_participant(task: Tuple) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    (family, cell_index, rep, pid, n_sessions, base_seed, calib_path, out_dir, start_date) = task
    calib = load_calibration(calib_path)
    cell = next(c for c in build_cells(calib, [family]) if c.index == cell_index)
    uid = participant_uid(cell.code, rep, pid)
    spec_ss = _seed_sequence(base_seed, family, cell_index, rep, pid, 0)
    spec = cell.make(np.random.default_rng(spec_ss), uid)

    manifest_rows: List[Dict[str, Any]] = []
    truth_frames: List[pd.DataFrame] = []
    for session in range(1, n_sessions + 1):
        ss = _seed_sequence(base_seed, family, cell_index, rep, pid, session)
        result = simulate_session(spec, session, n_sessions, calib, np.random.default_rng(ss))
        stem = file_stem(family, cell_index, uid, session, start_date)
        path = Path(out_dir) / "data" / f"{stem}.csv"
        write_srt_csv(result.events, path)

        # Which blocks survive the shared pipeline (error-free + 3-SD filter)? Round trip
        # through the real loader so the truth reflects exactly what methods see.
        df = load_srt_file(path)
        included = set()
        for seq in ("blue", "green", "yellow"):
            included |= {(seq, b) for b in extract_ikis(df, sequence_type=seq)}
        truth = result.truth.copy()
        truth["included_by_extract"] = [
            (s, int(b)) in included for s, b in zip(truth["sequence_type"], truth["block_number"])
        ]
        truth["window_modal_boundaries"] = window_modal_boundaries(truth).to_numpy()
        truth.insert(0, "source_stem", stem)
        truth.insert(1, "family", family)
        truth.insert(2, "cell", cell.code)
        truth.insert(3, "participant_uid", uid)
        truth.insert(4, "rep", rep)
        truth.insert(5, "session", session)
        truth_frames.append(truth)

        manifest_rows.append(
            {
                "source_stem": stem,
                "file": str(path),
                "family": family,
                "cell_index": cell_index,
                "cell": cell.code,
                "rep": rep,
                "pid": pid,
                "participant_uid": uid,
                "session": session,
                "n_sessions": n_sessions,
                "seed_entropy": base_seed,
                "seed_spawn_key": json.dumps(list(ss.spawn_key)),
                "factors": json.dumps(cell.factors, default=str),
                **spec.to_flat_dict(),
            }
        )
    return manifest_rows, truth_frames


def generate(config: SimulationConfig, jobs: int = 1, families: Optional[List[str]] = None) -> Dict[str, Any]:
    calib = load_calibration(config.calibration)
    summary: Dict[str, Any] = {}
    for family, size in config.families.items():
        if families and family not in families:
            continue
        out_dir = config.output_dir / family
        (out_dir / "data").mkdir(parents=True, exist_ok=True)
        cells = build_cells(calib, [family])
        if size.cells:
            cells = [c for c in cells if c.code in size.cells]
        tasks = [
            (family, c.index, rep, pid, size.n_sessions, config.base_seed, str(config.calibration),
             str(out_dir), config.start_date)
            for c in cells
            for rep in range(1, size.n_reps + 1)
            for pid in range(1, size.n_participants + 1)
        ]
        manifest: List[Dict[str, Any]] = []
        truth: List[pd.DataFrame] = []
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                results = list(ex.map(_generate_participant, tasks, chunksize=4))
        else:
            results = [_generate_participant(t) for t in tasks]
        for rows, frames in results:
            manifest.extend(rows)
            truth.extend(frames)
        manifest_df = pd.DataFrame(manifest).sort_values("source_stem")
        truth_df = pd.concat(truth, ignore_index=True).sort_values(["source_stem", "block_number"])
        manifest_df.to_csv(out_dir / "manifest.csv", index=False)
        truth_df.to_csv(out_dir / "ground_truth.csv", index=False)
        cell_table = pd.DataFrame(
            [{"cell_index": c.index, "cell": c.code, **{k: json.dumps(v) if isinstance(v, list) else v
                                                          for k, v in c.factors.items()}} for c in cells]
        )
        cell_table.to_csv(out_dir / "cells.csv", index=False)
        summary[family] = {"n_cells": len(cells), "n_files": len(manifest_df), "n_blocks": len(truth_df)}
        print(f"[{family}] cells={len(cells)} files={len(manifest_df)} blocks={len(truth_df)} -> {out_dir}")

    meta = {
        "tier": config.tier,
        "base_seed": config.base_seed,
        "calibration": str(config.calibration),
        "families": {k: asdict(v) for k, v in config.families.items()},
        "generated": summary,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "simulation_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return summary


def generate_from_file(path: str | Path, jobs: int = 1, families: Optional[List[str]] = None) -> Dict[str, Any]:
    return generate(load_config(path), jobs=jobs, families=families)
