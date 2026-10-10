"""Estimate the empirical parameters of the simulator from the real SRT files.

The result (``configs/simulation/empirical_calibration.json``) is the single source of
all "realistic" default values used by the generator, so the provenance of every
number is reproducible: ``python -m src.simulation calibrate``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .design import KEYS, YELLOW_POOL

DEFAULT_CALIBRATION_PATH = Path("configs/simulation/empirical_calibration.json")


def _read_real(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    df.columns = [str(c).strip() for c in df.columns]
    df["sequence"] = df["sequence"].astype(str).str.strip().str.lower()
    return df


def _session_of(path: Path) -> Optional[int]:
    m = re.search(r"_FRA_(\d+)", path.name)
    return int(m.group(1)) if m else None


def collect_records(files: List[Any]) -> Dict[str, Any]:
    """Flatten SRT files (paths or already loaded DataFrames) into per-IKI and per-block records."""
    iki_rows = []
    block_rows = []
    schedules: Counter = Counter()
    for fi, path in enumerate(files):
        if isinstance(path, pd.DataFrame):
            df = path
        else:
            try:
                df = _read_real(path)
            except Exception:
                continue
        n_blocks = int(df["BlockNumber"].nunique())
        sched = []
        for block, grp in df.groupby("BlockNumber"):
            grp = grp.sort_values("EventNumber")
            seq = str(grp["sequence"].iloc[0])
            sched.append(seq[0])
            t = grp["Time Since Block start"].to_numpy(float)
            hits = grp["isHit"].to_numpy(int)
            targets = grp["target"].to_numpy(int)
            if t.size != 8:
                continue
            ikis = np.diff(t)
            block_rows.append(
                {
                    "file": fi,
                    "block": int(block),
                    "n_blocks": n_blocks,
                    "seq": seq,
                    "first_press": float(t[0]),
                    "n_errors": int((hits == 0).sum()),
                    **{f"hit{p + 1}": int(hits[p]) for p in range(8)},
                }
            )
            all_hit = bool(hits.all())
            for k in range(7):
                iki_rows.append(
                    (fi, int(block), n_blocks, seq, k + 1, int(targets[k]), int(targets[k + 1]),
                     float(ikis[k]), int(hits[k]), all_hit)
                )
        schedules["".join(sched)] += 1
    iki = pd.DataFrame(
        iki_rows,
        columns=["file", "block", "n_blocks", "seq", "pos", "key_from", "key_to", "iki", "hit_prev", "all_hit"],
    )
    iki = iki[iki["iki"] > 0].copy()
    iki["log_iki"] = np.log(iki["iki"])
    return {"iki": iki, "blocks": pd.DataFrame(block_rows), "schedules": schedules}


def _robust_sd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def estimate(records: Dict[str, Any]) -> Dict[str, Any]:
    iki: pd.DataFrame = records["iki"]
    blocks: pd.DataFrame = records["blocks"]
    correct = iki[iki["all_hit"]].copy()

    # --- variance components (log scale) -------------------------------------------
    file_median = correct.groupby("file")["log_iki"].median()
    correct["lc"] = correct["log_iki"] - correct["file"].map(file_median)
    block_mean = correct.groupby(["file", "block"])["lc"].mean()
    correct["lr"] = correct["lc"] - correct.set_index(["file", "block"]).index.map(block_mean).to_numpy()
    ar1 = []
    for _, series in block_mean.groupby(level=0):
        v = series.to_numpy()
        if v.size > 10:
            ar1.append(np.corrcoef(v[:-1], v[1:])[0, 1])
    wide = correct.pivot_table(index=["file", "block"], columns="pos", values="lr")
    lag1 = [wide[k].corr(wide[k + 1]) for k in range(1, 7)]

    # --- positional + key-transition effects from yellow (no designed structure) ------
    yel = correct[correct["seq"] == "yellow"].copy()
    yel["y"] = yel["lc"] - yel.set_index(["file", "block"]).index.map(
        yel.groupby(["file", "block"])["lc"].mean()
    ).to_numpy()
    pairs = [(a, b) for a in KEYS for b in KEYS if a != b]
    pair_index = {p: i for i, p in enumerate(pairs)}
    n = len(yel)
    X = np.zeros((n, 7 + len(pairs)))
    X[np.arange(n), yel["pos"].to_numpy() - 1] = 1.0
    pi = [pair_index.get((a, b), -1) for a, b in zip(yel["key_from"], yel["key_to"])]
    valid = np.array([p >= 0 for p in pi])
    X[np.arange(n)[valid], 7 + np.array(pi)[valid]] = 1.0
    ridge = 1.0
    beta = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ yel["y"].to_numpy())
    pos_eff = beta[:7] - beta[:7].mean()
    trans_eff = beta[7:] - beta[7:].mean()
    transition_effects = {f"{a}{b}": float(trans_eff[pair_index[(a, b)]]) for a, b in pairs}

    # --- sequence offsets and learning ------------------------------------------------
    seq_offset = correct.groupby("seq")["lc"].mean()
    seq_offset = (seq_offset - seq_offset.get("yellow", 0.0)).to_dict()
    learning = {}
    correct["block_frac"] = correct["block"] / correct["n_blocks"]
    for seq, grp in correct.groupby("seq"):
        learning[seq] = float(np.polyfit(grp["block_frac"], grp["lc"], 1)[0])

    # Sequence speed-up net of the key-transition effects the sequence contains, so the
    # generator does not count the transition advantage twice.
    from .design import BLUE, GREEN

    def _mean_trans(seq_str: str) -> float:
        return float(np.mean([transition_effects.get(seq_str[i:i + 2], 0.0) for i in range(7)]))

    t_yellow = float(np.mean([_mean_trans(s) for s in YELLOW_POOL]))
    speedup_net = {
        "blue": float(seq_offset.get("blue", 0.0) - (_mean_trans(BLUE) - t_yellow)),
        "green": float(seq_offset.get("green", 0.0) - (_mean_trans(GREEN) - t_yellow)),
        "yellow": 0.0,
    }

    # Tail asymmetry of the within-file log-IKI distribution (documents that the long left
    # tail of the pooled raw IKIs stems from between-participant differences, not from a
    # separate component of anticipatory presses).
    lc_all = correct["lc"].to_numpy()
    med = float(np.median(lc_all))
    right = float(np.percentile(lc_all, 95) - med)
    left = float(med - np.percentile(lc_all, 5))

    # --- errors -----------------------------------------------------------------------
    err = {}
    for seq, grp in blocks.groupby("seq"):
        err[seq] = {
            "per_position": [float(1 - grp[f"hit{p}"].mean()) for p in range(1, 9)],
            "errors_per_block": float(grp["n_errors"].mean()),
            "frac_error_free_blocks": float((grp["n_errors"] == 0).mean()),
        }
    pes = float(iki[iki["hit_prev"] == 0]["log_iki"].median() - iki[iki["hit_prev"] == 1]["log_iki"].median())

    fp = blocks["first_press"].to_numpy(float)
    fp = fp[(fp > 0.5) & (fp < 20)]
    lfp = np.log(fp)

    quantiles = {}
    for seq, grp in correct.groupby("seq"):
        quantiles[seq] = {
            str(q): float(np.percentile(grp["iki"], q)) for q in (1, 5, 25, 50, 75, 95, 99)
        }
        quantiles[seq]["positional_median"] = [
            float(grp[grp["pos"] == k]["iki"].median()) for k in range(1, 8)
        ]

    schedules = [s for s, _ in records["schedules"].most_common() if set(s) <= {"b", "g", "y"}]

    return {
        "n_files": int(iki["file"].nunique()),
        "n_iki": int(len(iki)),
        "frame_seconds": 1.0 / 60.0,
        "mu_log_iki": float(file_median.median()),
        "sd_between_files": float(file_median.std()),
        "sd_block": float(block_mean.std()),
        "ar1_block": float(np.nanmean(ar1)),
        "sd_residual": float(correct["lr"].std()),
        "sd_residual_robust": _robust_sd(correct["lr"].to_numpy()),
        "lag1_within_block": float(np.nanmean(lag1)),
        "position_effects": [float(x) for x in pos_eff],
        "transition_effects": transition_effects,
        "sequence_offsets": {k: float(v) for k, v in seq_offset.items()},
        "sequence_speedup_net": speedup_net,
        "tail_left_log": float(left),
        "tail_right_log": float(right),
        "learning_slope_per_session": learning,
        "errors": err,
        "post_error_slowing_log": pes,
        "first_press_log_mu": float(np.median(lfp)),
        "first_press_log_sd": float((np.percentile(lfp, 75) - np.percentile(lfp, 25)) / 1.349),
        "iki_quantiles": quantiles,
        "schedules": schedules,
        "yellow_pool": list(YELLOW_POOL),
    }


def run_calibration(
    srt_dir: str | Path = "SRT",
    pattern: str = "VR_ML*.csv",
    n_files: int = 400,
    seed: int = 20260224,
    output_path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> Dict[str, Any]:
    files = sorted(Path(srt_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No real SRT files found in {srt_dir}/{pattern}")
    rng = np.random.default_rng(seed)
    if len(files) > n_files:
        files = [files[i] for i in sorted(rng.choice(len(files), n_files, replace=False))]
    calib = estimate(collect_records(files))
    calib["source"] = {"srt_dir": str(srt_dir), "pattern": pattern, "n_files_sampled": len(files), "seed": seed}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    return calib


def load_calibration(path: str | Path = DEFAULT_CALIBRATION_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Calibration file {p} not found. Run: python -m src.simulation calibrate"
        )
    return json.loads(p.read_text(encoding="utf-8"))
