"""Shared data pipeline: load SRT files and extract IKIs. Used by all chunking methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_PRESSES_PER_BLOCK = 8


def load_srt_file(filepath: str | Path) -> pd.DataFrame:
    """Load one SRT CSV file and coerce key columns into numeric types."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"SRT file not found: {path}")

    df = pd.read_csv(path, sep=";", decimal=",")
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [
        "BlockNumber",
        "EventNumber",
        "Time Since Block start",
        "isHit",
        "target",
        "pressed",
        "sequence",
    ]
    if missing := [c for c in required_cols if c not in df.columns]:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "BlockNumber",
        "EventNumber",
        "Time Since Block start",
        "isHit",
        "target",
        "pressed",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sequence"] = df["sequence"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=numeric_cols + ["sequence"]).reset_index(drop=True)
    df["BlockNumber"] = df["BlockNumber"].astype(int)
    df["EventNumber"] = df["EventNumber"].astype(int)
    df["isHit"] = df["isHit"].astype(int)
    return df


def extract_ikis(
    df: pd.DataFrame,
    sequence_type: str = "blue",
    expected_presses_per_block: int = EXPECTED_PRESSES_PER_BLOCK,
) -> dict[int, np.ndarray]:
    """
    Extract IKIs per block for a chosen sequence.

    Returns:
        Mapping {block_number: np.ndarray of IKIs}
    """
    seq = sequence_type.lower().strip()
    filtered = df[(df["isHit"] == 1) & (df["sequence"] == seq)].copy()

    ikis_by_block: dict[int, np.ndarray] = {}
    for block_number, block_df in filtered.groupby("BlockNumber"):
        block_df = block_df.sort_values("EventNumber")
        times = block_df["Time Since Block start"].to_numpy(dtype=float)
        if times.size != expected_presses_per_block:
            continue
        ikis = np.diff(times)
        if ikis.size != expected_presses_per_block - 1:
            continue
        if np.any(ikis <= 0):
            continue
        ikis_by_block[int(block_number)] = ikis

    if not ikis_by_block:
        return {}

    block_ids = sorted(ikis_by_block)
    matrix = np.vstack([ikis_by_block[b] for b in block_ids])
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    stds = np.where(stds == 0, np.nan, stds)
    z_scores = np.abs((matrix - means) / stds)

    # Keep trials where each IKI position is within 3 SD
    # (matches Wymbs-style outlier handling).
    keep_mask = np.nan_to_num(z_scores, nan=0.0) <= 3.0
    keep_rows = keep_mask.all(axis=1)

    return {
        block_id: ikis_by_block[block_id]
        for keep, block_id in zip(keep_rows, block_ids)
        if keep
    }
