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

    REQUIRED_COLUMNS = ["BlockNumber", "EventNumber", "isHit", "sequence"]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        found_cols = sorted(df.columns.tolist())
        raise ValueError(
            f"Missing required columns {missing} in file '{filepath}'. "
            f"Found: {found_cols}. Expected at least: {REQUIRED_COLUMNS}"
        )

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


def shuffle_ikis(ikis_dict: dict[int, np.ndarray], random_state: int | None = None) -> dict[int, np.ndarray]:
    """
    Randomly shuffle the temporal sequence of IKIs within each block.
    Maintains the distribution of IKIs for that participant/block but destroys temporal structure.
    """
    rng = np.random.default_rng(random_state)
    shuffled_dict = {}
    for block_id, ikis in ikis_dict.items():
        shuffled_dict[block_id] = rng.permutation(ikis)
    return shuffled_dict
