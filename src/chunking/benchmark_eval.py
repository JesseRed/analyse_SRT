"""
Benchmark evaluation: merge summary/trials from multiple methods, optional ARI.

Use after run_benchmark() to get comparison tables (per file × method) and
optionally Adjusted Rand Index between methods per file when both output chunk boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_benchmark_summary(benchmark_dir: str | Path) -> pd.DataFrame:
    """
    Load benchmark_summary.csv from a benchmark run (one row per source_file × method).
    """
    path = Path(benchmark_dir) / "benchmark_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Run with --benchmark first.")
    return pd.read_csv(path)


def load_method_trials(benchmark_dir: str | Path, method_name: str) -> pd.DataFrame:
    """Load trials.csv for one method from benchmark_dir/<method_name>/."""
    path = Path(benchmark_dir) / method_name / "trials.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path)


def comparison_table(
    benchmark_dir: str | Path,
    summary_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return a long-format table: one row per (source_file, method) with selected
    summary columns. Default: source_file, method, n_blocks, mean_n_chunks, and
    any column present in the summary that looks like a metric.
    """
    df = load_benchmark_summary(benchmark_dir)
    if summary_columns is not None:
        missing = [c for c in summary_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")
        return df[summary_columns].copy()

    # Default: core id columns + common metrics
    id_cols = ["source_file", "method", "n_blocks"]
    metric_candidates = [
        "mean_n_chunks",
        "mean_q_single_trial",
        "mean_phi",
        "empirical_q_multitrial",
        "p_value_permutation",
    ]
    cols = id_cols + [c for c in metric_candidates if c in df.columns]
    return df[[c for c in cols if c in df.columns]].copy()


def _boundaries_to_labels(boundaries: list[int], n_positions: int = 7) -> list[int]:
    """Convert chunk_boundaries (list of boundary positions 1..7) to cluster labels 0..K-1."""
    if not boundaries:
        return list(range(n_positions))
    boundaries = sorted(set(boundaries))
    labels = []
    c = 0
    for pos in range(1, n_positions + 1):
        if pos in boundaries:
            c += 1
        labels.append(c)
    return labels


def adjusted_rand_index(labels_a: list[int], labels_b: list[int]) -> float:
    """
    Compute Adjusted Rand Index between two partitions (same length).
    Uses sklearn.metrics.adjusted_rand_score if available, else 0.0.
    """
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(labels_a, labels_b))
    except ImportError:
        return float("nan")


def ari_per_file(
    benchmark_dir: str | Path,
    method_a: str,
    method_b: str,
    n_positions: int = 7,
) -> pd.DataFrame:
    """
    For each source_file, compute ARI between method_a and method_b chunk boundaries.
    Returns DataFrame with columns: source_file, n_blocks, ari_mean, ari_min, ari_max.
    Each trial's chunk_boundaries are converted to a partition; ARI is computed per
    trial then averaged per file.
    """
    trials_a = load_method_trials(benchmark_dir, method_a)
    trials_b = load_method_trials(benchmark_dir, method_b)

    if "chunk_boundaries" not in trials_a.columns or "chunk_boundaries" not in trials_b.columns:
        raise ValueError("Both trials must have 'chunk_boundaries' column.")

    def parse_boundaries(x: Any) -> list[int]:
        if isinstance(x, str):
            import ast
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        if isinstance(x, list):
            return [int(i) for i in x]
        return []

    def boundaries_to_labels(boundaries: list[int]) -> list[int]:
        return _boundaries_to_labels(boundaries, n_positions)

    merged = trials_a[["source_file", "block_number", "chunk_boundaries"]].merge(
        trials_b[["source_file", "block_number", "chunk_boundaries"]],
        on=["source_file", "block_number"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    merged["labels_a"] = merged["chunk_boundaries_a"].map(
        lambda x: boundaries_to_labels(parse_boundaries(x))
    )
    merged["labels_b"] = merged["chunk_boundaries_b"].map(
        lambda x: boundaries_to_labels(parse_boundaries(x))
    )
    merged["ari"] = merged.apply(
        lambda row: adjusted_rand_index(row["labels_a"], row["labels_b"]),
        axis=1,
    )
    agg = merged.groupby("source_file").agg(
        n_blocks=("ari", "count"),
        ari_mean=("ari", "mean"),
        ari_min=("ari", "min"),
        ari_max=("ari", "max"),
    ).reset_index()
    return agg


__all__ = [
    "load_benchmark_summary",
    "load_method_trials",
    "comparison_table",
    "ari_per_file",
    "adjusted_rand_index",
]
