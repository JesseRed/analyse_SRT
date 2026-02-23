"""Common types and schema for chunking results. All methods return ChunkingResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


# --- Schema (required columns for comparison) ---
# Summary: source_file, sequence_type, method, n_blocks; recommended: mean_n_chunks
# Trials: source_file, sequence_type, method, block_number, n_chunks, chunk_boundaries
#   chunk_boundaries: list of boundary positions 1..7 (between IKI indices)
# Errors: source_file, error


@dataclass
class ChunkingResult:
    """
    Unified result of a chunking analysis on one participant file.

    All methods must return this structure so that run.py can write
    summary.csv, trials.csv, and optional validation/artifacts consistently.
    """

    method_name: str
    """Short method id, e.g. 'community_network', 'change_point_pelt'."""

    source_file: str
    sequence_type: str
    n_blocks: int
    block_ids: list[int]

    summary_row: dict[str, Any]
    """
    One row for summary.csv. Must include at least:
    source_file, sequence_type, method, n_blocks.
    Recommended: mean_n_chunks. Method-specific fields allowed.
    """

    trials_df: pd.DataFrame
    """
    One row per block/trial. Must include at least:
    source_file, sequence_type, method, block_number, n_chunks, chunk_boundaries.
    chunk_boundaries: list of boundary positions (1..7). Method-specific columns allowed.
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    """All run parameters (JSON-serializable) for parameters.json."""

    validation: dict[str, Any] | None = None
    """Optional null-model / permutation results (e.g. p_value_permutation)."""

    algorithm_doc: str | None = None
    """Optional reference to algorithms/<method>.md."""


def result_to_summary_row(result: ChunkingResult) -> dict[str, Any]:
    """Ensure summary row has required keys; add method."""
    row = dict(result.summary_row)
    row.setdefault("source_file", result.source_file)
    row.setdefault("sequence_type", result.sequence_type)
    row.setdefault("method", result.method_name)
    row.setdefault("n_blocks", result.n_blocks)
    return row


def result_to_trials_df(result: ChunkingResult) -> pd.DataFrame:
    """Ensure trials DataFrame has required columns; add method if missing."""
    df = result.trials_df.copy()
    if "source_file" not in df.columns:
        df["source_file"] = result.source_file
    if "sequence_type" not in df.columns:
        df["sequence_type"] = result.sequence_type
    if "method" not in df.columns:
        df["method"] = result.method_name
    return df


class ChunkingMethod(Protocol):
    """Protocol for a chunking method's entry point."""

    def __call__(
        self,
        filepath: str | Path,
        sequence_type: str = "blue",
        **kwargs: Any,
    ) -> ChunkingResult:
        ...
