"""Backward-compatible API: run_full_analysis, run_batch_analysis, ChunkingParameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import run_batch


@dataclass(frozen=True)
class ChunkingParameters:
    gamma: float = 0.9
    coupling: float = 0.03
    n_iter: int = 100


def run_full_analysis(
    filepath: str | Path,
    sequence_type: str = "blue",
    gamma: float = 0.9,
    C: float = 0.03,
    n_iter: int = 100,
    n_permutations: int = 100,
    random_state: int | None = None,
) -> dict:
    """Legacy API: Run Wymbs/Mucha analysis; returns old-format dict."""
    from .methods import get_runner

    community_runner = get_runner("community_network")
    result = community_runner(
        filepath,
        sequence_type=sequence_type,
        gamma=gamma,
        coupling=C,
        n_iter=n_iter,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return {
        "filepath": result.source_file,
        "sequence_type": result.sequence_type,
        "n_blocks": result.n_blocks,
        "block_ids": result.block_ids,
        "parameters": result.parameters,
        "ikis": {},
        "multilayer_result": {},
        "partition_map": {},
        "metrics": result.trials_df,
        "validation": result.validation or {},
    }


def run_batch_analysis(
    input_dir: str | Path = "SRT",
    output_dir: str | Path = "outputs",
    pattern: str = "*.csv",
    sequence_type: str = "blue",
    gamma: float = 0.9,
    C: float = 0.03,
    n_iter: int = 20,
    n_permutations: int = 20,
    random_state: int | None = 42,
    limit: int | None = None,
) -> dict:
    """Legacy API: Batch community_network; writes to output_dir (no method subdir)."""
    res = run_batch(
        method_name="community_network",
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
        sequence_type=sequence_type,
        limit=limit,
        random_state=random_state,
        gamma=gamma,
        coupling=C,
        n_iter=n_iter,
        n_permutations=n_permutations,
    )
    return {
        "summary_path": res["summary_path"],
        "trials_path": res["trials_path"],
        "errors_path": res["errors_path"],
        "params_path": res["parameters_path"],
        "progress_log_path": str(Path(res["summary_path"]).parent / "progress.log"),
        "n_files_total": res["n_files_total"],
        "n_files_success": res["n_files_success"],
        "n_files_failed": res["n_files_failed"],
    }
