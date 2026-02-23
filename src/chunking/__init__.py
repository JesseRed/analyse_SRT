"""
Chunking analysis package: multiple methods, unified interface and output.

Usage:
  python -m src.chunking --method community_network --input-dir SRT --output-dir outputs
  python -m src.chunking --benchmark --input-dir SRT --output-dir outputs/benchmark
"""

from __future__ import annotations

# Base types and data (used by methods and run)
from ._base import (
    ChunkingResult,
    ChunkingMethod,
    result_to_summary_row,
    result_to_trials_df,
)
from ._data import (
    EXPECTED_PRESSES_PER_BLOCK,
    load_srt_file,
    extract_ikis,
)
from .methods import (
    REGISTRY,
    get_runner,
    list_methods,
)
from .run import run_single_file, run_batch, run_benchmark, merge_sequence_summaries
from ._compat import (
    ChunkingParameters,
    run_full_analysis,
    run_batch_analysis,
)


def _community_network_unavailable(*_: object, **__: object) -> None:
    raise ImportError(
        "community_network dependencies are not available in this environment."
    )


# Backward compat: expose community_network internals at package level.
try:
    from .methods import community_network

    build_trial_network = community_network.build_trial_network
    run_multilayer_community_detection = community_network.run_multilayer_community_detection
    compute_single_trial_modularity = community_network.compute_single_trial_modularity
    compute_chunk_metrics = community_network.compute_chunk_metrics
    statistical_validation = community_network.statistical_validation
except Exception:
    build_trial_network = _community_network_unavailable
    run_multilayer_community_detection = _community_network_unavailable
    compute_single_trial_modularity = _community_network_unavailable
    compute_chunk_metrics = _community_network_unavailable
    statistical_validation = _community_network_unavailable

# Benchmark evaluation (optional)
from . import benchmark_eval  # noqa: F401
from . import output_evaluation  # noqa: F401

__all__ = [
    "ChunkingResult",
    "ChunkingMethod",
    "result_to_summary_row",
    "result_to_trials_df",
    "EXPECTED_PRESSES_PER_BLOCK",
    "load_srt_file",
    "extract_ikis",
    "REGISTRY",
    "get_runner",
    "list_methods",
    "run_single_file",
    "run_batch",
    "run_benchmark",
    "merge_sequence_summaries",
    "ChunkingParameters",
    "run_full_analysis",
    "run_batch_analysis",
    "build_trial_network",
    "run_multilayer_community_detection",
    "compute_single_trial_modularity",
    "compute_chunk_metrics",
    "statistical_validation",
    "benchmark_eval",
    "output_evaluation",
]
