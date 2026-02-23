from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    import igraph as ig
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "python-igraph is required for chunking analysis. "
        "Install with: pip install python-igraph"
    ) from exc

try:
    import leidenalg as la
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "leidenalg is required for chunking analysis. "
        "Install with: pip install leidenalg"
    ) from exc


EXPECTED_PRESSES_PER_BLOCK = 8
DEFAULT_GAMMA = 0.9
DEFAULT_COUPLING = 0.03


@dataclass(frozen=True)
class ChunkingParameters:
    gamma: float = DEFAULT_GAMMA
    coupling: float = DEFAULT_COUPLING
    n_iter: int = 100


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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

    return {block_id: ikis_by_block[block_id] for keep, block_id in zip(keep_rows, block_ids) if keep}


def build_trial_network(ikis: np.ndarray) -> ig.Graph:
    """Build one weighted chain graph from IKIs of a single trial."""
    ikis = np.asarray(ikis, dtype=float)
    if ikis.ndim != 1 or ikis.size < 2:
        raise ValueError("IKI vector must be 1D with at least 2 elements.")

    n_nodes = int(ikis.size)
    pairwise_abs_diff = np.abs(ikis[:, None] - ikis[None, :])
    d_max = float(pairwise_abs_diff.max())

    if d_max == 0.0:
        weights = np.ones(n_nodes - 1, dtype=float)
    else:
        weights = np.array(
            [(d_max - abs(ikis[i] - ikis[i + 1])) / d_max for i in range(n_nodes - 1)],
            dtype=float,
        )

    edges = [(i, i + 1) for i in range(n_nodes - 1)]
    graph = ig.Graph(n=n_nodes, edges=edges, directed=False)
    graph.es["weight"] = weights.tolist()
    # `find_partition_temporal` needs a stable id across slices for each node position.
    graph.vs["id"] = list(range(n_nodes))
    graph.vs["name"] = [f"IKI_{i + 1}" for i in range(n_nodes)]
    return graph


def run_multilayer_community_detection(
    graphs: list[ig.Graph],
    gamma: float = DEFAULT_GAMMA,
    C: float = DEFAULT_COUPLING,
    n_iter: int = 100,
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Run temporal multilayer community detection using leidenalg.

    Uses Mucha-style temporal coupling via `find_partition_temporal`.
    """
    if not graphs:
        raise ValueError("No trial graphs provided.")

    for graph in graphs:
        if "weight" not in graph.es.attribute_names():
            raise ValueError("Each graph must contain edge attribute 'weight'.")

    rng = np.random.default_rng(random_state)
    quality_scores: list[float] = []
    all_memberships: list[list[list[int]]] = []

    for _ in range(n_iter):
        seed = int(rng.integers(0, 2**31 - 1))
        temporal_out = la.find_partition_temporal(
            graphs,
            la.RBConfigurationVertexPartition,
            interslice_weight=C,
            weights="weight",
            resolution_parameter=gamma,
            seed=seed,
        )

        memberships: list[list[int]]
        quality: float

        # Compatibility across leidenalg versions:
        # - Newer API may return (partitions, interslice_partition_obj)
        # - Older API returns (memberships, quality_float)
        if (
            isinstance(temporal_out, tuple)
            and len(temporal_out) == 2
            and isinstance(temporal_out[0], list)
            and temporal_out[0]
            and hasattr(temporal_out[0][0], "membership")
        ):
            partitions, interslice_partition = temporal_out
            memberships = [p.membership[:] for p in partitions]
            quality = float(
                sum(float(p.quality()) for p in partitions)
                + float(interslice_partition.quality())
            )
        elif (
            isinstance(temporal_out, tuple)
            and len(temporal_out) == 2
            and isinstance(temporal_out[0], list)
        ):
            raw_memberships, raw_quality = temporal_out
            memberships = [list(m) for m in raw_memberships]
            quality = float(raw_quality)
        else:
            raise RuntimeError(
                "Unexpected return format from leidenalg.find_partition_temporal."
            )

        all_memberships.append(memberships)
        quality_scores.append(quality)

    best_idx = int(np.argmax(quality_scores))
    best_memberships = all_memberships[best_idx]
    return {
        "best_memberships": best_memberships,
        "all_memberships": all_memberships,
        "quality_scores": quality_scores,
        "best_quality": float(quality_scores[best_idx]),
        "mean_quality": float(np.mean(quality_scores)),
        "std_quality": float(np.std(quality_scores, ddof=0)),
        "gamma": gamma,
        "coupling": C,
    }


def compute_single_trial_modularity(
    partition_layer: list[int] | np.ndarray,
    graph: ig.Graph,
) -> float:
    """Compute weighted modularity for a single trial partition (Q_single_trial)."""
    membership = np.asarray(partition_layer, dtype=int)
    if membership.ndim != 1 or membership.size != graph.vcount():
        raise ValueError("Partition length must match number of graph nodes.")

    adjacency = np.array(graph.get_adjacency(attribute="weight").data, dtype=float)
    degree = adjacency.sum(axis=1)
    two_m = float(degree.sum())
    if two_m <= 0:
        return 0.0

    same = membership[:, None] == membership[None, :]
    expected = np.outer(degree, degree) / two_m
    q = ((adjacency - expected) * same).sum() / two_m
    return float(q)


def compute_chunk_metrics(
    ikis_dict: dict[int, np.ndarray],
    partitions: dict[int, list[int]] | list[list[int]],
) -> pd.DataFrame:
    """Compute per-trial chunk metrics from IKIs and community labels."""
    block_ids = sorted(ikis_dict)
    if isinstance(partitions, dict):
        membership_map = partitions
    else:
        membership_map = dict(zip(block_ids, partitions))

    rows: list[dict[str, Any]] = []
    for block_id in block_ids:
        ikis = ikis_dict[block_id]
        membership = np.asarray(membership_map[block_id], dtype=int)
        graph = build_trial_network(ikis)
        q_single = compute_single_trial_modularity(membership, graph)
        phi = np.nan if q_single <= 0 else 1.0 / q_single
        boundaries = [
            i + 1 for i in range(len(membership) - 1) if membership[i] != membership[i + 1]
        ]
        rows.append(
            {
                "block_number": block_id,
                "q_single_trial": q_single,
                "phi": phi,
                "n_chunks": int(np.unique(membership).size),
                "chunk_boundaries": boundaries,
                "community_labels": membership.tolist(),
                "ikis": ikis.tolist(),
            }
        )

    out = pd.DataFrame(rows).sort_values("block_number").reset_index(drop=True)
    finite_phi = out["phi"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_phi.empty or float(finite_phi.mean()) == 0.0:
        out["phi_normalized"] = np.nan
    else:
        mean_phi = float(finite_phi.mean())
        out["phi_normalized"] = (out["phi"] - mean_phi) / mean_phi
    return out


def statistical_validation(
    ikis_dict: dict[int, np.ndarray],
    n_permutations: int = 100,
    gamma: float = DEFAULT_GAMMA,
    C: float = DEFAULT_COUPLING,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Compare empirical multilayer modularity to null model (shuffled IKI order)."""
    block_ids = sorted(ikis_dict)
    if not block_ids:
        raise ValueError("No IKIs available for validation.")

    empirical_graphs = [build_trial_network(ikis_dict[b]) for b in block_ids]
    empirical = run_multilayer_community_detection(
        empirical_graphs,
        gamma=gamma,
        C=C,
        n_iter=20,
        random_state=random_state,
    )
    empirical_q = float(empirical["best_quality"])

    rng = np.random.default_rng(random_state)
    null_scores: list[float] = []
    for _ in range(n_permutations):
        shuffled_graphs = []
        for b in block_ids:
            shuffled = rng.permutation(ikis_dict[b])
            shuffled_graphs.append(build_trial_network(shuffled))
        null_result = run_multilayer_community_detection(
            shuffled_graphs,
            gamma=gamma,
            C=C,
            n_iter=1,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        null_scores.append(float(null_result["best_quality"]))

    null_array = np.asarray(null_scores, dtype=float)
    p_permutation = float((np.sum(null_array >= empirical_q) + 1) / (null_array.size + 1))
    t_stat, p_two_sided = stats.ttest_1samp(null_array, popmean=empirical_q)

    return {
        "empirical_q_multitrial": empirical_q,
        "null_q_multitrial_mean": float(np.mean(null_array)),
        "null_q_multitrial_std": float(np.std(null_array, ddof=0)),
        "null_scores": null_scores,
        "p_value_permutation": p_permutation,
        "t_statistic": float(t_stat),
        "p_value_ttest_two_sided": float(p_two_sided),
    }


def run_full_analysis(
    filepath: str | Path,
    sequence_type: str = "blue",
    gamma: float = DEFAULT_GAMMA,
    C: float = DEFAULT_COUPLING,
    n_iter: int = 100,
    n_permutations: int = 100,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Run full Wymbs/Mucha chunking analysis pipeline on one participant file."""
    df = load_srt_file(filepath)
    ikis_dict = extract_ikis(df, sequence_type=sequence_type)
    if not ikis_dict:
        raise ValueError(f"No valid blocks found for sequence='{sequence_type}'.")

    block_ids = sorted(ikis_dict)
    graphs = [build_trial_network(ikis_dict[b]) for b in block_ids]
    multilayer = run_multilayer_community_detection(
        graphs,
        gamma=gamma,
        C=C,
        n_iter=n_iter,
        random_state=random_state,
    )
    partition_map = {
        b: multilayer["best_memberships"][i] for i, b in enumerate(block_ids)
    }
    metrics = compute_chunk_metrics(ikis_dict, partition_map)
    validation = statistical_validation(
        ikis_dict,
        n_permutations=n_permutations,
        gamma=gamma,
        C=C,
        random_state=random_state,
    )

    return {
        "filepath": str(filepath),
        "sequence_type": sequence_type,
        "n_blocks": len(block_ids),
        "block_ids": block_ids,
        "parameters": {
            "gamma": gamma,
            "coupling": C,
            "n_iter": n_iter,
            "n_permutations": n_permutations,
        },
        "ikis": ikis_dict,
        "multilayer_result": multilayer,
        "partition_map": partition_map,
        "metrics": metrics,
        "validation": validation,
    }


def run_batch_analysis(
    input_dir: str | Path = "SRT",
    output_dir: str | Path = "outputs",
    pattern: str = "*.csv",
    sequence_type: str = "blue",
    gamma: float = DEFAULT_GAMMA,
    C: float = DEFAULT_COUPLING,
    n_iter: int = 20,
    n_permutations: int = 20,
    random_state: int | None = 42,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Batch-run chunking analysis across many participant files.

    Writes:
      - chunking_summary.csv (one row per file)
      - chunking_trials.csv (one row per analyzed block/trial)
      - chunking_errors.csv (failed files with reason)
      - chunking_params.json (run parameters)
    """
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob(pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No files found: {input_path / pattern}")

    summary_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    error_rows: list[dict[str, str]] = []
    progress_log_path = out_path / "chunking_progress.log"
    if progress_log_path.exists():
        progress_log_path.unlink()
    total_files = len(files)
    start_time = time.time()

    def log_progress(message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{timestamp}] {message}\n"
        with progress_log_path.open("a", encoding="utf-8") as f:
            f.write(line)
        print(message)

    log_progress(
        "Batch start "
        f"(files={total_files}, sequence={sequence_type}, gamma={gamma}, "
        f"coupling={C}, n_iter={n_iter}, n_permutations={n_permutations})"
    )

    for idx, file_path in enumerate(files, start=1):
        status = "ok"
        err_msg = ""
        try:
            result = run_full_analysis(
                file_path,
                sequence_type=sequence_type,
                gamma=gamma,
                C=C,
                n_iter=n_iter,
                n_permutations=n_permutations,
                random_state=None if random_state is None else random_state + idx,
            )
            metrics = result["metrics"].copy()
            metrics["source_file"] = str(file_path)
            metrics["sequence_type"] = sequence_type
            trial_frames.append(metrics)

            summary_rows.append(
                {
                    "source_file": str(file_path),
                    "sequence_type": sequence_type,
                    "n_blocks": int(result["n_blocks"]),
                    "mean_q_single_trial": float(metrics["q_single_trial"].mean()),
                    "mean_phi": float(metrics["phi"].replace([np.inf, -np.inf], np.nan).mean()),
                    "mean_phi_normalized": float(metrics["phi_normalized"].mean()),
                    "mean_n_chunks": float(metrics["n_chunks"].mean()),
                    "empirical_q_multitrial": float(result["validation"]["empirical_q_multitrial"]),
                    "null_q_multitrial_mean": float(result["validation"]["null_q_multitrial_mean"]),
                    "p_value_permutation": float(result["validation"]["p_value_permutation"]),
                }
            )
        except Exception as exc:  # pragma: no cover - robust batch execution
            status = "failed"
            err_msg = str(exc)
            error_rows.append({"source_file": str(file_path), "error": str(exc)})

        elapsed = time.time() - start_time
        processed = idx
        success_count = len(summary_rows)
        failed_count = len(error_rows)
        files_per_sec = processed / elapsed if elapsed > 0 else 0.0
        eta_total = total_files / files_per_sec if files_per_sec > 0 else float("inf")
        eta_remaining = (total_files - processed) / files_per_sec if files_per_sec > 0 else float("inf")

        msg = (
            f"[{processed}/{total_files}] {status} file='{file_path.name}' "
            f"success={success_count} failed={failed_count} "
            f"elapsed={_format_seconds(elapsed)} "
            f"eta_remaining={_format_seconds(eta_remaining) if np.isfinite(eta_remaining) else 'N/A'} "
            f"eta_total={_format_seconds(eta_total) if np.isfinite(eta_total) else 'N/A'}"
        )
        if status == "failed":
            msg += f" error='{err_msg}'"
        log_progress(msg)

    summary_df = pd.DataFrame(summary_rows)
    trial_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    errors_df = pd.DataFrame(error_rows)

    summary_path = out_path / "chunking_summary.csv"
    trials_path = out_path / "chunking_trials.csv"
    errors_path = out_path / "chunking_errors.csv"
    params_path = out_path / "chunking_params.json"

    summary_df.to_csv(summary_path, index=False)
    trial_df.to_csv(trials_path, index=False)
    errors_df.to_csv(errors_path, index=False)
    params_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_path),
                "pattern": pattern,
                "sequence_type": sequence_type,
                "gamma": gamma,
                "coupling": C,
                "n_iter": n_iter,
                "n_permutations": n_permutations,
                "random_state": random_state,
                "limit": limit,
                "n_files_total": len(files),
                "n_files_success": len(summary_df),
                "n_files_failed": len(errors_df),
                "progress_log": str(progress_log_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed_final = time.time() - start_time
    log_progress(
        "Batch finished "
        f"(success={len(summary_df)}, failed={len(errors_df)}, "
        f"elapsed={_format_seconds(elapsed_final)})"
    )

    return {
        "summary_path": str(summary_path),
        "trials_path": str(trials_path),
        "errors_path": str(errors_path),
        "params_path": str(params_path),
        "progress_log_path": str(progress_log_path),
        "n_files_total": len(files),
        "n_files_success": len(summary_df),
        "n_files_failed": len(errors_df),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SRTT chunking analysis (Wymbs/Mucha multilayer community detection)."
    )
    parser.add_argument("--input-dir", default="SRT", help="Directory containing participant CSV files.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for analysis outputs.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for input files.")
    parser.add_argument(
        "--sequence-type",
        default="blue",
        choices=["blue", "green", "yellow"],
        help="Sequence type to analyze.",
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Intralayer resolution parameter.")
    parser.add_argument("--coupling", type=float, default=DEFAULT_COUPLING, help="Interlayer coupling parameter.")
    parser.add_argument("--n-iter", type=int, default=20, help="Community-detection repeats per file.")
    parser.add_argument("--n-permutations", type=int, default=20, help="Null-model permutations per file.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducible batch runs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    result = run_batch_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        sequence_type=args.sequence_type,
        gamma=args.gamma,
        C=args.coupling,
        n_iter=args.n_iter,
        n_permutations=args.n_permutations,
        random_state=args.seed,
        limit=args.limit,
    )

    print("Batch chunking analysis complete:")
    print(f"- total files:   {result['n_files_total']}")
    print(f"- success files: {result['n_files_success']}")
    print(f"- failed files:  {result['n_files_failed']}")
    print(f"- summary:       {result['summary_path']}")
    print(f"- trials:        {result['trials_path']}")
    print(f"- errors:        {result['errors_path']}")
    print(f"- params:        {result['params_path']}")
    return 0


__all__ = [
    "ChunkingParameters",
    "load_srt_file",
    "extract_ikis",
    "build_trial_network",
    "run_multilayer_community_detection",
    "compute_single_trial_modularity",
    "compute_chunk_metrics",
    "statistical_validation",
    "run_full_analysis",
    "run_batch_analysis",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
