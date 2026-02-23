"""Community detection in temporal networks (Wymbs/Mucha, Leiden multilayer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .._base import ChunkingResult
from .._data import extract_ikis, load_srt_file

try:
    import igraph as ig
except ImportError as exc:
    raise ImportError(
        "python-igraph is required for community_network. "
        "Install with: pip install python-igraph"
    ) from exc

try:
    import leidenalg as la
except ImportError as exc:
    raise ImportError(
        "leidenalg is required for community_network. "
        "Install with: pip install leidenalg"
    ) from exc

DEFAULT_GAMMA = 0.9
DEFAULT_COUPLING = 0.03


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
    graph.vs["id"] = list(range(n_nodes))
    graph.vs["name"] = [f"IKI_{i + 1}" for i in range(n_nodes)]
    return graph


def run_multilayer_community_detection(
    graphs: list[ig.Graph],
    gamma: float = DEFAULT_GAMMA,
    C: float = DEFAULT_COUPLING,
    n_iter: int = 100,
    random_state: int | None = None,
) -> dict:
    """Run temporal multilayer community detection using leidenalg."""
    if not graphs:
        raise ValueError("No trial graphs provided.")

    for graph in graphs:
        if "weight" not in graph.es.attribute_names():
            raise ValueError("Each graph must contain edge attribute 'weight'.")

    rng = np.random.default_rng(random_state)
    quality_scores = []
    all_memberships = []

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
    """Compute weighted modularity for a single trial partition."""
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
    membership_map = (
        partitions
        if isinstance(partitions, dict)
        else dict(zip(block_ids, partitions))
    )

    rows = []
    for block_id in block_ids:
        ikis = ikis_dict[block_id]
        membership = np.asarray(membership_map[block_id], dtype=int)
        graph = build_trial_network(ikis)
        q_single = compute_single_trial_modularity(membership, graph)
        phi = np.nan if q_single <= 0 else 1.0 / q_single
        boundaries = [
            i + 1
            for i in range(len(membership) - 1)
            if membership[i] != membership[i + 1]
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
) -> dict:
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
    null_scores = []
    for _ in range(n_permutations):
        shuffled_graphs = [
            build_trial_network(rng.permutation(ikis_dict[b])) for b in block_ids
        ]
        null_result = run_multilayer_community_detection(
            shuffled_graphs,
            gamma=gamma,
            C=C,
            n_iter=1,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        null_scores.append(float(null_result["best_quality"]))

    null_array = np.asarray(null_scores, dtype=float)
    p_permutation = float(
        (np.sum(null_array >= empirical_q) + 1) / (null_array.size + 1)
    )
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


def run_analysis(
    filepath: str | Path,
    sequence_type: str = "blue",
    *,
    gamma: float = DEFAULT_GAMMA,
    coupling: float = DEFAULT_COUPLING,
    n_iter: int = 100,
    n_permutations: int = 100,
    random_state: int | None = None,
) -> ChunkingResult:
    """Run full Wymbs/Mucha chunking analysis on one participant file."""
    filepath = Path(filepath)
    df = load_srt_file(filepath)
    ikis_dict = extract_ikis(df, sequence_type=sequence_type)
    if not ikis_dict:
        raise ValueError(f"No valid blocks found for sequence='{sequence_type}'.")

    block_ids = sorted(ikis_dict)
    graphs = [build_trial_network(ikis_dict[b]) for b in block_ids]
    multilayer = run_multilayer_community_detection(
        graphs,
        gamma=gamma,
        C=coupling,
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
        C=coupling,
        random_state=random_state,
    )

    summary_row = {
        "source_file": str(filepath),
        "sequence_type": sequence_type,
        "method": "community_network",
        "n_blocks": len(block_ids),
        "mean_n_chunks": float(metrics["n_chunks"].mean()),
        "mean_q_single_trial": float(metrics["q_single_trial"].mean()),
        "mean_phi": float(metrics["phi"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_phi_normalized": float(metrics["phi_normalized"].mean()),
        "empirical_q_multitrial": validation["empirical_q_multitrial"],
        "null_q_multitrial_mean": validation["null_q_multitrial_mean"],
        "p_value_permutation": validation["p_value_permutation"],
    }

    trials = metrics.copy()
    trials["source_file"] = str(filepath)
    trials["sequence_type"] = sequence_type
    trials["method"] = "community_network"

    return ChunkingResult(
        method_name="community_network",
        source_file=str(filepath),
        sequence_type=sequence_type,
        n_blocks=len(block_ids),
        block_ids=block_ids,
        summary_row=summary_row,
        trials_df=trials,
        parameters={
            "gamma": gamma,
            "coupling": coupling,
            "n_iter": n_iter,
            "n_permutations": n_permutations,
            "random_state": random_state,
        },
        validation=validation,
        algorithm_doc="algorithms/community_network.md",
    )
