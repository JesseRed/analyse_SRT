"""Rational Chunking algorithm based on Wu et al. (2023)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from typing import Any

from .._base import ChunkingResult, StatisticalValidation
from .._data import extract_ikis, load_srt_file, shuffle_ikis


def _get_all_partitions() -> np.ndarray:
    """Generate all 128 possible boundary masks for 7 IKIs."""
    # 7 positions, 2^7 = 128 combinations
    masks = []
    for i in range(128):
        # Convert i to bit array of length 7
        mask = np.array([int(x) for x in format(i, "07b")], dtype=bool)
        masks.append(mask)
    return np.array(masks)


def _compute_utility(
    partition_masks: np.ndarray,
    log_ikis: np.ndarray,
    error_rates: np.ndarray,
    mu: float,
    delta: float,
    sigma: float,
    kappa: float,
    lam: float,
) -> np.ndarray:
    """
    Compute utility J(R) for each partition mask.
    J(R) = NLL(data|R) + kappa * ErrorCost + lam * ComplexityCost
    """
    n_partitions = partition_masks.shape[0]
    n_blocks = log_ikis.shape[0]
    
    # Complexity Cost: Number of boundaries (1s in mask)
    # Length of 128
    complexity_costs = lam * np.sum(partition_masks, axis=1)
    
    # Error Cost: Simplified to kappa * window_error_rate (constant across partitions for now if we don't have per-partition error model)
    # If we wanted per-partition errors, we'd need a model linking R to specific error positions.
    # For now, we follow Step 7: kappa * window_error_rate
    # This acts as a constant offset in the softmax unless kappa or error_rate varies with R.
    # Note: If constant, it won't affect the posterior P(R|data).
    # However, if we follow "Option 5B", we could make it dependent on chunk starts.
    # Let's stick to the simplest interpretation first.
    avg_error = np.mean(error_rates)
    error_cost = kappa * avg_error
    
    # Timing Likelihood (NLL)
    # model: x = mu + delta * Boundary + epsilon
    # Boundary is 1 if it's a chunk start (boundary), else 0.
    # x - (mu + delta * Boundary) should be ~ N(0, sigma^2)
    
    # log_ikis shape: (n_blocks, 7)
    # partition_masks shape: (128, 7)
    
    # Expand for broadcasting
    # (128, 1, 7)
    masks_exp = partition_masks[:, np.newaxis, :]
    # (1, n_blocks, 7)
    ikis_exp = log_ikis[np.newaxis, :, :]
    
    # Residuals: (128, n_blocks, 7)
    # If mask is 1, mean is mu + delta. If 0, mean is mu.
    means = mu + masks_exp * delta
    residuals = ikis_exp - means
    
    # NLL = sum over blocks and positions
    # log_val = -0.5 * log(2*pi*sigma^2) - 0.5 * (residual/sigma)^2
    # We sum over positions (axis 2) and blocks (axis 1)
    sq_err = np.sum(residuals**2, axis=(1, 2))
    const = 0.5 * np.log(2 * np.pi * sigma**2) * (n_blocks * 7)
    
    nll = const + (0.5 / sigma**2) * sq_err
    
    # Total Utility J(R) for each of the 128 partitions
    return nll + error_cost + complexity_costs


def statistical_validation(
    filepath: str | Path,
    sequence_type: str,
    target_ikis: dict[int, np.ndarray],
    n_null_runs: int = 100,
    random_state: int | None = None,
    **settings: Any,
) -> StatisticalValidation:
    """Compare empirical MAP log-probability to null model (shuffled IKI order)."""
    empirical_res = run_analysis(
        filepath,
        sequence_type,
        _ikis_dict=target_ikis,
        random_state=random_state,
        **settings
    )
    empirical_metric = float(empirical_res.summary_row.get("map_log_prob", 0.0))

    null_scores = []
    for i in range(n_null_runs):
        run_seed = None if random_state is None else random_state + i + 1000
        shuffled_target = shuffle_ikis(target_ikis, random_state=run_seed)
        null_res = run_analysis(
            filepath,
            sequence_type,
            _ikis_dict=shuffled_target,
            random_state=run_seed,
            **settings
        )
        null_scores.append(float(null_res.summary_row.get("map_log_prob", 0.0)))

    null_array = np.asarray(null_scores, dtype=float)
    null_mean = float(np.mean(null_array))
    null_std = float(np.std(null_array, ddof=0))
    p_val = float((np.sum(null_array >= empirical_metric) + 1) / (null_array.size + 1))
    z_score = 0.0 if null_std == 0 else (empirical_metric - null_mean) / null_std

    return StatisticalValidation(
        empirical_metric=empirical_metric,
        null_metric_mean=null_mean,
        null_metric_std=null_std,
        p_value=p_val,
        z_score=z_score,
        n_null_runs=n_null_runs,
        metric_name="map_log_probability",
        null_values=null_scores,
    )


def run_analysis(
    filepath: str | Path,
    sequence_type: str = "blue",
    *,
    window_size: int = 30,
    step: int = 10,
    min_blocks: int = 6,
    lam: float = 1.0,
    kappa: float = 1.0,
    beta: float = 5.0,
    delta_val: float | None = None,
    sigma_val: float | None = None,
    n_null_runs: int = 100,
    random_state: int | None = None,
    _ikis_dict: dict[int, np.ndarray] | None = None,
    **kwargs: Any,
) -> ChunkingResult:
    """Run Rational Chunking analysis."""
    filepath = Path(filepath)
    df = load_srt_file(filepath)
    
    if _ikis_dict is not None:
        ikis_all = _ikis_dict
    else:
        df = load_srt_file(filepath)
        # Filter only correct blocks (as per Step 1)
        correct_mask = df.groupby("BlockNumber")["isHit"].transform("all")
        df_correct = df[correct_mask].copy()
        ikis_all = extract_ikis(df_correct, sequence_type=sequence_type)

    if not ikis_all:
        raise ValueError(f"No valid correct blocks found for {sequence_type}")
        
    # Get error rates per block (from original df)
    if _ikis_dict is None:
        block_errors = df.groupby("BlockNumber")["isHit"].apply(lambda x: 1.0 - np.mean(x)).to_dict()
    else:
        # If we have _ikis_dict, we might not have the original df. Use 0 for now.
        block_errors = {b: 0.0 for b in ikis_all}
    
    # Define windows
    block_ids = sorted(ikis_all.keys())
    if not block_ids:
        raise ValueError("No blocks to analyze.")
        
    start_all = min(block_ids)
    end_all = max(block_ids)
    
    windows = []
    curr = start_all
    while curr + window_size - 1 <= end_all:
        w_start = curr
        w_end = curr + window_size - 1
        w_center = w_start + (window_size - 1) // 2
        
        # Collect blocks in window
        w_blocks = [b for b in block_ids if w_start <= b <= w_end]
        if len(w_blocks) >= min_blocks:
            windows.append({
                "start": w_start,
                "end": w_end,
                "center": w_center,
                "blocks": w_blocks
            })
        curr += step
        
    if not windows:
        raise ValueError("No valid windows found.")

    all_masks = _get_all_partitions()
    
    trial_rows = []
    window_results = []
    
    for w in windows:
        # Extract log-IKIs for this window
        w_ikis = np.array([np.log(ikis_all[b]) for b in w["blocks"]]) # (N, 7)
        w_errs = np.array([block_errors.get(b, 0.0) for b in w["blocks"]])
        
        # Estimate mu and sigma from window if not provided
        mu = np.mean(w_ikis)
        sigma = np.std(w_ikis) if sigma_val is None else sigma_val
        # delta: expected speed-up at boundary. 
        # If not provided, use a robust estimate of peak vs non-peak diff
        if delta_val is None:
            # Simple heuristic: difference between top 20% and remaining 80%
            flat = w_ikis.flatten()
            q = np.percentile(flat, 80)
            delta = np.mean(flat[flat >= q]) - np.mean(flat[flat < q])
            if delta < 0.1: delta = 0.5 # fallback
        else:
            delta = delta_val
            
        # Compute Utility J(R) for each partition
        utilities = _compute_utility(
            all_masks, w_ikis, w_errs, mu, delta, sigma, kappa, lam
        )
        
        # Posterior P(R|data) via softmax
        # P(R) = exp(-beta * J(R)) / sum(exp(-beta * J(R')))
        logits = -beta * utilities
        log_probs = logits - logsumexp(logits)
        probs = np.exp(log_probs)
        
        # Derived metrics
        exp_n_chunks = np.sum(probs * (np.sum(all_masks, axis=1) + 1))
        # Prob(boundary at pos p) = sum(P(R) if mask[p]==1)
        # mask shape (128, 7), probs shape (128,)
        boundary_probs = np.sum(all_masks * probs[:, np.newaxis], axis=0) # (7,)
        
        map_idx = np.argmax(probs)
        map_log_prob = log_probs[map_idx]
        map_mask = all_masks[map_idx]
        map_boundaries = [i + 1 for i, val in enumerate(map_mask) if val]
        
        window_results.append({
            "center_block": w["center"],
            "exp_n_chunks": float(exp_n_chunks),
            "boundary_probs": boundary_probs.tolist(),
            "map_boundaries": map_boundaries,
            "map_log_prob": float(map_log_prob)
        })
        
        # Map back to individual blocks
        for b in w["blocks"]:
            trial_rows.append({
                "block_number": int(b),
                "n_chunks": float(exp_n_chunks), # Reporting expected value as per requirement for "n_chunks" column
                "chunk_boundaries": map_boundaries, # Using MAP boundaries
                "boundary_probs": boundary_probs.tolist()
            })

    # Average trial rows if multiple windows overlap a block (take nearest center)
    trials_df = pd.DataFrame(trial_rows)
    # For each block_number, keep the one from the window with nearest center
    # Actually, the loop above adds all. Let's do a proper mapping.
    block_to_window = {}
    for block_num in block_ids:
        # Find containing windows
        containing = [win for win in windows if win["start"] <= block_num <= win["end"]]
        if not containing: continue
        # Choose nearest center
        best_win = min(containing, key=lambda win: abs(win["center"] - block_num))
        # Find result for this win
        res = [r for r in window_results if r["center_block"] == best_win["center"]][0]
        block_to_window[block_num] = res
        
    final_trial_rows = []
    for bn in sorted(block_to_window):
        r = block_to_window[bn]
        final_trial_rows.append({
            "block_number": bn,
            "n_chunks": r["exp_n_chunks"],
            "chunk_boundaries": r["map_boundaries"],
            "boundary_probs": r["boundary_probs"]
        })
        
    # Perform statistical validation if this is the empirical run
    validation_res: StatisticalValidation | None = None
    if _ikis_dict is None and n_null_runs > 0:
        validation_res = statistical_validation(
            filepath, sequence_type, ikis_all,
            n_null_runs=n_null_runs,
            random_state=random_state,
            window_size=window_size, step=step, min_blocks=min_blocks,
            lam=lam, kappa=kappa, beta=beta,
            delta_val=delta_val, sigma_val=sigma_val
        )

    trials_df = pd.DataFrame(final_trial_rows)
    trials_df["source_file"] = str(filepath)
    trials_df["sequence_type"] = sequence_type
    trials_df["method"] = "rational_chunking"

    summary_row = {
        "source_file": str(filepath),
        "sequence_type": sequence_type,
        "method": "rational_chunking",
        "n_blocks": len(block_ids),
        "mean_n_chunks": float(trials_df["n_chunks"].mean()),
        "avg_boundary_probs": np.mean([r["boundary_probs"] for r in window_results], axis=0).tolist(),
        "map_log_prob": float(np.mean([r["map_log_prob"] for r in window_results])),
    }

    if validation_res:
        summary_row.update({
            "empirical_map_log_prob": validation_res.empirical_metric,
            "null_map_log_prob_mean": validation_res.null_metric_mean,
            "p_value": validation_res.p_value,
            "z_score": validation_res.z_score,
        })

    return ChunkingResult(
        method_name="rational_chunking",
        source_file=str(filepath),
        sequence_type=sequence_type,
        n_blocks=len(block_ids),
        block_ids=block_ids,
        summary_row=summary_row,
        trials_df=trials_df,
        parameters={
            "lam": lam,
            "kappa": kappa,
            "beta": beta,
            "window_size": window_size,
            "step": step,
            "n_null_runs": n_null_runs
        },
        validation=validation_res.__dict__ if validation_res else None,
        algorithm_doc="algorithms/rational_chunking.md"
    )
