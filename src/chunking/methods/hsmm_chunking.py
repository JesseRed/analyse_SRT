"""HSMM-based chunking: Bayesian HDP-HSMM (pyhsmm), one model per file x sequence."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .._base import ChunkingResult, StatisticalValidation
from .._data import extract_ikis, load_srt_file, shuffle_ikis

try:
    import pyhsmm
    import pyhsmm.basic.distributions as distributions
    from pyhsmm import models
    from pyhsmm.internals import hsmm_states as _hsmm_states
except ImportError as exc:
    raise ImportError(
        "pyhsmm is required for hsmm_chunking. Install with: pip install pyhsmm"
    ) from exc

# Patch Python forward sampler: cumulative_obs_potentials returns (array, 0.) but
# the code does offset[state]; support scalar offset (pyhsmm compatibility).
def _hsmm_sample_forwards_log_patched(
    trans_potentials, initial_state_potential,
    cumulative_obs_potentials, dur_potentials, dur_survival_potentails,
    betal, betastarl,
    left_censoring=False, right_censoring=True,
):
    from scipy.special import logsumexp
    from pyhsmm.util.stats import sample_discrete
    T, _ = betal.shape
    stateseq = np.empty(T, dtype=np.int32)
    durations = []
    t = 0
    if left_censoring:
        raise NotImplementedError
    nextstate_unsmoothed = initial_state_potential
    while t < T:
        nextstate_distn_log = nextstate_unsmoothed + betastarl[t]
        nextstate_distn = np.exp(nextstate_distn_log - logsumexp(nextstate_distn_log))
        assert nextstate_distn.sum() > 0
        state = sample_discrete(nextstate_distn)
        dur_logpmf = dur_potentials(t)[:, state]
        obs, offset_val = cumulative_obs_potentials(t)
        obs = obs[:, state]
        offset = offset_val[state] if np.ndim(offset_val) > 0 else offset_val
        durprob = np.random.random()
        dur = 0
        while durprob > 0 and dur < dur_logpmf.shape[0] and t + dur < T:
            p_d = np.exp(
                dur_logpmf[dur] + obs[dur] - offset
                + betal[t + dur, state] - betastarl[t, state]
            )
            assert not np.isnan(p_d)
            durprob -= p_d
            dur += 1
        stateseq[t : t + dur] = state
        durations.append(dur)
        t += dur
        nextstate_log_distn = trans_potentials(t)[state]
    return stateseq, durations


_hsmm_states.hsmm_sample_forwards_log = _hsmm_sample_forwards_log_patched


DEFAULT_N_STATES = 3
DEFAULT_N_STATES_YELLOW = 2
DEFAULT_MAX_DURATION = 7
DEFAULT_WINDOW_SIZE = 30
DEFAULT_STEP = 10
DEFAULT_MIN_BLOCKS = 6
DEFAULT_MIN_WINDOWS_RELIABLE = 3
DEFAULT_N_GIBBS_ITER = 150
SEQ_LEN = 7  # 7 IKIs per block

# Observation dimension (log-IKI per step)
OBS_DIM = 1


def _validate_sequence_type(sequence_type: str) -> str:
    seq = sequence_type.strip().lower()
    if seq not in {"blue", "green", "yellow"}:
        raise ValueError("sequence_type must be one of: blue, green, yellow.")
    return seq


def _yellow_median_per_position(ikis_by_block: dict[int, np.ndarray]) -> np.ndarray:
    """Median log-IKI per position (1..7) across all blocks."""
    if not ikis_by_block:
        raise ValueError("No blocks for yellow median.")
    block_ids = sorted(ikis_by_block)
    matrix = np.vstack([np.log(np.asarray(ikis_by_block[b], dtype=float)) for b in block_ids])
    return np.median(matrix, axis=0)


def _build_sequences(
    ikis_by_block: dict[int, np.ndarray],
    yellow_median: np.ndarray | None,
    baseline_correction: bool,
) -> tuple[list[int], list[np.ndarray]]:
    """
    Build list of sequences (one per block). Each sequence shape (SEQ_LEN,) log-IKI or baseline-corrected.
    Returns (block_ids, list of arrays length SEQ_LEN).
    """
    block_ids = sorted(ikis_by_block)
    sequences: list[np.ndarray] = []
    for b in block_ids:
        log_iki = np.log(np.asarray(ikis_by_block[b], dtype=float))
        if baseline_correction and yellow_median is not None:
            log_iki = log_iki - yellow_median
        sequences.append(log_iki.astype(np.float64))
    return block_ids, sequences


def _build_pyhsmm_model(
    n_max_states: int,
    obs_dim: int,
    max_duration: int,
    random_state: int | None,
):
    """Build WeakLimitHDPHSMM (Python backend) with Gaussian emissions and Poisson durations (1-indexed).
    Uses WeakLimitHDPHSMMPython to avoid C-extension assertion on short sequences."""
    if random_state is not None:
        np.random.seed(random_state)

    obs_hypparams = {
        "mu_0": np.zeros(obs_dim),
        "sigma_0": np.eye(obs_dim),
        "kappa_0": 0.3,
        "nu_0": obs_dim + 5,
    }
    dur_hypparams = {
        "alpha_0": 2 * max_duration,
        "beta_0": 2.0,
    }
    obs_distns = [distributions.Gaussian(**obs_hypparams) for _ in range(n_max_states)]
    dur_distns = [distributions.PoissonDuration(**dur_hypparams) for _ in range(n_max_states)]

    return models.WeakLimitHDPHSMMPython(
        alpha=6.0,
        gamma=6.0,
        init_state_concentration=6.0,
        obs_distns=obs_distns,
        dur_distns=dur_distns,
    )


def _states_to_boundaries(states: np.ndarray) -> list[int]:
    """Chunk boundaries = positions 1..6 where state changes (1-indexed)."""
    boundaries: list[int] = []
    for i in range(len(states) - 1):
        if states[i] != states[i + 1]:
            boundaries.append(i + 1)
    return boundaries


def _window_specs(min_block: int, max_block: int, window_size: int, step: int) -> list[dict]:
    if window_size <= 0 or step <= 0 or max_block - min_block + 1 < window_size:
        return []
    specs: list[dict] = []
    start = min_block
    idx = 0
    while start + window_size - 1 <= max_block:
        end = start + window_size - 1
        center = start + (window_size - 1) // 2
        specs.append({
            "window_index": idx,
            "start_block": start,
            "end_block": end,
            "center_block": center,
        })
        idx += 1
        start += step
    return specs


def _select_blocks_in_window(
    block_ids: list[int],
    sequences: list[np.ndarray],
    start_block: int,
    end_block: int,
) -> tuple[list[int], list[np.ndarray]]:
    """Return (block_ids, sequences) for blocks in [start_block, end_block]."""
    out_ids: list[int] = []
    out_seqs: list[np.ndarray] = []
    for bid, seq in zip(block_ids, sequences):
        if start_block <= bid <= end_block:
            out_ids.append(bid)
            out_seqs.append(seq)
    return out_ids, out_seqs


def statistical_validation(
    filepath: str | Path,
    sequence_type: str,
    target_ikis: dict[int, np.ndarray],
    n_null_runs: int = 100,
    random_state: int | None = None,
    **settings: Any,
) -> StatisticalValidation:
    """Compare empirical log-likelihood to null model (shuffled IKI order)."""
    empirical_res = run_analysis(
        filepath,
        sequence_type,
        _ikis_dict=target_ikis,
        random_state=random_state,
        **settings
    )
    # We need to extract log-likelihood from the model. 
    # Current run_analysis doesn't return it in summary_row. 
    # Let's update run_analysis first.
    empirical_ll = float(empirical_res.summary_row.get("log_likelihood", 0.0))

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
        null_scores.append(float(null_res.summary_row.get("log_likelihood", 0.0)))

    null_array = np.asarray(null_scores, dtype=float)
    null_mean = float(np.mean(null_array))
    null_std = float(np.std(null_array, ddof=0))
    p_val = float((np.sum(null_array >= empirical_ll) + 1) / (null_array.size + 1))
    z_score = 0.0 if null_std == 0 else (empirical_ll - null_mean) / null_std

    return StatisticalValidation(
        empirical_metric=empirical_ll,
        null_metric_mean=null_mean,
        null_metric_std=null_std,
        p_value=p_val,
        z_score=z_score,
        n_null_runs=n_null_runs,
        metric_name="log_likelihood",
        null_values=null_scores,
    )


def run_analysis(
    filepath: str | Path,
    sequence_type: str = "blue",
    *,
    n_states: int | None = None,
    max_duration: int = DEFAULT_MAX_DURATION,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    min_windows_reliable: int = DEFAULT_MIN_WINDOWS_RELIABLE,
    baseline_correction: bool = True,
    n_gibbs_iter: int = DEFAULT_N_GIBBS_ITER,
    n_null_runs: int = 100,
    random_state: int | None = None,
    _ikis_dict: dict[int, np.ndarray] | None = None,
) -> ChunkingResult:
    """Run HSMM-based chunking for one sequence type (blue/green/yellow) using pyhsmm HDP-HSMM."""
    filepath = Path(filepath)
    seq = _validate_sequence_type(sequence_type)
    if n_states is None:
        n_states = DEFAULT_N_STATES_YELLOW if seq == "yellow" else DEFAULT_N_STATES
    if n_states < 2 or max_duration < 1:
        raise ValueError("n_states >= 2 and max_duration >= 1 required.")

    if _ikis_dict is not None:
        target_ikis = _ikis_dict
    else:
        df = load_srt_file(filepath)
        target_ikis = extract_ikis(df, sequence_type=seq)

    if not target_ikis:
        raise ValueError(f"No valid blocks found for sequence='{seq}'.")

    yellow_median: np.ndarray | None = None
    if baseline_correction and seq != "yellow":
        yellow_ikis = extract_ikis(df if _ikis_dict is None else load_srt_file(filepath), sequence_type="yellow")
        if yellow_ikis:
            yellow_median = _yellow_median_per_position(yellow_ikis)

    do_baseline = baseline_correction and seq != "yellow" and yellow_median is not None
    block_ids, sequences = _build_sequences(target_ikis, yellow_median, baseline_correction=do_baseline)
    if len(sequences) < 2:
        raise ValueError(f"Need at least 2 blocks for HSMM fit; got {len(sequences)}.")

    # Nmax for weak-limit HDP: use n_states as upper bound on number of states
    n_max_states = max(n_states, 6)

    if random_state is not None:
        np.random.seed(random_state)

    model = _build_pyhsmm_model(
        n_max_states=n_max_states,
        obs_dim=OBS_DIM,
        max_duration=max_duration,
        random_state=None,  # already set above
    )

    # Add each block as a separate sequence so segment boundaries cannot cross blocks
    for seq_arr in sequences:
        data_2d = np.asarray(seq_arr, dtype=np.float64).reshape(-1, OBS_DIM)
        model.add_data(data_2d, trunc=max_duration)

    # Gibbs sampling (suppress pyhsmm log(0) warning from transition matrix)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in log", module="pyhsmm"
        )
        for _ in range(n_gibbs_iter):
            model.resample_model()

    # Empirical log-likelihood
    log_likelihood = float(model.log_likelihood())

    # Perform statistical validation if this is the empirical run
    validation_res: StatisticalValidation | None = None
    if _ikis_dict is None and n_null_runs > 0:
        validation_res = statistical_validation(
            filepath, seq, target_ikis,
            n_null_runs=n_null_runs,
            random_state=random_state,
            n_states=n_states, max_duration=max_duration,
            window_size=window_size, step=step, min_blocks=min_blocks,
            min_windows_reliable=min_windows_reliable,
            baseline_correction=baseline_correction,
            n_gibbs_iter=n_gibbs_iter
        )

    # State sequence per block -> chunk_boundaries per block
    trial_rows: list[dict] = []
    for bid, state_obj in zip(block_ids, model.states_list):
        stateseq = state_obj.stateseq
        boundaries = _states_to_boundaries(stateseq)
        n_chunks = len(boundaries) + 1
        trial_rows.append({
            "block_number": int(bid),
            "n_chunks": int(n_chunks),
            "chunk_boundaries": list(boundaries),
        })

    trials_df = pd.DataFrame(trial_rows)
    trials_df["source_file"] = str(filepath)
    trials_df["sequence_type"] = seq
    trials_df["method"] = "hsmm_chunking"

    # Window evaluation: use same model's state sequences per block; aggregate by window
    block_ids_all = sorted(set(block_ids))
    window_specs = _window_specs(
        int(min(block_ids_all)),
        int(max(block_ids_all)),
        window_size,
        step,
    )
    boundary_heatmap_rows: list[list[float]] = []
    window_n_chunks_list: list[int] = []
    window_bounds: list[list[int]] = []

    for spec in window_specs:
        w_ids, w_seqs = _select_blocks_in_window(
            block_ids, sequences,
            int(spec["start_block"]),
            int(spec["end_block"]),
        )
        if len(w_seqs) < min_blocks:
            continue
        # Boundary probability per position 1..6 from state sequences (already computed per block)
        counts = np.zeros(6, dtype=float)
        n_chunks_in_window: list[int] = []
        for bid in w_ids:
            row = trials_df[trials_df["block_number"] == bid].iloc[0]
            b = row["chunk_boundaries"]
            n_chunks_in_window.append(row["n_chunks"])
            for pos in b:
                if 1 <= pos <= 6:
                    counts[pos - 1] += 1.0
        prob = (counts / len(w_seqs)).tolist()
        boundary_heatmap_rows.append(prob)
        window_n_chunks_list.append(int(round(np.mean(n_chunks_in_window))))
        window_bounds.append([int(spec["start_block"]), int(spec["end_block"]), int(spec["center_block"])])

    # Drift
    drift_values: list[float] = []
    if len(boundary_heatmap_rows) >= 2:
        mat = np.array(boundary_heatmap_rows)
        for i in range(mat.shape[0] - 1):
            drift_values.append(float(np.sum(np.abs(mat[i + 1] - mat[i]))))

    mean_n_chunks = float(trials_df["n_chunks"].mean())
    n_windows_valid = len(boundary_heatmap_rows)
    reliable = n_windows_valid >= min_windows_reliable
    mean_drift = float(np.mean(drift_values)) if drift_values else float("nan")

    summary_row = {
        "source_file": str(filepath),
        "sequence_type": seq,
        "method": "hsmm_chunking",
        "n_blocks": len(block_ids),
        "mean_n_chunks": mean_n_chunks,
        "n_windows_valid": n_windows_valid,
        "mean_drift": mean_drift,
        "reliable": reliable,
        "log_likelihood": log_likelihood,
    }

    if validation_res:
        summary_row.update({
            "empirical_log_likelihood": validation_res.empirical_metric,
            "null_log_likelihood_mean": validation_res.null_metric_mean,
            "p_value": validation_res.p_value,
            "z_score": validation_res.z_score,
        })

    validation: dict[str, object] = {
        "window_indices": list(range(n_windows_valid)),
        "window_bounds": window_bounds,
        "boundary_heatmap": boundary_heatmap_rows,
        "window_n_chunks": window_n_chunks_list,
        "drift": drift_values,
    }

    parameters = {
        "n_states": n_states,
        "n_max_states": n_max_states,
        "max_duration": max_duration,
        "window_size": window_size,
        "step": step,
        "min_blocks": min_blocks,
        "min_windows_reliable": min_windows_reliable,
        "baseline_correction": baseline_correction,
        "random_state": random_state,
        "n_gibbs_iter": n_gibbs_iter,
    }

    return ChunkingResult(
        method_name="hsmm_chunking",
        source_file=str(filepath),
        sequence_type=seq,
        n_blocks=len(block_ids),
        block_ids=block_ids,
        summary_row=summary_row,
        trials_df=trials_df,
        parameters=parameters,
        validation=validation_res.__dict__ if validation_res else validation,
        algorithm_doc="algorithms/hsmm_chunking.md",
    )
