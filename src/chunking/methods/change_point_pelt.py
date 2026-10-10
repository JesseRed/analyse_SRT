"""Change-point chunking using PELT on position profiles with sliding windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import logging
import numpy as np
import pandas as pd

from .._base import ChunkingResult, StatisticalValidation
from .._data import extract_ikis, load_srt_file, shuffle_ikis

logger = logging.getLogger("chunking")

try:
    import ruptures as rpt
except ImportError as exc:
    raise ImportError(
        "ruptures is required for change_point_pelt. Install with: pip install ruptures"
    ) from exc


DEFAULT_WINDOW_SIZE = 30
DEFAULT_STEP = 10
DEFAULT_MIN_BLOCKS = 6
DEFAULT_N_BOOTSTRAP = 500
DEFAULT_FPR_TARGET = 0.05
DEFAULT_MEAN_CP_TARGET = 0.1
DEFAULT_N_NULL_SPLITS = 32
DEFAULT_PENALTY_GRID_SIZE = 80
DEFAULT_MIN_WINDOWS_RELIABLE = 3
DEFAULT_FALLBACK_PENALTY = 1.0


@dataclass
class WindowAnalysis:
    """Analysis result for one sliding window."""

    window_index: int
    start_block: int
    end_block: int
    center_block: int
    signal: np.ndarray
    chunk_boundaries: list[int]
    n_chunks: int
    boundary_probs: np.ndarray
    bootstrap_n_chunks_mean: float
    bootstrap_n_chunks_std: float


def _validate_sequence_type(sequence_type: str) -> str:
    seq = sequence_type.strip().lower()
    if seq not in {"blue", "green", "yellow"}:
        raise ValueError("sequence_type must be one of: blue, green, yellow.")
    return seq


def _window_specs(min_block: int, max_block: int, window_size: int, step: int) -> list[dict]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0.")
    if step <= 0:
        raise ValueError("step must be > 0.")
    if max_block - min_block + 1 < window_size:
        return []

    specs: list[dict] = []
    start = min_block
    idx = 0
    while start + window_size - 1 <= max_block:
        end = start + window_size - 1
        # Integer center for deterministic block-to-window mapping.
        center = start + (window_size - 1) // 2
        specs.append(
            {
                "window_index": idx,
                "start_block": start,
                "end_block": end,
                "center_block": center,
            }
        )
        idx += 1
        start += step
    return specs


def _select_blocks_in_window(
    ikis_dict: dict[int, np.ndarray],
    start_block: int,
    end_block: int,
) -> dict[int, np.ndarray]:
    return {b: ikis for b, ikis in ikis_dict.items() if start_block <= b <= end_block}


def _median_profile(ikis_dict: dict[int, np.ndarray]) -> np.ndarray:
    if not ikis_dict:
        raise ValueError("Cannot compute profile for empty block set.")
    block_ids = sorted(ikis_dict)
    matrix = np.vstack([np.log(np.asarray(ikis_dict[b], dtype=float)) for b in block_ids])
    return np.median(matrix, axis=0)


def _run_pelt_boundaries(
    signal: np.ndarray,
    penalty: float,
    *,
    cost_model: str = "l2",
    rbf_gamma: float | None = None,
) -> list[int]:
    x = np.asarray(signal, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    if n < 2:
        return []

    if cost_model == "rbf":
        params = {} if rbf_gamma is None else {"gamma": float(rbf_gamma)}
        algo = rpt.Pelt(model="rbf", min_size=1, params=params)
    else:
        algo = rpt.Pelt(model="l2", min_size=1)
    cps = algo.fit(x).predict(pen=penalty)
    # With 7 IKIs, valid boundary positions are 1..6.
    return sorted({int(cp) for cp in cps if 1 <= int(cp) < n})


def _calibrate_penalty(
    signals: list[np.ndarray],
    *,
    fpr_target: float,
    mean_cp_target: float,
    grid_size: int,
    cost_model: str = "l2",
    rbf_gamma: float | None = None,
) -> tuple[float, dict[str, float]]:
    if not signals:
        raise ValueError("No signals available for penalty calibration.")
    if grid_size < 5:
        raise ValueError("penalty_grid_size must be >= 5.")

    centered_vars = [
        float(np.var(np.asarray(sig, dtype=float) - float(np.mean(sig)), ddof=0))
        for sig in signals
    ]
    scale = max(float(np.median(centered_vars)), 1e-6)
    penalties = scale * np.geomspace(1e-3, 1e3, num=grid_size)

    chosen_penalty = float(penalties[-1])
    chosen_rate = 1.0
    chosen_mean_cp = float("inf")

    def _run(sig: np.ndarray, pen: float) -> int:
        return len(
            _run_pelt_boundaries(
                sig, pen, cost_model=cost_model, rbf_gamma=rbf_gamma
            )
        )

    for pen in penalties:
        counts = np.array([_run(sig, float(pen)) for sig in signals], dtype=float)
        rate = float(np.mean(counts > 0))
        mean_cp = float(np.mean(counts))

        if rate <= fpr_target and mean_cp <= mean_cp_target:
            chosen_penalty = float(pen)
            chosen_rate = rate
            chosen_mean_cp = mean_cp
            break

    return chosen_penalty, {
        "calibration_signal_count": float(len(signals)),
        "calibration_scale": scale,
        "calibration_fpr_observed": chosen_rate,
        "calibration_mean_cp_observed": chosen_mean_cp,
    }


def _bootstrap_boundary_probs(
    target_window_blocks: dict[int, np.ndarray],
    yellow_window_blocks: dict[int, np.ndarray] | None,
    *,
    sequence_type: str,
    penalty: float,
    n_bootstrap: int,
    random_state: int | None,
    cost_model: str = "l2",
    rbf_gamma: float | None = None,
) -> tuple[np.ndarray, float, float]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0.")

    rng = np.random.default_rng(random_state)
    target_ids = np.array(sorted(target_window_blocks), dtype=int)
    yellow_ids = (
        np.array(sorted(yellow_window_blocks), dtype=int)
        if yellow_window_blocks is not None
        else np.array([], dtype=int)
    )

    counts = np.zeros(6, dtype=float)
    n_chunks_samples: list[float] = []

    for _ in range(n_bootstrap):
        boot_target = rng.choice(target_ids, size=target_ids.size, replace=True)
        target_mat = np.vstack(
            [np.log(np.asarray(target_window_blocks[int(b)], dtype=float)) for b in boot_target]
        )
        target_profile = np.median(target_mat, axis=0)

        if sequence_type == "yellow":
            signal = target_profile
        else:
            if yellow_window_blocks is None or yellow_ids.size == 0:
                continue
            boot_yellow = rng.choice(yellow_ids, size=yellow_ids.size, replace=True)
            yellow_mat = np.vstack(
                [np.log(np.asarray(yellow_window_blocks[int(b)], dtype=float)) for b in boot_yellow]
            )
            yellow_profile = np.median(yellow_mat, axis=0)
            signal = target_profile - yellow_profile

        boundaries = _run_pelt_boundaries(
            signal, penalty=penalty, cost_model=cost_model, rbf_gamma=rbf_gamma
        )
        for b in boundaries:
            if 1 <= b <= 6:
                counts[b - 1] += 1.0
        n_chunks_samples.append(float(len(boundaries) + 1))

    probs = counts / float(n_bootstrap)
    if not n_chunks_samples:
        return probs, float("nan"), float("nan")
    chunks_arr = np.asarray(n_chunks_samples, dtype=float)
    return probs, float(np.mean(chunks_arr)), float(np.std(chunks_arr, ddof=0))


def _fit_windows_for_sequence(
    sequence_type: str,
    target_ikis: dict[int, np.ndarray],
    yellow_ikis: dict[int, np.ndarray] | None,
    window_specs: list[dict],
    *,
    min_blocks: int,
    penalty: float,
    n_bootstrap: int,
    random_state: int | None,
    cost_model: str = "l2",
    rbf_gamma: float | None = None,
) -> list[WindowAnalysis]:
    windows: list[WindowAnalysis] = []

    for spec in window_specs:
        start_block = int(spec["start_block"])
        end_block = int(spec["end_block"])

        target_window = _select_blocks_in_window(target_ikis, start_block, end_block)
        if len(target_window) < min_blocks:
            continue

        yellow_window: dict[int, np.ndarray] | None = None
        if sequence_type != "yellow":
            if yellow_ikis is None:
                continue
            yellow_window = _select_blocks_in_window(yellow_ikis, start_block, end_block)
            if len(yellow_window) < min_blocks:
                continue

        target_profile = _median_profile(target_window)
        if sequence_type == "yellow":
            signal = target_profile
        else:
            if yellow_window is None:
                continue
            yellow_profile = _median_profile(yellow_window)
            signal = target_profile - yellow_profile

        boundaries = _run_pelt_boundaries(
            signal, penalty=penalty, cost_model=cost_model, rbf_gamma=rbf_gamma
        )
        probs, n_chunks_mean, n_chunks_std = _bootstrap_boundary_probs(
            target_window,
            yellow_window,
            sequence_type=sequence_type,
            penalty=penalty,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            cost_model=cost_model,
            rbf_gamma=rbf_gamma,
        )

        windows.append(
            WindowAnalysis(
                window_index=int(spec["window_index"]),
                start_block=start_block,
                end_block=end_block,
                center_block=int(spec["center_block"]),
                signal=signal,
                chunk_boundaries=boundaries,
                n_chunks=len(boundaries) + 1,
                boundary_probs=probs,
                bootstrap_n_chunks_mean=n_chunks_mean,
                bootstrap_n_chunks_std=n_chunks_std,
            )
        )

    return windows


def _build_null_signals_from_yellow(
    yellow_ikis: dict[int, np.ndarray],
    window_specs: list[dict],
    *,
    min_blocks: int,
    n_null_splits: int,
    random_state: int | None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(random_state)
    signals: list[np.ndarray] = []

    for spec in window_specs:
        yellow_window = _select_blocks_in_window(
            yellow_ikis,
            int(spec["start_block"]),
            int(spec["end_block"]),
        )
        block_ids = np.array(sorted(yellow_window), dtype=int)
        if block_ids.size < min_blocks:
            continue

        for _ in range(n_null_splits):
            shuffled = rng.permutation(block_ids)
            split_idx = shuffled.size // 2
            if split_idx == 0 or split_idx >= shuffled.size:
                continue
            a_ids = shuffled[:split_idx]
            b_ids = shuffled[split_idx:]
            if a_ids.size == 0 or b_ids.size == 0:
                continue

            a_mat = np.vstack(
                [np.log(np.asarray(yellow_window[int(b)], dtype=float)) for b in a_ids]
            )
            b_mat = np.vstack(
                [np.log(np.asarray(yellow_window[int(b)], dtype=float)) for b in b_ids]
            )
            a_profile = np.median(a_mat, axis=0)
            b_profile = np.median(b_mat, axis=0)
            signals.append(a_profile - b_profile)

    return signals


def _map_blocks_to_nearest_windows(
    block_ids: list[int],
    windows: list[WindowAnalysis],
) -> pd.DataFrame:
    rows: list[dict] = []
    if not windows:
        return pd.DataFrame(rows)

    for block_id in sorted(block_ids):
        containing = [w for w in windows if w.start_block <= block_id <= w.end_block]
        if not containing:
            continue
        chosen = min(containing, key=lambda w: abs(w.center_block - block_id))
        rows.append(
            {
                "block_number": int(block_id),
                "window_index": int(chosen.window_index),
                "window_start_block": int(chosen.start_block),
                "window_end_block": int(chosen.end_block),
                "window_center_block": int(chosen.center_block),
                "n_chunks": int(chosen.n_chunks),
                "chunk_boundaries": list(chosen.chunk_boundaries),
            }
        )

    return pd.DataFrame(rows).sort_values("block_number").reset_index(drop=True)


def _compute_drift(boundary_prob_matrix: np.ndarray) -> list[float]:
    if boundary_prob_matrix.size == 0 or boundary_prob_matrix.shape[0] < 2:
        return []
    drift_values: list[float] = []
    for i in range(boundary_prob_matrix.shape[0] - 1):
        drift = float(
            np.sum(np.abs(boundary_prob_matrix[i + 1, :] - boundary_prob_matrix[i, :]))
        )
        drift_values.append(drift)
    return drift_values


def _compute_entropy(probs: np.ndarray) -> float:
    """Compute binary entropy of a boundary probability profile."""
    if probs.size == 0:
        return 0.0
    p = np.clip(probs, 1e-12, 1.0 - 1e-12)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return float(np.mean(h))


def statistical_validation(
    filepath: str | Path,
    sequence_type: str,
    target_ikis: dict[int, np.ndarray],
    yellow_ikis: dict[int, np.ndarray] | None,
    n_null_runs: int = 100,
    random_state: int | None = None,
    **window_settings: Any,
) -> StatisticalValidation:
    """Compare empirical mean_n_chunks to null model (shuffled IKI order)."""
    # Run empirical first to get the baseline mean_n_chunks
    # (Actually we already have the summary_row in run_analysis, but stand-alone is safer)
    empirical_res = run_analysis(
        filepath,
        sequence_type,
        _ikis_dict=target_ikis,
        _yellow_ikis_dict=yellow_ikis,
        random_state=random_state,
        **window_settings
    )
    empirical_val = float(empirical_res.summary_row["mean_n_chunks"])

    null_scores = []
    for i in range(n_null_runs):
        run_seed = None if random_state is None else random_state + i + 1000
        shuffled_target = shuffle_ikis(target_ikis, random_state=run_seed)
        # Note: we usually don't shuffle the baseline (yellow) as it's already a null condition
        # but the prompt says shuffle the participant's IKIs.
        # If we shuffle both, we destroy structure in both. 
        # Usually, null-model means comparing structured to non-structured.
        null_res = run_analysis(
            filepath,
            sequence_type,
            _ikis_dict=shuffled_target,
            _yellow_ikis_dict=yellow_ikis,
            random_state=run_seed,
            **window_settings
        )
        null_scores.append(float(null_res.summary_row["mean_n_chunks"]))

    null_array = np.asarray(null_scores, dtype=float)
    null_mean = float(np.mean(null_array))
    null_std = float(np.std(null_array, ddof=0))
    # For PELT, we expect mean_n_chunks to be HIGHER in structured than in random?
    # Actually, in random data, PELT might find many or zero spurious boundaries depending on penalty.
    # Wymbs/Mucha use Q (modularity). For PELT, let's use mean_n_chunks.
    # If the user says "signifikant höher", then we check null_array <= empirical_val.
    p_val = float((np.sum(null_array >= empirical_val) + 1) / (null_array.size + 1))
    z_score = 0.0 if null_std == 0 else (empirical_val - null_mean) / null_std

    return StatisticalValidation(
        empirical_metric=empirical_val,
        null_metric_mean=null_mean,
        null_metric_std=null_std,
        p_value=p_val,
        z_score=z_score,
        n_null_runs=n_null_runs,
        metric_name="mean_n_chunks",
        null_values=null_scores,
    )


def run_analysis(
    filepath: str | Path,
    sequence_type: str = "blue",
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    fpr_target: float = DEFAULT_FPR_TARGET,
    mean_cp_target: float = DEFAULT_MEAN_CP_TARGET,
    n_null_splits: int = DEFAULT_N_NULL_SPLITS,
    penalty_grid_size: int = DEFAULT_PENALTY_GRID_SIZE,
    min_windows_reliable: int = DEFAULT_MIN_WINDOWS_RELIABLE,
    allow_fallback_penalty: bool = False,
    fallback_penalty: float | None = None,
    cost_model: str = "l2",
    rbf_gamma: float | None = None,
    n_null_runs: int = 100,
    random_state: int | None = None,
    _ikis_dict: dict[int, np.ndarray] | None = None,
    _yellow_ikis_dict: dict[int, np.ndarray] | None = None,
) -> ChunkingResult:
    """
    Run PELT-based change-point chunking with automatic parameter adaptation.
    If calibration fails, it retries with reduced window_size and min_blocks.
    """
    current_window_size = window_size
    current_min_blocks = min_blocks
    
    # Pre-load data once to avoid repeated IO in the retry loop
    if _ikis_dict is not None:
        target_ikis = _ikis_dict
        yellow_ikis = _yellow_ikis_dict
        input_file = str(filepath)
    else:
        df = load_srt_file(filepath)
        seq = _validate_sequence_type(sequence_type)
        target_ikis = extract_ikis(df, sequence_type=seq)
        yellow_ikis = extract_ikis(df, sequence_type="yellow") if seq != "yellow" else None
        input_file = str(filepath)

    while True:
        try:
            return _run_analysis_single_attempt(
                filepath=input_file,
                sequence_type=sequence_type,
                window_size=current_window_size,
                step=step,
                min_blocks=current_min_blocks,
                n_bootstrap=n_bootstrap,
                fpr_target=fpr_target,
                mean_cp_target=mean_cp_target,
                n_null_splits=n_null_splits,
                penalty_grid_size=penalty_grid_size,
                min_windows_reliable=min_windows_reliable,
                allow_fallback_penalty=allow_fallback_penalty,
                fallback_penalty=fallback_penalty,
                cost_model=cost_model,
                rbf_gamma=rbf_gamma,
                n_null_runs=n_null_runs,
                random_state=random_state,
                _ikis_dict=target_ikis,
                _yellow_ikis_dict=yellow_ikis,
            )
        except ValueError as exc:
            msg = str(exc)
            if "No valid windows" in msg or "No valid yellow windows" in msg or "No sliding windows possible" in msg:
                can_reduce = False
                new_ws = current_window_size
                new_mb = current_min_blocks
                
                if current_window_size > 10:
                    new_ws = max(10, current_window_size - 5)
                    can_reduce = True
                if current_min_blocks > 2:
                    new_mb = max(2, current_min_blocks - 1)
                    can_reduce = True
                
                if can_reduce:
                    logger.warning(
                        f"PELT failed for {Path(input_file).name} with ws={current_window_size}, mb={current_min_blocks}. "
                        f"Retrying with ws={new_ws}, mb={new_mb}. Error: {msg}"
                    )
                    current_window_size = new_ws
                    current_min_blocks = new_mb
                    continue
            
            # If we can't reduce further or it's a different error, re-raise
            raise


def _run_analysis_single_attempt(
    filepath: str | Path,
    sequence_type: str = "blue",
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    fpr_target: float = DEFAULT_FPR_TARGET,
    mean_cp_target: float = DEFAULT_MEAN_CP_TARGET,
    n_null_splits: int = DEFAULT_N_NULL_SPLITS,
    penalty_grid_size: int = DEFAULT_PENALTY_GRID_SIZE,
    min_windows_reliable: int = DEFAULT_MIN_WINDOWS_RELIABLE,
    allow_fallback_penalty: bool = False,
    fallback_penalty: float | None = None,
    cost_model: str = "l2",
    rbf_gamma: float | None = None,
    n_null_runs: int = 100,
    random_state: int | None = None,
    _ikis_dict: dict[int, np.ndarray] | None = None,
    _yellow_ikis_dict: dict[int, np.ndarray] | None = None,
) -> ChunkingResult:
    """Core PELT analysis logic for a single set of parameters."""
    filepath = Path(filepath)
    seq = _validate_sequence_type(sequence_type)
    if cost_model not in ("l2", "rbf"):
        raise ValueError("cost_model must be 'l2' or 'rbf'.")
    if min_blocks < 2:
        raise ValueError("min_blocks must be >= 2.")

    if _ikis_dict is not None:
        target_ikis = _ikis_dict
    else:
        df = load_srt_file(filepath)
        target_ikis = extract_ikis(df, sequence_type=seq)

    if not target_ikis:
        raise ValueError(f"No valid blocks found for sequence='{seq}'.")

    yellow_ikis: dict[int, np.ndarray] | None = None
    if seq != "yellow":
        if _yellow_ikis_dict is not None:
            yellow_ikis = _yellow_ikis_dict
        else:
            if _ikis_dict is None:
                # df already loaded
                yellow_ikis = extract_ikis(df, sequence_type="yellow")
            else:
                # Need to load df for yellow if not provided
                df = load_srt_file(filepath)
                yellow_ikis = extract_ikis(df, sequence_type="yellow")
        
        if not yellow_ikis:
            raise ValueError("No valid yellow blocks found for baseline correction.")

    block_ids_all = sorted(
        set(target_ikis).union(set(yellow_ikis) if yellow_ikis is not None else set())
    )
    if not block_ids_all:
        raise ValueError("No valid blocks available after IKI extraction.")

    window_specs = _window_specs(
        min_block=int(min(block_ids_all)),
        max_block=int(max(block_ids_all)),
        window_size=window_size,
        step=step,
    )
    if not window_specs:
        raise ValueError(
            "No sliding windows possible. Increase data length or reduce window_size."
        )

    # 5A: calibrate penalty on yellow profiles (null condition).
    yellow_for_calibration = yellow_ikis if yellow_ikis is not None else target_ikis
    yellow_signals: list[np.ndarray] = []
    for spec in window_specs:
        yellow_window = _select_blocks_in_window(
            yellow_for_calibration,
            int(spec["start_block"]),
            int(spec["end_block"]),
        )
        if len(yellow_window) < min_blocks:
            continue
        yellow_signals.append(_median_profile(yellow_window))

    penalty_fallback_used = False
    if not yellow_signals:
        if not allow_fallback_penalty:
            raise ValueError(
                "No valid yellow windows for penalty calibration. "
                "Try smaller min_blocks/window_size."
            )
        penalty_yellow = float(
            fallback_penalty if fallback_penalty is not None else DEFAULT_FALLBACK_PENALTY
        )
        penalty_fallback_used = True
        yellow_cal_meta = {
            "calibration_signal_count": 0.0,
            "calibration_scale": float("nan"),
            "calibration_fpr_observed": float("nan"),
            "calibration_mean_cp_observed": float("nan"),
            "fallback_used": 1.0,
        }
    else:
        penalty_yellow, yellow_cal_meta = _calibrate_penalty(
            yellow_signals,
            fpr_target=fpr_target,
            mean_cp_target=mean_cp_target,
            grid_size=penalty_grid_size,
            cost_model=cost_model,
            rbf_gamma=rbf_gamma,
        )

    # 5B: structured penalty via random-minus-random null trick.
    if seq == "yellow":
        penalty_structured = penalty_yellow
        struct_cal_meta = {
            "calibration_signal_count": yellow_cal_meta["calibration_signal_count"],
            "calibration_scale": yellow_cal_meta["calibration_scale"],
            "calibration_fpr_observed": yellow_cal_meta["calibration_fpr_observed"],
            "calibration_mean_cp_observed": yellow_cal_meta[
                "calibration_mean_cp_observed"
            ],
        }
    else:
        if yellow_ikis is None:
            raise ValueError("Yellow IKIs required for structured calibration.")
        null_signals = _build_null_signals_from_yellow(
            yellow_ikis,
            window_specs,
            min_blocks=min_blocks,
            n_null_splits=n_null_splits,
            random_state=random_state,
        )
        if not null_signals:
            if not allow_fallback_penalty:
                raise ValueError(
                    "No valid random-minus-random null signals for structured calibration."
                )
            penalty_structured = float(
                fallback_penalty
                if fallback_penalty is not None
                else DEFAULT_FALLBACK_PENALTY
            )
            penalty_fallback_used = True
            struct_cal_meta = {
                "calibration_signal_count": 0.0,
                "calibration_scale": float("nan"),
                "calibration_fpr_observed": float("nan"),
                "calibration_mean_cp_observed": float("nan"),
                "fallback_used": 1.0,
            }
        else:
            penalty_structured, struct_cal_meta = _calibrate_penalty(
                null_signals,
                fpr_target=fpr_target,
                mean_cp_target=mean_cp_target,
                grid_size=penalty_grid_size,
                cost_model=cost_model,
                rbf_gamma=rbf_gamma,
            )

    penalty = penalty_yellow if seq == "yellow" else penalty_structured
    windows = _fit_windows_for_sequence(
        seq,
        target_ikis=target_ikis,
        yellow_ikis=yellow_ikis,
        window_specs=window_specs,
        min_blocks=min_blocks,
        penalty=penalty,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        cost_model=cost_model,
        rbf_gamma=rbf_gamma,
    )

    # Perform statistical validation if this is the empirical run (not called by validation)
    validation_res: StatisticalValidation | None = None
    if _ikis_dict is None and n_null_runs > 0:
        validation_res = statistical_validation(
            filepath, seq, target_ikis, yellow_ikis,
            n_null_runs=n_null_runs,
            random_state=random_state,
            window_size=window_size, step=step, min_blocks=min_blocks,
            n_bootstrap=n_bootstrap, fpr_target=fpr_target,
            mean_cp_target=mean_cp_target, n_null_splits=n_null_splits,
            penalty_grid_size=penalty_grid_size,
            min_windows_reliable=min_windows_reliable,
            allow_fallback_penalty=allow_fallback_penalty,
            fallback_penalty=fallback_penalty, cost_model=cost_model,
            rbf_gamma=rbf_gamma
        )
    if not windows:
        raise ValueError(
            "No valid windows for the selected sequence under current settings "
            "(min_blocks/window_size/step)."
        )

    trials = _map_blocks_to_nearest_windows(sorted(target_ikis), windows)
    if trials.empty:
        raise ValueError("Could not map any blocks to valid windows.")

    boundary_prob_matrix = np.vstack([w.boundary_probs for w in windows])
    drift_values = _compute_drift(boundary_prob_matrix)
    entropy_by_window = [_compute_entropy(w.boundary_probs) for w in windows]
    finite_entropies = [e for e in entropy_by_window if np.isfinite(e)]

    summary_row = {
        "source_file": str(filepath),
        "sequence_type": seq,
        "method": "change_point_pelt",
        "n_blocks": len(target_ikis),
        "n_windows_valid": len(windows),
        "mean_n_chunks": float(trials["n_chunks"].mean()),
        "mean_window_n_chunks": float(np.mean([w.n_chunks for w in windows])),
        "mean_bootstrap_n_chunks": float(
            np.nanmean([w.bootstrap_n_chunks_mean for w in windows])
        ),
        "mean_boundary_entropy": float(np.mean(finite_entropies))
        if finite_entropies
        else float("nan"),
        "mean_drift": float(np.mean(drift_values)) if drift_values else float("nan"),
        "reliable": bool(len(windows) >= int(min_windows_reliable)),
        "penalty_fallback_used": bool(penalty_fallback_used),
        "used_window_size": int(window_size),
        "used_min_blocks": int(min_blocks),
    }
    if validation_res:
        summary_row.update({
            "empirical_mean_n_chunks": validation_res.empirical_metric,
            "null_mean_n_chunks": validation_res.null_metric_mean,
            "p_value": validation_res.p_value,
            "z_score": validation_res.z_score,
        })

    trials = trials.copy()
    trials["source_file"] = str(filepath)
    trials["sequence_type"] = seq
    trials["method"] = "change_point_pelt"

    validation = {
        "window_indices": [int(w.window_index) for w in windows],
        "window_bounds": [
            [int(w.start_block), int(w.end_block), int(w.center_block)] for w in windows
        ],
        "window_chunk_boundaries": [list(w.chunk_boundaries) for w in windows],
        "window_n_chunks": [int(w.n_chunks) for w in windows],
        "boundary_heatmap": boundary_prob_matrix.tolist(),
        "drift": drift_values,
        "entropy_by_window": entropy_by_window,
        "penalty_yellow": float(penalty_yellow),
        "penalty_structured": float(penalty_structured),
    }

    parameters = {
        "window_size": window_size,
        "step": step,
        "min_blocks": min_blocks,
        "n_bootstrap": n_bootstrap,
        "fpr_target": fpr_target,
        "mean_cp_target": mean_cp_target,
        "n_null_splits": n_null_splits,
        "penalty_grid_size": penalty_grid_size,
        "min_windows_reliable": min_windows_reliable,
        "allow_fallback_penalty": allow_fallback_penalty,
        "fallback_penalty": fallback_penalty,
        "cost_model": cost_model,
        "rbf_gamma": rbf_gamma,
        "random_state": random_state,
        "penalty_selected": float(penalty),
        "penalty_yellow": float(penalty_yellow),
        "penalty_structured": float(penalty_structured),
        "penalty_fallback_used": bool(penalty_fallback_used),
        "penalty_calibration_yellow": yellow_cal_meta,
        "penalty_calibration_structured": struct_cal_meta,
    }

    return ChunkingResult(
        method_name="change_point_pelt",
        source_file=str(filepath),
        sequence_type=seq,
        n_blocks=len(target_ikis),
        block_ids=sorted(target_ikis),
        summary_row=summary_row,
        trials_df=trials,
        parameters=parameters,
        validation=validation_res.__dict__ if validation_res else validation,
        algorithm_doc="algorithms/change_point_pelt.md",
    )
