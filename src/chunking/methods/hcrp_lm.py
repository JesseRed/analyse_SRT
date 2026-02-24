"""HCRP-LM chunking: surprisal-based chunk detection via a hierarchical CRP language model.

Vendored HCRP_LM class adapted from:
  https://github.com/noemielteto/HCRP_sequence_learning/blob/main/ddHCRP_LM.py
  Eltető et al. (2022) – MIT-compatible research code (no license file in repo).

Method: for each participant file × sequence_type, parse the sequence of stimulus keys
online through the HCRP model. Surprisal spikes (−log P(w_t | context)) indicate
positions where learned chunks start → mapped to chunk_boundaries (IKI positions 1–7).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from .._base import ChunkingResult
from .._data import load_srt_file

# ---------------------------------------------------------------------------
# Vendored HCRP_LM class (adapted from ddHCRP_LM.py)
# ---------------------------------------------------------------------------

_E = math.e


class _HCRP_LM:
    """Hierarchical Chinese Restaurant Process Language Model.

    Based on Teh (2006) / Eltető et al. (2022).  Optionally distance-dependent
    (exponential forgetting) when *decay_constant* is supplied.

    Parameters
    ----------
    strength : list[float]
        α parameter at each hierarchy level.  Length defines the maximum context
        depth (n_levels = len(strength); max context = n_levels − 1 elements).
    decay_constant : list[float] | None
        λ (decay rate) per level.  None / all-None → plain (non-forgetful) CRP.
    n_samples : int
        Independent seating-arrangement samples; predictions are averaged.
    dishes : list | None
        Known observation types (learned from data if None).
    """

    def __init__(
        self,
        strength: List[float],
        decay_constant: Optional[List[float]] = None,
        n_samples: int = 5,
        dishes: Optional[list] = None,
    ) -> None:
        self.strength = strength
        self.decay_constant = decay_constant  # None → non-dd
        self.n = len(strength)
        self.dishes: list = list(dishes) if dishes else []
        self.number_of_dishes = len(self.dishes)
        self.n_samples = n_samples
        self.samples: Dict[int, dict] = {s: {} for s in range(n_samples)}
        self._MAX_TIMESTAMPS = 1000  # ring-buffer size per dish per restaurant

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_restaurant(self, sample: int, str_u: str) -> None:
        """Create a restaurant slot if it doesn't exist yet."""
        if str_u not in self.samples[sample]:
            if self.decay_constant is not None:
                self.samples[sample][str_u] = np.full(
                    (self.number_of_dishes, self._MAX_TIMESTAMPS), np.nan
                )
            else:
                self.samples[sample][str_u] = np.zeros(self.number_of_dishes)

    def _grow_restaurant(self, sample: int, str_u: str) -> None:
        """Expand restaurant arrays when a new dish is discovered."""
        if self.decay_constant is not None:
            arr = self.samples[sample][str_u]
            while arr.shape[0] < self.number_of_dishes:
                arr = np.vstack((arr, np.full(self._MAX_TIMESTAMPS, np.nan)))
            self.samples[sample][str_u] = arr
        else:
            arr = self.samples[sample][str_u]
            while len(arr) < self.number_of_dishes:
                arr = np.append(arr, 0.0)
            self.samples[sample][str_u] = arr

    def _d_u(self, sample: int, str_u: str, t: int, level: int) -> float:
        """Total occupancy / affinity for restaurant u."""
        arr = self.samples[sample][str_u]
        if self.decay_constant is not None:
            ts = arr[~np.isnan(arr)].ravel()
            if len(ts) == 0:
                return 0.0
            return float(np.sum(_E ** (-(t - ts) / self.decay_constant[level])))
        else:
            return float(arr.sum())

    def _d_u_w(self, sample: int, str_u: str, w_i: int, t: int, level: int) -> float:
        """Occupancy / affinity for dish w in restaurant u."""
        arr = self.samples[sample][str_u]
        if self.decay_constant is not None:
            ts = arr[w_i][~np.isnan(arr[w_i])]
            if len(ts) == 0:
                return 0.0
            return float(np.sum(_E ** (-(t - ts) / self.decay_constant[level])))
        else:
            return float(arr[w_i])

    # ------------------------------------------------------------------
    # Core: word_probability (generative / recognition)
    # ------------------------------------------------------------------

    def word_probability(
        self, t: int, u: list, w, sample: int, n: Optional[int] = None
    ) -> float:
        """P(w | context u), for one sample, via back-off recursion."""
        if w not in self.dishes:
            self.dishes.append(w)
            self.number_of_dishes += 1

        w_i = self.dishes.index(w)

        if n is None:
            u = list(u)[-(self.n - 1):]  # truncate to max context
            n = len(u) + 1

        if n == 0:
            return 1.0 / max(self.number_of_dishes, 1)

        str_u = str(u)
        level = len(u)

        if str_u not in self.samples[sample]:
            d_u_val, d_u_w_val = 0.0, 0.0
            self._ensure_restaurant(sample, str_u)
        elif (self.decay_constant is None and len(self.samples[sample][str_u]) <= w_i) or (
            self.decay_constant is not None
            and self.samples[sample][str_u].shape[0] <= w_i
        ):
            self._grow_restaurant(sample, str_u)
            d_u_val = self._d_u(sample, str_u, t, level)
            d_u_w_val = 0.0
        else:
            d_u_val = self._d_u(sample, str_u, t, level)
            d_u_w_val = self._d_u_w(sample, str_u, w_i, t, level)

        alpha = self.strength[level]
        denom = d_u_val + alpha
        if denom == 0.0:
            denom = 1e-12
        prob_seat = d_u_w_val / denom
        prob_backoff = (alpha / denom) * self.word_probability(t, u[1:], w, sample, n - 1)
        return prob_seat + prob_backoff

    def word_probability_all_samples(self, t: int, u: list, w) -> float:
        """Average P(w | context u) across independent samples."""
        return sum(self.word_probability(t, u, w, s) for s in range(self.n_samples)) / self.n_samples

    def get_predictive_distribution(self, t: int, u: list) -> np.ndarray:
        """P(w | context u) for all known dishes; returns probability array."""
        return np.array([self.word_probability_all_samples(t, u, w) for w in self.dishes])

    # ------------------------------------------------------------------
    # Core: add_customer (recognition / update)
    # ------------------------------------------------------------------

    def add_customer(self, t: int, u: list, w, sample: int, n: Optional[int] = None) -> None:
        """Update model with observation w in context u."""
        if w not in self.dishes:
            self.dishes.append(w)
            self.number_of_dishes += 1

        w_i = self.dishes.index(w)

        if n is None:
            u = list(u)[-(self.n - 1):]
            n = len(u) + 1

        if n == 0:
            return

        str_u = str(u)
        level = len(u)

        if str_u not in self.samples[sample]:
            self._ensure_restaurant(sample, str_u)
            if self.decay_constant is not None:
                self.samples[sample][str_u][w_i][
                    np.where(np.isnan(self.samples[sample][str_u][w_i]))[0][0]
                ] = t
            else:
                self.samples[sample][str_u][w_i] += 1.0
            self.add_customer(t, u[1:], w, sample, n - 1)
            return

        if (self.decay_constant is None and len(self.samples[sample][str_u]) <= w_i) or (
            self.decay_constant is not None
            and self.samples[sample][str_u].shape[0] <= w_i
        ):
            self._grow_restaurant(sample, str_u)

        d_u_val = self._d_u(sample, str_u, t, level)
        d_u_w_val = self._d_u_w(sample, str_u, w_i, t, level)
        alpha = self.strength[level]
        denom = d_u_val + alpha
        if denom == 0.0:
            denom = 1e-12

        p_seat = d_u_w_val / denom
        if np.random.random() < p_seat:
            # sit at existing table → add timestamp / count for w in u
            if self.decay_constant is not None:
                nan_idx = np.where(np.isnan(self.samples[sample][str_u][w_i]))[0]
                if len(nan_idx) > 0:
                    self.samples[sample][str_u][w_i][nan_idx[0]] = t
                else:
                    # ring buffer full: overwrite oldest
                    self.samples[sample][str_u][w_i][0] = t
            else:
                self.samples[sample][str_u][w_i] += 1.0
        else:
            # back off to shallower context
            if self.decay_constant is not None:
                nan_idx = np.where(np.isnan(self.samples[sample][str_u][w_i]))[0]
                if len(nan_idx) > 0:
                    self.samples[sample][str_u][w_i][nan_idx[0]] = t
                else:
                    self.samples[sample][str_u][w_i][0] = t
            else:
                self.samples[sample][str_u][w_i] += 1.0
            self.add_customer(t, u[1:], w, sample, n - 1)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

_SEQUENCE_COL_MAP = {
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
}


def _extract_target_sequences(
    df: pd.DataFrame,
    sequence_type: str,
) -> Dict[int, List[int]]:
    """Extract per-block target-key sequences (only full 8-hit blocks).

    Returns
    -------
    dict[block_number -> list of 8 integer target keys]
    """
    seq = sequence_type.lower().strip()

    # Filter for the requested sequence type (normalized in load_srt_file)
    if "sequence" in df.columns:
        df = df[df["sequence"] == seq]

    # Keep only hits
    if "isHit" in df.columns:
        df = df[df["isHit"] == 1]

    block_col = "BlockNumber"
    target_col = "target"

    if block_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Required columns '{block_col}' and '{target_col}' not found in data. "
            f"Available: {df.columns.tolist()}"
        )

    sequences: Dict[int, List[int]] = {}
    for block_id, grp in df.groupby(block_col):
        targets = grp[target_col].tolist()
        if len(targets) == 8:
            sequences[int(block_id)] = [int(x) for x in targets]

    return sequences


# ---------------------------------------------------------------------------
# HCRP parse & surprisal computation
# ---------------------------------------------------------------------------

def _build_model(
    strength: List[float],
    decay_constant: Optional[List[float]],
    n_samples: int,
    dishes: Optional[list],
    random_state: Optional[int],
) -> _HCRP_LM:
    if random_state is not None:
        np.random.seed(random_state)
    return _HCRP_LM(
        strength=strength,
        decay_constant=decay_constant,
        n_samples=n_samples,
        dishes=dishes,
    )


def _parse_and_surprisal(
    sequences: Dict[int, List[int]],
    model: _HCRP_LM,
) -> Dict[int, List[float]]:
    """Parse all blocks online through the model; return per-block surprisal profiles.

    For each block, surprisal at position p (0-indexed within block) is:
        s_p = -log2(P(target_p | context of previous targets *in this block*))

    Only positions 1..7 (0-indexed: index 1 to 7) are returned (length 7),
    because position 0 has no within-block context and maps to IKI index 1..7.

    Global trial counter `t` increments across all blocks for the distance-dep. CRP.
    """
    block_ids = sorted(sequences.keys())
    surprisal_by_block: Dict[int, List[float]] = {}

    t = 0  # global trial counter (for distance-dependent CRP)
    for block_id in block_ids:
        targets = sequences[block_id]
        block_surprisals: List[float] = []

        for pos, w in enumerate(targets):
            # Within-block context (up to max context depth)
            u = targets[max(0, pos - (model.n - 1)) : pos]

            # Evaluate P(w | u) BEFORE adding
            if pos == 0:
                # No within-block context for the first item – use unigram
                prob = model.word_probability_all_samples(t, [], w)
            else:
                prob = model.word_probability_all_samples(t, u, w)

            surprisal = -math.log2(max(prob, 1e-12))
            block_surprisals.append(surprisal)

            # Update model with this observation (all samples)
            for s in range(model.n_samples):
                model.add_customer(t, u, w, s)

            t += 1

        # Return positions 1..7 (IKI-index aligned: after position 0 within block)
        surprisal_by_block[block_id] = block_surprisals[1:]  # length 7

    return surprisal_by_block


# ---------------------------------------------------------------------------
# Chunk boundary detection
# ---------------------------------------------------------------------------

def _detect_boundaries(
    surprisal_profile: List[float],
    threshold_z: float,
) -> List[int]:
    """Z-score threshold → chunk boundary positions (1-indexed, range 1–7).

    Position i corresponds to the transition *before* IKI i (i.e., between
    stimulus i and i+1, matching the convention used by all other methods).
    """
    arr = np.asarray(surprisal_profile, dtype=float)
    if arr.size == 0:
        return []
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-10:
        return []
    z = (arr - mu) / sigma
    # boundary at position i (1-indexed) if z-score at that position exceeds threshold
    return [int(i + 1) for i in range(len(z)) if z[i] > threshold_z]


# ---------------------------------------------------------------------------
# Public run_analysis
# ---------------------------------------------------------------------------

def run_analysis(
    filepath: str | Path,
    sequence_type: str = "blue",
    *,
    n_levels: int = 3,
    strength: float | List[float] = 0.5,
    decay_constant: float | List[float] | None = 50.0,
    n_samples: int = 5,
    threshold_z: float = 1.0,
    random_state: Optional[int] = None,
) -> ChunkingResult:
    """Run HCRP-LM surprisal-based chunking on one participant file.

    Parameters
    ----------
    filepath : path to SRT CSV
    sequence_type : ``"blue"``, ``"green"``, or ``"yellow"``
    n_levels : int
        Hierarchy depth; max context = n_levels − 1 previous stimuli.
    strength : float or list[float]
        α strength parameter(s).  A scalar is broadcast to length n_levels.
    decay_constant : float or list[float] or None
        λ forgetting rate(s).  None → non-distance-dependent (static counts).
        A scalar is broadcast to length n_levels.
        Set to 0 to use plain CRP (no forgetting).
    n_samples : int
        Number of independent HCRP samples for MC averaging.
    threshold_z : float
        Z-score threshold above which surprisal signals a chunk boundary.
    random_state : int or None
        Random seed.
    """
    filepath = Path(filepath)
    df = load_srt_file(filepath)

    # --- build parameter lists ------------------------------------------------
    if isinstance(strength, (int, float)):
        strength_list = [float(strength)] * n_levels
    else:
        strength_list = [float(s) for s in strength]
        if len(strength_list) != n_levels:
            raise ValueError(f"len(strength)={len(strength_list)} must equal n_levels={n_levels}")

    if decay_constant is None:
        decay_list = None
    elif isinstance(decay_constant, (int, float)):
        if float(decay_constant) == 0.0:
            decay_list = None
        else:
            decay_list = [float(decay_constant)] * n_levels
    else:
        dc = [float(x) for x in decay_constant]
        if len(dc) != n_levels:
            raise ValueError(f"len(decay_constant)={len(dc)} must equal n_levels={n_levels}")
        decay_list = dc if any(v > 0 for v in dc) else None

    # --- load and extract sequences -------------------------------------------
    sequences = _extract_target_sequences(df, sequence_type)
    if not sequences:
        raise ValueError(f"No valid 8-hit blocks found for sequence='{sequence_type}'.")

    block_ids = sorted(sequences.keys())

    # --- build HCRP and parse -------------------------------------------------
    model = _build_model(
        strength=strength_list,
        decay_constant=decay_list,
        n_samples=n_samples,
        dishes=None,
        random_state=random_state,
    )
    surprisal_by_block = _parse_and_surprisal(sequences, model)

    # --- chunk boundary detection per block -----------------------------------
    trials_rows = []
    all_surprisals: List[float] = []
    boundary_surprisals: List[float] = []

    for block_id in block_ids:
        profile = surprisal_by_block[block_id]
        boundaries = _detect_boundaries(profile, threshold_z)
        n_chunks = len(boundaries) + 1

        all_surprisals.extend(profile)
        for i, s in enumerate(profile):
            if (i + 1) in boundaries:  # boundaries are 1-indexed
                boundary_surprisals.append(s)

        trials_rows.append(
            {
                "block_number": block_id,
                "n_chunks": n_chunks,
                "chunk_boundaries": boundaries,
                "surprisal_profile": profile,
            }
        )

    trials_df = pd.DataFrame(trials_rows).sort_values("block_number").reset_index(drop=True)
    trials_df["source_file"] = str(filepath)
    trials_df["sequence_type"] = sequence_type
    trials_df["method"] = "hcrp_lm"

    mean_n_chunks = float(trials_df["n_chunks"].mean())
    mean_surprisal = float(np.mean(all_surprisals)) if all_surprisals else float("nan")
    mean_boundary_surprisal = (
        float(np.mean(boundary_surprisals)) if boundary_surprisals else float("nan")
    )

    summary_row: Dict[str, Any] = {
        "source_file": str(filepath),
        "sequence_type": sequence_type,
        "method": "hcrp_lm",
        "n_blocks": len(block_ids),
        "mean_n_chunks": mean_n_chunks,
        "mean_surprisal": mean_surprisal,
        "mean_boundary_surprisal": mean_boundary_surprisal,
    }

    parameters: Dict[str, Any] = {
        "n_levels": n_levels,
        "strength": strength_list,
        "decay_constant": decay_list,
        "n_samples": n_samples,
        "threshold_z": threshold_z,
        "random_state": random_state,
    }

    return ChunkingResult(
        method_name="hcrp_lm",
        source_file=str(filepath),
        sequence_type=sequence_type,
        n_blocks=len(block_ids),
        block_ids=block_ids,
        summary_row=summary_row,
        trials_df=trials_df,
        parameters=parameters,
        validation=None,
        algorithm_doc="algorithms/hcrp_lm.md",
    )
