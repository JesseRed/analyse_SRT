"""Generative model of SRTT key-press timing with a known (latent) chunk structure.

Per block b and IKI position k (1..7), on the log scale:

    log IKI = mu_participant
            + pos_scale   * position_effect[k]           (motor nuisance, empirical)
            + trans_scale * transition_effect[key_k->key_k+1]   (motor nuisance, empirical)
            + seq_speedup[seq]                           (sequence-specific learning, uniform)
            + learn_slope * b/N + session_gain * (s-1) + fatigue * late(b)
            + AR(1) block noise                          (shared by all IKIs of the block)
            + strength * delta          if IKI_k is a between-chunk transition  (spike channel)
            + strength * level * a_c    if IKI_k is within chunk c              (level channel)
            - strength * ramp * r_k     within-chunk acceleration               (ramp channel)
            + strength * u_{b,c}        shared chunk noise, u ~ N(0, chunk_sd)  (correlation channel)
            + predictive_b * surprisal_k (optional, n-gram learner over the stimulus stream)
            + post-error slowing

    IKI = exp(log IKI) * R,   R = (exp(N(0, sd_core)) + Exp(tau_ratio)) / median

The multiplicative ex-Gaussian residual R (median 1) reproduces the right-skewed
within-participant log-IKI distribution of the real data; because R has median 1,
delta stays interpretable as the log ratio of median boundary vs. baseline IKIs.

IKIs are converted to press times (first press ~ log-normal), timestamp jitter is added
(produces the negative lag-1 IKI correlation seen in the real data), lapses add a
heavy right tail, and times are quantised to the 60 Hz frame grid. Errors are
wrong-key presses (isHit=0, the block continues with 8 events) whose probability is
position-specific (empirical) and elevated at chunk starts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import structures as st
from .design import KEYS, NGramLearner, choose_schedule, target_string, yellow_assignment

SEQ_LETTER = {"b": "blue", "g": "green", "y": "yellow"}


@dataclass
class SequenceSpec:
    """Latent chunk structure and its dynamics for one sequence type."""

    structure: str = "8"  # partition name, "8" = no chunk boundaries
    # per-session partition names (concatenation / fragmentation / boundary shift);
    # session s uses structure_schedule[min(s, len)-1]
    structure_schedule: Optional[List[str]] = None
    alt_structure: Optional[str] = None  # trial-level mixture partner
    mixture_p: float = 0.0
    delta: float = 0.0  # spike channel (log units)
    level: float = 0.0  # level channel amplitude (log units)
    ramp: float = 0.0  # within-chunk acceleration (log units, full ramp)
    chunk_sd: float = 0.0  # correlation channel: SD of shared within-chunk noise
    # emergence: logistic in the number of blocks of this sequence within the session
    emergence_midpoint: Optional[float] = None  # None = full strength from the first block
    emergence_slope: float = 5.0
    retention: float = 1.0  # strength carried over to the next session
    offline_gain: float = 0.0  # added overnight (consolidation)
    speedup: float = 0.0  # uniform sequence-specific speed-up (no chunk information)
    error_boundary_mult: float = 1.0  # error probability multiplier at chunk starts


@dataclass
class ParticipantSpec:
    """All concrete generating parameters of one simulated participant."""

    uid: str
    mu: float
    sd_block: float
    ar1_block: float
    sd_core: float  # SD of the log-normal core of the residual
    tau_ratio: float  # exponential component of the residual, relative to the core median
    jitter_sd: float  # seconds, per press timestamp
    lapse_prob: float
    lapse_mean: float  # seconds (exponential)
    pos_scale: float
    trans_scale: float
    learn_slope: float  # log change over one session (b/N from 0 to 1)
    session_gain: float  # log change per additional session
    fatigue: float  # log slowing reached at the end of a session (starts at 50 %)
    error_scale: float
    pes: float  # post-error slowing (log)
    first_press_mu: float
    first_press_sd: float
    predictive_b: float
    n_blocks: int
    drop_block_frac: float
    error_block_shape: float = 0.17  # Gamma shape of per-block error propensity (error clustering)
    group: str = ""
    sequences: Dict[str, SequenceSpec] = field(default_factory=dict)
    level_pattern: Dict[str, List[float]] = field(default_factory=dict)

    def to_flat_dict(self) -> Dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if k not in ("sequences", "level_pattern")}
        for seq, spec in self.sequences.items():
            for k, v in asdict(spec).items():
                out[f"{seq}_{k}"] = json.dumps(v) if isinstance(v, list) else v
        return out


@lru_cache(maxsize=64)
def _exgauss_median(sd_core: float, tau_ratio: float) -> float:
    rng = np.random.default_rng(12345)
    z = np.exp(rng.normal(0.0, sd_core, 200_000)) + rng.exponential(max(tau_ratio, 1e-12), 200_000)
    return float(np.median(z))


def exgauss_residual(rng: np.random.Generator, sd_core: float, tau_ratio: float, size: int) -> np.ndarray:
    """Multiplicative residual with median 1: log-normal core plus exponential tail."""
    z = np.exp(rng.normal(0.0, sd_core, size))
    if tau_ratio > 0:
        z = z + rng.exponential(tau_ratio, size)
    return z / _exgauss_median(round(sd_core, 6), round(tau_ratio, 6))


def _structure_for(spec: SequenceSpec, session: int, rng: np.random.Generator) -> str:
    name = spec.structure
    if spec.structure_schedule:
        name = spec.structure_schedule[min(session, len(spec.structure_schedule)) - 1]
    if spec.alt_structure and spec.mixture_p > 0 and rng.random() < spec.mixture_p:
        name = spec.alt_structure
    return name


def _logistic01(j: float, mid: float, slope: float) -> float:
    f = lambda x: 1.0 / (1.0 + np.exp(-(x - mid) / max(slope, 1e-6)))  # noqa: E731
    lo = f(0.0)
    return float(np.clip((f(j) - lo) / (1.0 - lo), 0.0, 1.0))


def session_start_strengths(spec: SequenceSpec, n_sessions: int, n_seq_blocks: int) -> List[float]:
    """Strength at the start of each session (consolidation / retention model)."""
    if spec.emergence_midpoint is None:
        return [1.0] * n_sessions
    starts = [0.0]
    for _ in range(1, n_sessions):
        s0 = starts[-1]
        end = s0 + (1.0 - s0) * _logistic01(n_seq_blocks, spec.emergence_midpoint, spec.emergence_slope)
        starts.append(float(min(1.0, spec.retention * end + spec.offline_gain)))
    return starts


def _strength(spec: SequenceSpec, s0: float, j: int) -> float:
    if spec.emergence_midpoint is None:
        return 1.0
    return s0 + (1.0 - s0) * _logistic01(j, spec.emergence_midpoint, spec.emergence_slope)


def draw_level_pattern(rng: np.random.Generator, n: int = 8) -> List[float]:
    """Alternating-sign chunk level offsets (fixed per participant and sequence)."""
    return [float((1.0 if i % 2 == 0 else -1.0) * rng.uniform(0.5, 1.0)) for i in range(n)]


@dataclass
class SessionResult:
    events: pd.DataFrame
    truth: pd.DataFrame


def simulate_session(
    p: ParticipantSpec,
    session: int,
    n_sessions: int,
    calib: Dict[str, Any],
    rng: np.random.Generator,
) -> SessionResult:
    frame = float(calib.get("frame_seconds", 1.0 / 60.0))
    pos_eff = np.asarray(calib["position_effects"], dtype=float)
    trans_eff = calib["transition_effects"]
    err_pos = {s: np.asarray(calib["errors"][s]["per_position"], dtype=float) for s in ("blue", "green", "yellow")}

    schedule = choose_schedule(p.n_blocks, rng, calib.get("schedules"))
    seq_types = [SEQ_LETTER[c] for c in schedule]
    yellow_seqs = iter(yellow_assignment(seq_types.count("yellow"), rng))
    n_blocks = len(seq_types)
    n_seq_blocks = {s: seq_types.count(s) for s in ("blue", "green", "yellow")}
    s0 = {
        s: session_start_strengths(p.sequences[s], n_sessions, n_seq_blocks[s])[session - 1]
        for s in p.sequences
    }

    keep = np.ones(n_blocks, dtype=bool)
    if p.drop_block_frac > 0:
        n_drop = int(round(p.drop_block_frac * n_blocks))
        keep[rng.choice(n_blocks, n_drop, replace=False)] = False

    learner = NGramLearner(order=2, alpha=1.0)
    seen = {"blue": 0, "green": 0, "yellow": 0}
    block_noise = 0.0
    rows: List[Tuple] = []
    truth_rows: List[Dict[str, Any]] = []
    event_number = 0

    for bi in range(n_blocks):
        block_number = bi + 1
        seq = seq_types[bi]
        targets_str = target_string(seq, next(yellow_seqs) if seq == "yellow" else None)
        targets = [int(c) for c in targets_str]
        spec = p.sequences.get(seq, SequenceSpec())

        # surprisal of each upcoming item given the within-block context (learner sees all blocks)
        surprisal = np.zeros(7)
        for i, key in enumerate(targets):
            ctx = tuple(targets[max(0, i - 2): i])
            if i > 0:
                surprisal[i - 1] = learner.surprisal(ctx, key)
            learner.update(ctx, key)

        block_noise = p.ar1_block * block_noise + np.sqrt(1.0 - p.ar1_block ** 2) * p.sd_block * rng.standard_normal()
        if not keep[bi]:
            continue

        j = seen[seq]
        seen[seq] += 1
        strength = _strength(spec, s0.get(seq, 1.0), j)
        structure = _structure_for(spec, session, rng)
        lengths = st.parse_structure(structure)
        boundaries = st.lengths_to_boundaries(lengths)
        bmask = np.asarray(st.boundary_iki_mask(boundaries))
        chunk_idx = np.asarray(st.chunk_of_iki(lengths))
        rel_pos = np.asarray(st.position_in_chunk(lengths))

        frac = bi / max(n_blocks - 1, 1)
        log_iki = np.full(7, p.mu, dtype=float)
        log_iki += p.pos_scale * pos_eff
        log_iki += p.trans_scale * np.array([trans_eff.get(f"{targets[k]}{targets[k + 1]}", 0.0) for k in range(7)])
        log_iki += spec.speedup + p.learn_slope * frac + p.session_gain * (session - 1)
        log_iki += p.fatigue * max(0.0, 2.0 * (frac - 0.5))
        log_iki += block_noise

        within = ~bmask
        log_iki[bmask] += strength * spec.delta
        if spec.level:
            pattern = np.asarray(p.level_pattern.get(seq, draw_level_pattern(rng)))
            log_iki[within] += strength * spec.level * pattern[chunk_idx[within]]
        if spec.ramp:
            log_iki[within] -= strength * spec.ramp * (rel_pos[within] - 0.5)
        if spec.chunk_sd:
            shared = rng.normal(0.0, spec.chunk_sd, size=len(lengths))
            log_iki[within] += strength * shared[chunk_idx[within]]
        if p.predictive_b:
            log_iki += p.predictive_b * (surprisal - surprisal.mean())

        # errors: wrong key presses (block continues), more likely at chunk starts
        # per-block error propensity (mean 1): errors cluster in few blocks as in the real data
        propensity = rng.gamma(p.error_block_shape, 1.0 / p.error_block_shape) if p.error_block_shape > 0 else 1.0
        press_err_p = err_pos[seq] * p.error_scale * propensity
        chunk_start = np.zeros(8, dtype=bool)
        chunk_start[[b for b in boundaries]] = True  # press index b (0-based) starts a new chunk
        press_err_p = np.where(chunk_start, press_err_p * spec.error_boundary_mult, press_err_p)
        is_error = rng.random(8) < np.clip(press_err_p, 0.0, 0.9)
        pressed = list(targets)
        for i in np.where(is_error)[0]:
            pressed[i] = int(rng.choice([k for k in KEYS if k != targets[i]]))
        log_iki += p.pes * is_error[:7]  # IKI following an erroneous press

        ikis = np.exp(log_iki) * exgauss_residual(rng, p.sd_core, p.tau_ratio, 7)
        lapses = rng.random(7) < p.lapse_prob
        ikis = ikis + lapses * rng.exponential(p.lapse_mean, size=7)

        first = float(np.exp(rng.normal(p.first_press_mu, p.first_press_sd)))
        times = first + np.concatenate([[0.0], np.cumsum(ikis)])
        times = times + rng.normal(0.0, p.jitter_sd, size=8)
        phase = rng.uniform(0.0, frame)
        times = np.round((times - phase) / frame) * frame + phase
        for i in range(1, 8):
            if times[i] <= times[i - 1]:
                times[i] = times[i - 1] + frame
        times = np.maximum(times, frame)

        for i in range(8):
            event_number += 1
            rows.append((block_number, event_number, float(times[i]), int(not is_error[i]),
                         targets[i], pressed[i], seq))
        truth_rows.append(
            {
                "block_number": block_number,
                "sequence_type": seq,
                "target": targets_str,
                "structure": structure,
                "true_boundaries": json.dumps(boundaries),
                "true_cp_segments": json.dumps(st.boundaries_to_segment_cps(boundaries)),
                "true_n_chunks": len(lengths),
                "strength": round(float(strength), 4),
                "effective_delta": round(float(strength * spec.delta), 4),
                "effective_level": round(float(strength * spec.level), 4),
                "effective_chunk_sd": round(float(strength * spec.chunk_sd), 4),
                "n_errors": int(is_error.sum()),
                "error_positions": json.dumps([int(i + 1) for i in np.where(is_error)[0]]),
                "n_lapses": int(lapses.sum()),
                "surprisal_profile": json.dumps([round(float(x), 3) for x in surprisal]),
            }
        )

    events = pd.DataFrame(
        rows,
        columns=["BlockNumber", "EventNumber", "Time Since Block start", "isHit", "target", "pressed", "sequence"],
    )
    return SessionResult(events=events, truth=pd.DataFrame(truth_rows))


def window_modal_boundaries(truth: pd.DataFrame, window_size: int = 30, step: int = 10) -> pd.Series:
    """Most frequent true partition per sliding window (same windows as PELT/rational).

    Each block is mapped to the window whose centre is nearest (ties -> earlier window),
    as the window-based methods do; the modal partition is taken over the blocks of the
    same sequence type inside that window.
    """
    out = pd.Series(index=truth.index, dtype=object)
    if truth.empty:
        return out
    lo, hi = int(truth["block_number"].min()), int(truth["block_number"].max())
    centers = []
    start = lo
    while start + window_size - 1 <= hi:
        centers.append((start, start + window_size - 1, start + (window_size - 1) // 2))
        start += step
    if not centers:
        centers = [(lo, hi, (lo + hi) // 2)]
    for seq, grp in truth.groupby("sequence_type"):
        for idx, row in grp.iterrows():
            b = int(row["block_number"])
            w = min(centers, key=lambda c: (abs(c[2] - b), c[2]))
            members = grp[(grp["block_number"] >= w[0]) & (grp["block_number"] <= w[1])]
            out[idx] = members["true_boundaries"].mode().iloc[0] if len(members) else row["true_boundaries"]
    return out
