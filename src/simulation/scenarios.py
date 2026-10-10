"""Scenario families (data-generating mechanisms) of the simulation study.

Each family consists of cells (one combination of factor levels). A cell knows how to
draw a concrete ``ParticipantSpec`` from a random generator; the tier configuration
(configs/simulation/*.yaml) decides how many participants, sessions and replicates are
generated per cell. See simulations/README.md for the ADEMP protocol.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .generator import ParticipantSpec, SequenceSpec, draw_level_pattern
from .structures import STRUCTURE_SETS

# Noise levels as multiples of the fitted real-data noise ("mid" = real data, see
# noisefit.py and simulations/core/realism_report.md). Scales sd_core, tau_ratio, sd_block.
NOISE_LEVELS = {"low": 0.6, "mid": 1.0, "high": 1.5}
FALLBACK_NOISE = {"sd_core": 0.07, "tau_ratio": 0.45, "jitter_sd": 0.03, "sd_block": 0.11, "ar1_block": 0.6}


def fitted_noise(calib: Dict[str, Any]) -> Dict[str, float]:
    return dict(calib.get("noise_fit", {}).get("params", FALLBACK_NOISE))
EFFECT_SIZES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
CHANNELS = ["spike", "level", "corr"]
BOUNDARY_ERROR_MULT = 1.75


@dataclass
class Cell:
    family: str  # directory name, e.g. "B_static"
    index: int
    code: str  # filename-safe cell code (letters, digits, '-')
    factors: Dict[str, Any]
    make: Callable[[np.random.Generator, str], ParticipantSpec] = field(repr=False)


# ----------------------------------------------------------------------------------
# base participant
# ----------------------------------------------------------------------------------


def base_participant(
    rng: np.random.Generator,
    uid: str,
    calib: Dict[str, Any],
    *,
    heterogeneity: float = 0.5,
    noise: str = "mid",
    motor: bool = True,
    trends: bool = True,
    seq_speedup: bool = True,
    n_blocks: int = 160,
) -> ParticipantSpec:
    """Participant with realistic nuisance structure but no chunking (all sequences '8')."""
    nf = fitted_noise(calib)
    scale = NOISE_LEVELS[noise]
    offsets = calib.get("sequence_speedup_net", calib.get("sequence_offsets", {}))
    seqs = {
        s: SequenceSpec(speedup=float(offsets.get(s, 0.0)) if seq_speedup else 0.0)
        for s in ("blue", "green", "yellow")
    }
    learn = calib.get("learning_slope_per_session", {}).get("blue", -0.08)
    # The calibrated median already contains the average sequence speed-up and the
    # mean within-session learning; remove them so the simulated file median matches.
    mu0 = float(calib["mu_log_iki"])
    if seq_speedup:
        mu0 -= 0.5 * offsets.get("blue", 0.0) + 0.25 * offsets.get("green", 0.0)
    if trends:
        mu0 -= 0.5 * learn
    return ParticipantSpec(
        uid=uid,
        mu=float(mu0 + rng.normal(0.0, calib["sd_between_files"] * heterogeneity)),
        sd_block=float(nf["sd_block"] * scale * rng.uniform(0.8, 1.2)),
        ar1_block=float(nf["ar1_block"]),
        sd_core=float(nf["sd_core"] * scale * rng.uniform(0.85, 1.15)),
        tau_ratio=float(nf["tau_ratio"] * scale * rng.uniform(0.85, 1.15)),
        jitter_sd=float(nf["jitter_sd"]),
        lapse_prob=0.015,
        lapse_mean=0.6,
        pos_scale=1.0 if motor else 0.0,
        trans_scale=1.0 if motor else 0.0,
        learn_slope=float(learn) if trends else 0.0,
        session_gain=-0.04 if trends else 0.0,
        fatigue=0.0,
        error_scale=float(rng.uniform(0.7, 1.3)),
        pes=float(calib.get("post_error_slowing_log", 0.59)),
        first_press_mu=float(calib["first_press_log_mu"]),
        first_press_sd=float(calib["first_press_log_sd"]),
        predictive_b=0.0,
        n_blocks=n_blocks,
        drop_block_frac=0.0,
        sequences=seqs,
        level_pattern={s: draw_level_pattern(rng) for s in ("blue", "green", "yellow")},
    )


def _apply_chunking(
    p: ParticipantSpec,
    seq: str,
    structure: str,
    effect: float,
    channel: str = "spike",
    **extra: Any,
) -> None:
    spec = p.sequences[seq]
    spec.structure = structure
    spec.error_boundary_mult = BOUNDARY_ERROR_MULT if effect > 0 else 1.0
    if channel == "spike":
        spec.delta = effect
    elif channel == "level":
        spec.level = effect
    elif channel == "corr":
        spec.chunk_sd = effect
    elif channel == "mixed":  # spike + some within-chunk correlation and acceleration
        spec.delta = effect
        spec.chunk_sd = 0.5 * effect
        spec.ramp = 0.5 * effect
    else:
        raise ValueError(f"unknown channel {channel}")
    for k, v in extra.items():
        setattr(spec, k, v)


def _pick(rng: np.random.Generator, options: List[str]) -> str:
    return str(options[int(rng.integers(len(options)))])


# ----------------------------------------------------------------------------------
# families
# ----------------------------------------------------------------------------------


def family_a_null(calib: Dict[str, Any]) -> List[Cell]:
    """A: no chunk structure anywhere -> false-positive rate and calibration."""
    variants = {
        "A1-pure": dict(motor=False, trends=False, seq_speedup=False),
        "A2-motor": dict(motor=True, trends=False, seq_speedup=False),
        "A3-trends": dict(motor=True, trends=True, seq_speedup=False, fatigue=0.10),
        "A4-hetero": dict(motor=True, trends=True, seq_speedup=False, heterogeneity=1.0,
                          noise="high", lapse_prob=0.04),
        "A5-seqspeedup": dict(motor=True, trends=True, seq_speedup=True),
    }
    cells = []
    for i, (code, kw) in enumerate(variants.items()):
        kw = dict(kw)
        fatigue = kw.pop("fatigue", 0.0)
        lapse = kw.pop("lapse_prob", None)

        def make(rng, uid, kw=kw, fatigue=fatigue, lapse=lapse):
            p = base_participant(rng, uid, calib, **kw)
            p.fatigue = fatigue
            if lapse is not None:
                p.lapse_prob = lapse
            return p

        cells.append(Cell("A_null", i, code, {"variant": code, "effect": 0.0, **kw}, make))
    return cells


def family_b_static(calib: Dict[str, Any]) -> List[Cell]:
    """B: static chunk structures, effect size x noise x signal channel."""
    cells = []
    grid = itertools.product(EFFECT_SIZES, NOISE_LEVELS, CHANNELS)
    for i, (effect, noise, channel) in enumerate(grid):
        code = f"d{int(round(effect * 100)):03d}-n{noise}-{channel}"

        def make(rng, uid, effect=effect, noise=noise, channel=channel):
            p = base_participant(rng, uid, calib, noise=noise)
            for seq in ("blue", "green"):
                _apply_chunking(p, seq, _pick(rng, STRUCTURE_SETS["static_pool"]), effect, channel)
            return p

        cells.append(Cell("B_static", i, code,
                          {"effect": effect, "noise": noise, "channel": channel}, make))
    return cells


def family_c_dynamics(calib: Dict[str, Any]) -> List[Cell]:
    """C: chunk structures that change within or across sessions (blue and green alike)."""
    effect = 0.30
    variants: Dict[str, Dict[str, Any]] = {
        # strength rises within session 1 and is retained
        "C1-emergence": dict(structure="4-4", emergence_midpoint=20.0, emergence_slope=6.0),
        # 2-2-2-2 -> 4-4 -> 8 (concatenation, Wymbs/Grafton)
        "C2-concatenation": dict(structure="2-2-2-2",
                                 structure_schedule=["2-2-2-2", "2-2-2-2", "4-4", "4-4", "8"]),
        # 8 -> 4-4 -> 2-2-2-2
        "C3-fragmentation": dict(structure="8",
                                 structure_schedule=["8", "4-4", "4-4", "2-2-2-2", "2-2-2-2"]),
        # one boundary migrates 3 -> 4 -> 5
        "C4-shift": dict(structure="3-5", structure_schedule=["3-5", "3-5", "4-4", "5-3", "5-3"]),
        # trial-level mixture of two partitions
        "C5-mixture20": dict(structure="4-4", alt_structure="2-2-2-2", mixture_p=0.2),
        "C6-mixture50": dict(structure="4-4", alt_structure="2-2-2-2", mixture_p=0.5),
        # within-session emergence, partial overnight loss + offline gain
        "C7-consolidation": dict(structure="3-3-2", emergence_midpoint=15.0, emergence_slope=5.0,
                                 retention=0.6, offline_gain=0.15),
    }
    cells = []
    for i, (code, kw) in enumerate(variants.items()):
        def make(rng, uid, kw=kw):
            p = base_participant(rng, uid, calib)
            for seq in ("blue", "green"):
                _apply_chunking(p, seq, kw["structure"], effect, "mixed",
                                **{k: v for k, v in kw.items() if k != "structure"})
            return p

        cells.append(Cell("C_dynamics", i, code, {"variant": code, "effect": effect, **kw}, make))

    # predictive timing without discrete chunks (graded surprisal effect)
    def make_pred(rng, uid):
        p = base_participant(rng, uid, calib)
        p.predictive_b = 0.08
        return p

    cells.append(Cell("C_dynamics", len(cells), "C8-predictive",
                      {"variant": "C8-predictive", "effect": 0.0, "predictive_b": 0.08}, make_pred))
    return cells


def family_d_groups(calib: Dict[str, Any]) -> List[Cell]:
    """D: two populations with different chunk sizes (young: 4-5 items, old: 2-3 items)."""
    cells = []
    i = 0
    for effect in (0.15, 0.30):
        for group in ("young", "old"):
            code = f"d{int(round(effect * 100)):03d}-{group}"

            def make(rng, uid, effect=effect, group=group):
                p = base_participant(rng, uid, calib, noise="mid")
                p.group = group
                if group == "old":
                    p.mu += 0.35
                    p.error_scale *= 1.5
                    p.sd_core *= 1.15
                    p.tau_ratio *= 1.15
                for seq in ("blue", "green"):
                    _apply_chunking(p, seq, _pick(rng, STRUCTURE_SETS[group]), effect, "mixed")
                return p

            cells.append(Cell("D_groups", i, code, {"effect": effect, "group": group}, make))
            i += 1
    return cells


def family_e_realistic(calib: Dict[str, Any]) -> List[Cell]:
    """E: end-to-end benchmark resembling the real study (heterogeneous everything)."""

    def make(rng, uid):
        p = base_participant(rng, uid, calib, heterogeneity=1.0)
        p.predictive_b = float(rng.uniform(0.0, 0.05))
        p.fatigue = float(rng.uniform(0.0, 0.08))
        dynamics = _pick(rng, ["static", "static", "emergence", "concatenation", "mixture"])
        base_effect = float(max(0.0, rng.normal(0.22, 0.10)))
        for seq, scale, mid in (("blue", 1.0, 15.0), ("green", 0.6, 12.0)):
            structure = _pick(rng, STRUCTURE_SETS["realistic"])
            extra: Dict[str, Any] = {}
            if dynamics == "emergence":
                extra = dict(emergence_midpoint=mid, emergence_slope=5.0, retention=0.85, offline_gain=0.1)
            elif dynamics == "concatenation":
                extra = dict(structure_schedule=[structure, structure, "4-4", "4-4", "8"])
            elif dynamics == "mixture":
                extra = dict(alt_structure=_pick(rng, STRUCTURE_SETS["realistic"]), mixture_p=0.3)
            _apply_chunking(p, seq, structure, base_effect * scale, "mixed", **extra)
        p.group = dynamics
        return p

    return [Cell("E_realistic", 0, "E1-realistic", {"variant": "E1-realistic"}, make)]


def family_f_quality(calib: Dict[str, Any]) -> List[Cell]:
    """F: robustness against data-quality problems at a fixed, moderate effect."""
    effect = 0.30
    variants = {
        "F1-120blocks": dict(n_blocks=120),
        "F2-higherrors": dict(error_scale=3.0),
        "F3-lapses": dict(lapse_prob=0.05, lapse_mean=1.0),
        "F4-missingblocks": dict(drop_block_frac=0.15),
    }
    cells = []
    for i, (code, kw) in enumerate(variants.items()):
        def make(rng, uid, kw=kw):
            p = base_participant(rng, uid, calib, n_blocks=kw.get("n_blocks", 160))
            if "error_scale" in kw:
                p.error_scale *= kw["error_scale"]
            if "lapse_prob" in kw:
                p.lapse_prob, p.lapse_mean = kw["lapse_prob"], kw["lapse_mean"]
            p.drop_block_frac = kw.get("drop_block_frac", 0.0)
            for seq in ("blue", "green"):
                _apply_chunking(p, seq, _pick(rng, STRUCTURE_SETS["static_pool"][1:]), effect, "mixed")
            return p

        cells.append(Cell("F_quality", i, code, {"variant": code, "effect": effect, **kw}, make))
    return cells


FAMILIES: Dict[str, Callable[[Dict[str, Any]], List[Cell]]] = {
    "A_null": family_a_null,
    "B_static": family_b_static,
    "C_dynamics": family_c_dynamics,
    "D_groups": family_d_groups,
    "E_realistic": family_e_realistic,
    "F_quality": family_f_quality,
}


def build_cells(calib: Dict[str, Any], families: Optional[List[str]] = None) -> List[Cell]:
    names = families or list(FAMILIES)
    cells: List[Cell] = []
    for name in names:
        if name not in FAMILIES:
            raise ValueError(f"Unknown family {name!r}. Available: {sorted(FAMILIES)}")
        cells.extend(FAMILIES[name](calib))
    return cells
