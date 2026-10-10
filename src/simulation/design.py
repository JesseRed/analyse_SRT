"""Experimental design of the SRTT as used in the real data (sequences, block schedules)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

KEYS = (1, 2, 3, 4)

# Fixed sequences, identical for all participants in the real data.
BLUE = "23143124"
GREEN = "34213241"
# Pool used for "yellow" blocks. Every real file uses all 8 (mostly 5x each in 40 blocks).
# NOTE: the pool contains the blue sequence itself (23143124).
YELLOW_POOL = (
    "43143142",
    "21413243",
    "13424132",
    "24134231",
    "23143124",
    "41324312",
    "12413423",
    "21342412",
)

SEQUENCE_NAMES = ("blue", "green", "yellow")
RATIO = {"blue": 2, "green": 1, "yellow": 1}


def _max_runs_ok(schedule: Sequence[str], max_run: Dict[str, int]) -> bool:
    run_char, run_len = None, 0
    for c in schedule:
        run_len = run_len + 1 if c == run_char else 1
        run_char = c
        if run_len > max_run.get(c, 99):
            return False
    return True


def random_schedule(n_blocks: int, rng: np.random.Generator) -> str:
    """Pseudo-random 2:1:1 schedule with the run-length limits seen in real data."""
    n_unit = n_blocks // 4
    counts = {"b": 2 * n_unit, "g": n_unit, "y": n_blocks - 3 * n_unit}
    letters = [c for c, n in counts.items() for _ in range(n)]
    max_run = {"b": 7, "g": 4, "y": 4}
    for _ in range(10_000):
        perm = rng.permutation(letters)
        if _max_runs_ok(perm, max_run):
            return "".join(perm)
    raise RuntimeError("Could not construct a schedule satisfying the run-length limits.")


def choose_schedule(
    n_blocks: int,
    rng: np.random.Generator,
    real_schedules: Optional[Sequence[str]] = None,
) -> str:
    """Use a real schedule of the requested length if available, else a random one."""
    if real_schedules:
        pool = [s for s in real_schedules if len(s) == n_blocks and _ratio_ok(s)]
        if pool:
            return str(pool[int(rng.integers(len(pool)))])
    return random_schedule(n_blocks, rng)


def _ratio_ok(schedule: str) -> bool:
    n = len(schedule)
    return schedule.count("b") == n // 2 and schedule.count("g") in (n // 4, n // 4 + 1)


def yellow_assignment(n_yellow: int, rng: np.random.Generator) -> List[str]:
    """Balanced, shuffled assignment of the yellow pool to the yellow blocks."""
    reps = int(np.ceil(n_yellow / len(YELLOW_POOL)))
    seqs = list(YELLOW_POOL) * reps
    seqs = [seqs[i] for i in rng.permutation(len(seqs))][:n_yellow]
    return seqs


def target_string(seq_type: str, yellow_seq: Optional[str] = None) -> str:
    if seq_type == "blue":
        return BLUE
    if seq_type == "green":
        return GREEN
    if yellow_seq is None:
        raise ValueError("yellow blocks need a pool sequence")
    return yellow_seq


class NGramLearner:
    """Online Bayesian n-gram learner over the interleaved stimulus stream.

    Used for the optional 'predictive' timing component: log IKI grows with the
    surprisal of the upcoming stimulus. Context resets at block start (each block is
    preceded by a pause), Dirichlet smoothing with back-off to lower orders.
    """

    def __init__(self, order: int = 2, alpha: float = 1.0, decay: float = 0.0):
        self.order = int(order)
        self.alpha = float(alpha)
        self.decay = float(decay)
        self.counts: Dict[tuple, np.ndarray] = {}

    def _dist(self, context: tuple) -> np.ndarray:
        base = np.full(len(KEYS), 1.0 / len(KEYS))
        for n in range(0, self.order + 1):
            if n > len(context):
                break
            ctx = context[len(context) - n:] if n > 0 else ()
            c = self.counts.get(ctx)
            if c is None:
                continue
            base = (c + self.alpha * len(KEYS) * base) / (c.sum() + self.alpha * len(KEYS))
        return base

    def surprisal(self, context: tuple, key: int) -> float:
        p = self._dist(context)[key - 1]
        return float(-np.log2(max(p, 1e-12)))

    def update(self, context: tuple, key: int) -> None:
        if self.decay > 0:
            for k in self.counts:
                self.counts[k] *= (1.0 - self.decay)
        for n in range(0, self.order + 1):
            if n > len(context):
                break
            ctx = context[len(context) - n:] if n > 0 else ()
            if ctx not in self.counts:
                self.counts[ctx] = np.zeros(len(KEYS))
            self.counts[ctx][key - 1] += 1.0
