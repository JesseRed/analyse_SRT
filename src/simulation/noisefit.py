"""Fit the residual-noise parameters of the generator to the real data (grid search).

The noise model has parameters that cannot be read off the data directly (ex-Gaussian
core/tail, timestamp jitter, block noise, block AR(1)). They are chosen by simulating
null participants for every grid point, summarising them with the *same* estimator as
the real data (``calibrate.estimate``) and minimising the relative squared deviation of
the within-participant statistics below. The winner is stored as ``noise_fit`` in the
calibration file and used as the "mid" noise level of all scenarios.
"""

from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .calibrate import DEFAULT_CALIBRATION_PATH, collect_records, estimate, load_calibration

TARGET_KEYS = ["sd_block", "ar1_block", "sd_residual", "sd_residual_robust",
               "tail_left_log", "tail_right_log", "lag1_within_block"]
DEFAULT_GRID = {
    "sd_core": [0.04, 0.07, 0.10],
    "tau_ratio": [0.30, 0.45, 0.60],
    "jitter_sd": [0.02, 0.03],
    "sd_block": [0.08, 0.11],
    "ar1_block": [0.45, 0.60],
}


def _evaluate_point(args: Tuple[Dict[str, float], str, int]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    from .generator import simulate_session
    from .scenarios import base_participant

    params, calib_path, n_participants = args
    calib = load_calibration(calib_path)
    sessions = []
    for i in range(n_participants):
        rng = np.random.default_rng(1000 + i)
        p = base_participant(rng, "fit", calib)
        for k, v in params.items():
            setattr(p, k, v)
        sessions.append(simulate_session(p, 1, 1, calib, rng).events)
    est = estimate(collect_records(sessions))
    loss = float(sum(((est[k] - calib[k]) / max(abs(calib[k]), 0.05)) ** 2 for k in TARGET_KEYS))
    return loss, params, {k: float(est[k]) for k in TARGET_KEYS}


def fit_noise(calibration_path: str | Path = DEFAULT_CALIBRATION_PATH, grid: Dict[str, List[float]] = None,
              n_participants: int = 24, jobs: int = 8) -> Dict[str, Any]:
    grid = grid or DEFAULT_GRID
    names = list(grid)
    points = [dict(zip(names, vals)) for vals in itertools.product(*(grid[n] for n in names))]
    tasks = [(pt, str(calibration_path), n_participants) for pt in points]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_evaluate_point, tasks))
    else:
        results = [_evaluate_point(t) for t in tasks]
    results.sort(key=lambda r: r[0])
    best_loss, best_params, best_stats = results[0]
    calib = load_calibration(calibration_path)
    calib["noise_fit"] = {
        "params": best_params,
        "loss": best_loss,
        "simulated": best_stats,
        "target": {k: calib[k] for k in TARGET_KEYS},
        "grid": grid,
        "n_participants": n_participants,
    }
    Path(calibration_path).write_text(json.dumps(calib, indent=2), encoding="utf-8")
    return calib["noise_fit"]
