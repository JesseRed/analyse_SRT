"""Tier configuration (how much data to generate) for the simulation study."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .scenarios import FAMILIES


@dataclass
class FamilySize:
    n_participants: int
    n_sessions: int
    n_reps: int = 1
    cells: Optional[List[str]] = None  # restrict to these cell codes


@dataclass
class SimulationConfig:
    tier: str
    output_dir: Path
    base_seed: int
    calibration: Path
    start_date: _dt.date
    families: Dict[str, FamilySize] = field(default_factory=dict)


def load_config(path: str | Path) -> SimulationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    families: Dict[str, FamilySize] = {}
    for name, size in (raw.get("families") or {}).items():
        if name not in FAMILIES:
            raise ValueError(f"Unknown family {name!r} in {path}. Available: {sorted(FAMILIES)}")
        families[name] = FamilySize(
            n_participants=int(size["n_participants"]),
            n_sessions=int(size["n_sessions"]),
            n_reps=int(size.get("n_reps", 1)),
            cells=size.get("cells"),
        )
    start = raw.get("start_date", "2025-01-06")
    if isinstance(start, str):
        start = _dt.date.fromisoformat(start)
    return SimulationConfig(
        tier=str(raw["tier"]),
        output_dir=Path(raw["output_dir"]),
        base_seed=int(raw["base_seed"]),
        calibration=Path(raw.get("calibration", "configs/simulation/empirical_calibration.json")),
        start_date=start,
        families=families,
    )
