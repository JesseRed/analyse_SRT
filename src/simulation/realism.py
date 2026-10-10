"""Realism check: compare simulated files with the real data using the same estimators.

Simulated files are summarised with exactly the estimator used for calibration
(``calibrate.estimate``), so differences reflect the generator, not the measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .calibrate import collect_records, estimate, load_calibration


def _fmt(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, (float, int, np.floating)) else ("" if x is None else str(x))


def _rows(real: Dict[str, Any], sims: List[Dict[str, Any]]) -> List[List[str]]:
    def row(label: str, getter) -> List[str]:
        vals = [getter(real)] + [getter(s) for s in sims]
        return [label] + [_fmt(v) for v in vals]

    rows: List[List[str]] = []
    for key in ("mu_log_iki", "sd_between_files", "sd_block", "ar1_block", "sd_residual",
                "sd_residual_robust", "tail_left_log", "tail_right_log", "lag1_within_block",
                "post_error_slowing_log", "first_press_log_mu", "first_press_log_sd"):
        rows.append(row(key, lambda d, k=key: d.get(k)))
    for seq in ("blue", "green", "yellow"):
        rows.append(row(f"sequence_offset[{seq}]", lambda d, s=seq: d["sequence_offsets"].get(s)))
        rows.append(row(f"errors_per_block[{seq}]", lambda d, s=seq: d["errors"].get(s, {}).get("errors_per_block")))
        rows.append(row(f"frac_error_free[{seq}]", lambda d, s=seq: d["errors"].get(s, {}).get("frac_error_free_blocks")))
        for q in ("5", "50", "95", "99"):
            rows.append(row(f"IKI q{q} [{seq}] (s)", lambda d, s=seq, q=q: d["iki_quantiles"].get(s, {}).get(q)))
        rows.append([f"positional median [{seq}] (s)"] + [
            " ".join(f"{x:.2f}" for x in d["iki_quantiles"].get(seq, {}).get("positional_median", []))
            for d in [real] + sims
        ])
    return rows


def _plot(real: Dict[str, Any], sims: List[Dict[str, Any]], labels: List[str], out_png: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    colors = {"blue": "#2b6cb0", "green": "#2f855a", "yellow": "#b7791f"}
    styles = ["--s", ":^", "-.d", "--x"]
    for ax, seq in zip(axes, ("blue", "green", "yellow")):
        ax.plot(range(1, 8), real["iki_quantiles"][seq]["positional_median"], "-o", color=colors[seq], label="real")
        for sim, label, style in zip(sims, labels, styles):
            sq = sim["iki_quantiles"].get(seq, {}).get("positional_median")
            if sq:
                ax.plot(range(1, 8), sq, style, color="#555555", alpha=0.85, label=label)
        ax.set_title(f"{seq}: Median-IKI je Position")
        ax.set_xlabel("IKI-Position")
        ax.set_ylabel("s")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def realism_report(
    sim_dirs: List[str | Path],
    calibration_path: str | Path = "configs/simulation/empirical_calibration.json",
    output_md: str | Path = "simulations/core/realism_report.md",
    max_files: int = 300,
    seed: int = 1,
) -> Path:
    real = load_calibration(calibration_path)
    rng = np.random.default_rng(seed)
    sims: List[Dict[str, Any]] = []
    labels: List[str] = []
    counts: List[int] = []
    for spec in sim_dirs:
        # "<family dir>@<cell code>" restricts the comparison to one cell
        d, _, cell = str(spec).partition("@")
        pattern = f"*_{cell}-r*.csv" if cell else "*.csv"
        files = sorted(Path(d).glob(f"data/{pattern}")) or sorted(Path(d).glob(pattern))
        if not files:
            raise FileNotFoundError(f"No simulated CSVs for {spec}")
        if len(files) > max_files:
            files = [files[i] for i in sorted(rng.choice(len(files), max_files, replace=False))]
        sims.append(estimate(collect_records(files)))
        labels.append(Path(d).name + (f"@{cell}" if cell else ""))
        counts.append(len(files))

    out = Path(output_md)
    png = _plot(real, sims, labels, out.with_suffix(".png"))
    lines = [
        "# Realismus-Check: Simulation vs. echte Daten",
        "",
        f"Echte Daten: Kalibrierung `{calibration_path}` ({real.get('n_files')} Dateien). "
        f"Simuliert: {', '.join(f'{l} ({n} Dateien)' for l, n in zip(labels, counts))}.  ",
        "Alle Spalten wurden mit demselben Schätzer (`src/simulation/calibrate.estimate`) berechnet.",
        "",
        "Lesehilfe: `A_null@A5-seqspeedup` enthält alle realistischen Störfaktoren, aber keine Chunks, "
        "und prüft den Realismus der Simulation direkt. "
        "Familien mit Chunk-Struktur (z. B. `E_realistic`) weichen bei Blue/Green absichtlich ab "
        "(Grenz-Spikes verlangsamen einzelne Positionen); Yellow muss dort weiterhin passen. "
        "`sd_between_files` ist in den meisten Familien absichtlich auf 50 % der echten Streuung gesetzt.",
        "",
        "| Kennwert | echt | " + " | ".join(labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    lines += ["| " + " | ".join(r) + " |" for r in _rows(real, sims)]
    if png:
        lines += ["", f"![Positionsprofile]({png.name})"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {lab: {k: v for k, v in sim.items() if k != "schedules"} for lab, sim in zip(labels, sims)}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
