"""
Auswerteroutine für Chunking-Outputs (change_point_pelt etc.).

Liest die Summary-Daten aus outputs/<run>/change_point_pelt/{blue,green,yellow},
erstellt eine zusammengeführte Tabelle und einen wissenschaftlich aufbereiteten
Bericht: Verlauf des Chunkings über die Tage (pro Sequenz, pro Proband) und
Unterschiede zwischen den Sequenzen (Blue/Green vs Yellow, Blue vs Green).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .run import merge_sequence_summaries


def load_merged_summary(method_output_dir: str | Path) -> pd.DataFrame:
    """
    Lade die zusammengeführte Summary-Tabelle (blue/green/yellow).
    Erstellt combined_summary.csv per merge_sequence_summaries, falls nicht vorhanden.
    """
    base = Path(method_output_dir)
    combined_path = base / "combined_summary.csv"
    if not combined_path.exists():
        merge_sequence_summaries(base)
    return pd.read_csv(combined_path)


def _section_change_over_days(merged: pd.DataFrame) -> str:
    """Berichtsabschnitt: Veränderung des Chunkings über die Tage."""
    # Long-format pro Sequenz aus merged bauen (merged hat pro Zeile participant, session, day + mean_n_chunks_blue etc.)
    lines = [
        "## 1. Veränderung des Chunkings über die Tage",
        "",
        "Für jede Sequenz (Blue, Green, Yellow) und jeden Probanden wird die mittlere Chunk-Anzahl "
        "(*mean_n_chunks*) über die Messzeitpunkte (Tag 1, 2, 3) beschrieben. *mean_n_chunks* > 1 "
        "deutet auf erkennbare Chunk-Grenzen hin.",
        "",
    ]

    participants = merged["participant_id"].dropna().unique()
    for seq in ("blue", "green", "yellow"):
        col = f"mean_n_chunks_{seq}"
        if col not in merged.columns:
            continue
        lines.append(f"### Sequenz: {seq.capitalize()}")
        lines.append("")
        for pid in sorted(participants):
            sub = merged[merged["participant_id"] == pid].sort_values("day_index")
            if sub.empty:
                continue
            vals = sub[["day_index", col]].dropna(subset=[col])
            if vals.empty:
                lines.append(f"- **{pid}**: keine Daten")
            else:
                traj = ", ".join([f"Tag {int(d)}: {v:.2f}" for d, v in vals.values])
                lines.append(f"- **{pid}**: {traj}")
        lines.append("")

    # Hinweis bei wenigen Fenstern
    min_windows_reliable = 3
    low_window_rows: list[str] = []
    for seq in ("blue", "green", "yellow"):
        n_col = f"n_windows_valid_{seq}"
        if n_col not in merged.columns:
            continue
        for _, row in merged.iterrows():
            n = row.get(n_col)
            if pd.notna(n) and int(n) < min_windows_reliable:
                pid = row.get("participant_id", "?")
                day = row.get("day_index", "?")
                day_str = int(day) if pd.notna(day) else day
                low_window_rows.append(
                    f"**{pid}**, Sequenz {seq.capitalize()}, Tag {day_str}: "
                    f"nur {int(n)} gültige Fenster (Ergebnisse mit Vorsicht interpretieren)"
                )
    if low_window_rows:
        lines.append("### Hinweise zu Datenqualität")
        lines.append("")
        lines.append(
            f"Folgende Messzeitpunkte haben weniger als {min_windows_reliable} gültige Fenster; "
            "die Ergebnisse sind mit Vorsicht zu interpretieren:"
        )
        lines.append("")
        for r in low_window_rows:
            lines.append(f"- {r}")
        lines.append("")

    # Kurzfassung: Wer hat über die Tage Zunahme/Abnahme?
    lines.append("### Kurzfassung Verlauf")
    lines.append("")
    for seq in ("blue", "green", "yellow"):
        col = f"mean_n_chunks_{seq}"
        if col not in merged.columns:
            continue
        per_part = merged.groupby("participant_id")[col].agg(["mean", "min", "max", "std"])
        per_part = per_part.round(3)
        lines.append(f"- **{seq.capitalize()}**: Mittelwerte (M, Min, Max, SD) pro Proband: ")
        lines.append(per_part.to_string())
        lines.append("")
    return "\n".join(lines)


def _section_between_sequences(merged: pd.DataFrame) -> str:
    """Berichtsabschnitt: Unterschiede zwischen den Sequenzen."""
    lines = [
        "## 2. Unterschiede zwischen den Sequenzen",
        "",
        "Yellow dient als Null-Bedingung (zufällige Sequenz); Blue und Green sind strukturierte "
        "Sequenzen. Positive Deltas (Blue/Green minus Yellow) bedeuten mehr Chunk-Struktur "
        "in der strukturierten Sequenz.",
        "",
    ]

    delta_blue = "delta_mean_n_chunks_blue_vs_yellow"
    delta_green = "delta_mean_n_chunks_green_vs_yellow"
    has_delta_blue = delta_blue in merged.columns
    has_delta_green = delta_green in merged.columns

    if has_delta_blue or has_delta_green:
        lines.append("### Strukturierte Sequenzen vs. Yellow (pro Messzeitpunkt)")
        lines.append("")
        for _, row in merged.iterrows():
            pid = row.get("participant_id", "?")
            day = row.get("day_index", "?")
            delta_parts = []
            if has_delta_blue and pd.notna(row.get(delta_blue)):
                delta_parts.append(f"Blue−Yellow = {row[delta_blue]:.3f}")
            if has_delta_green and pd.notna(row.get(delta_green)):
                delta_parts.append(f"Green−Yellow = {row[delta_green]:.3f}")
            if delta_parts:
                day_str = int(day) if pd.notna(day) else day
                lines.append(f"**{pid}**, Tag {day_str}: " + ", ".join(delta_parts))
        lines.append("")

    # Aggregation über alle Messzeitpunkte
    lines.append("### Aggregation über Probanden und Tage")
    lines.append("")
    agg_rows = []
    for seq in ("blue", "green", "yellow"):
        col = f"mean_n_chunks_{seq}"
        if col not in merged.columns:
            continue
        agg_rows.append({
            "Sequenz": seq.capitalize(),
            "M (mean_n_chunks)": merged[col].mean(),
            "SD": merged[col].std(),
            "Min": merged[col].min(),
            "Max": merged[col].max(),
            "N": merged[col].notna().sum(),
        })
    if agg_rows:
        agg_df = pd.DataFrame(agg_rows).round(4)
        lines.append(agg_df.to_string(index=False))
        lines.append("")

    if has_delta_blue and has_delta_green:
        lines.append("Deltas (strukturiert − Yellow):")
        lines.append(f"- Blue−Yellow: M = {merged[delta_blue].mean():.3f}, SD = {merged[delta_blue].std():.3f}")
        lines.append(f"- Green−Yellow: M = {merged[delta_green].mean():.3f}, SD = {merged[delta_green].std():.3f}")
        lines.append("")

    return "\n".join(lines)


def build_evaluation_report(merged: pd.DataFrame) -> str:
    """Erzeuge den vollständigen Markdown-Bericht aus der zusammengeführten Tabelle."""
    # merged hat pro Zeile einen Messzeitpunkt (participant_id, session, day_index) mit mean_n_chunks_blue/green/yellow
    # Für "Verlauf über Tage" brauchen wir pro Proband und Sequenz die Zeitreihe. Dazu long-format pro Sequenz:
    # Wir haben bereits pro Zeile alle drei Sequenzen. Verlauf = Zeilen pro Proband sortiert nach day_index.
    intro = [
        "# Auswertung: Chunking über Tage und Sequenzen",
        "",
        "Dieser Bericht fasst die Change-Point-basierte Chunk-Analyse (PELT) für die Sequenztypen "
        "Blue, Green und Yellow zusammen. Untersucht werden (1) die Veränderung des Chunkings über "
        "die Messzeitpunkte (Tage) pro Sequenz und Proband sowie (2) die Unterschiede zwischen "
        "den Sequenzen (strukturierte Sequenzen Blue/Green vs. Null-Bedingung Yellow).",
        "",
    ]

    report = "\n".join(intro)
    report += "\n\n"
    report += _section_change_over_days(merged)
    report += "\n\n"
    report += _section_between_sequences(merged)
    return report


def evaluate_outputs(
    method_output_dir: str | Path,
    output_report_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Auswerteroutine: merged Summary laden, Bericht und optional CSV schreiben.

    - method_output_dir: z.B. outputs/SRT_Test2/change_point_pelt
    - output_report_path: Pfad für evaluation_report.md (Default: <method_output_dir>/evaluation_report.md)
    - output_csv_path: optional, Pfad für evaluation_data.csv (merged + Deltas)
    """
    base = Path(method_output_dir)
    merged = load_merged_summary(base)

    # Für Verlauf über Tage: long-format pro Sequenz erzeugen (für einheitliche Darstellung)
    # merged ist bereits one row per (source_file / participant, session, day) mit blue/green/yellow Spalten
    report = build_evaluation_report(merged)

    report_path = Path(output_report_path) if output_report_path else base / "evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    out: dict[str, Any] = {
        "report_path": str(report_path),
        "merged_rows": len(merged),
        "participants": merged["participant_id"].dropna().unique().tolist(),
    }

    if output_csv_path is not None:
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(csv_path, index=False)
        out["csv_path"] = str(csv_path)

    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: Auswertung für ein Method-Output-Verzeichnis."""
    import argparse
    parser = argparse.ArgumentParser(description="Chunking-Outputs auswerten (Bericht + optional CSV).")
    parser.add_argument(
        "method_output_dir",
        type=Path,
        default=Path("outputs/SRT_Test2/change_point_pelt"),
        nargs="?",
        help="Verzeichnis mit blue/green/yellow/summary.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Pfad für evaluation_report.md",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional: Pfad für evaluation_data.csv",
    )
    args = parser.parse_args(argv)
    result = evaluate_outputs(
        args.method_output_dir,
        output_report_path=args.report,
        output_csv_path=args.csv,
    )
    print(f"Report: {result['report_path']}")
    if result.get("csv_path"):
        print(f"CSV:   {result['csv_path']}")
    print(f"Zeilen: {result['merged_rows']}, Probanden: {result['participants']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
