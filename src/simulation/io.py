"""Writing simulated sessions in the exact format of the real SRT files."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

FAMILY_TOKEN = {
    "A_null": "Anull",
    "B_static": "Bstatic",
    "C_dynamics": "Cdynamics",
    "D_groups": "Dgroups",
    "E_realistic": "Erealistic",
    "F_quality": "Fquality",
}


def participant_uid(cell_code: str, rep: int, pid: int) -> str:
    return f"{cell_code}-r{rep:02d}-p{pid:03d}"


def file_stem(family: str, cell_index: int, uid: str, session: int, start_date: _dt.date) -> str:
    """Stem compatible with src.chunking.run._parse_source_file_meta.

    SIM_<Family>_<NNN>_<uid>_Sim_<YYYYMMDD>_FRA_<session>_fertig -> participant_id '<uid>_Sim'.
    """
    token = FAMILY_TOKEN.get(family, "".join(c for c in family if c.isalpha()))
    date = (start_date + _dt.timedelta(days=session - 1)).strftime("%Y%m%d")
    return f"SIM_{token}_{cell_index:03d}_{uid}_Sim_{date}_FRA_{session}_fertig"


def write_srt_csv(events: pd.DataFrame, path: Path) -> None:
    """';'-separated, decimal comma, 7 significant digits, CRLF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(sep=";", decimal=",", float_format="%.7g", index=False)
    try:
        events.to_csv(path, lineterminator="\r\n", **kwargs)
    except TypeError:  # pandas < 1.5
        events.to_csv(path, line_terminator="\r\n", **kwargs)
