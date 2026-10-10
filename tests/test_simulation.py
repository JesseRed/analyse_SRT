"""Tests for the simulation package (run: python -m pytest tests/test_simulation.py)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.chunking._data import extract_ikis, load_srt_file
from src.chunking.run import _parse_source_file_meta
from src.simulation import structures as st
from src.simulation.calibrate import DEFAULT_CALIBRATION_PATH, load_calibration
from src.simulation.design import BLUE, GREEN, YELLOW_POOL
from src.simulation.evaluate import adjusted_rand_index, score_block
from src.simulation.generate import _generate_participant
from src.simulation.generator import ParticipantSpec, simulate_session
from src.simulation.io import file_stem, write_srt_csv
from src.simulation.scenarios import base_participant, build_cells

pytestmark = pytest.mark.skipif(not DEFAULT_CALIBRATION_PATH.exists(), reason="calibration file missing")


@pytest.fixture(scope="module")
def calib():
    return load_calibration()


# --- structures ---------------------------------------------------------------------

@pytest.mark.parametrize("name,boundaries", [
    ("8", []), ("4-4", [4]), ("3-3-2", [3, 6]), ("2-2-2-2", [2, 4, 6]), ("1-7", [1]), ("7-1", [7]),
])
def test_partition_roundtrip(name, boundaries):
    lengths = st.parse_structure(name)
    assert st.lengths_to_boundaries(lengths) == boundaries
    assert st.boundaries_to_lengths(boundaries) == lengths
    assert st.structure_name(lengths) == name


def test_segment_conversion():
    assert st.boundaries_to_segment_cps([4]) == [3, 4]
    assert st.boundaries_to_segment_cps([1]) == [1]       # IKI_1 slow -> only cp 1 visible
    assert st.boundaries_to_segment_cps([7]) == [6]
    assert st.segment_cps_to_iki_labels([3, 4]) == [0, 0, 0, 1, 2, 2, 2]


def test_chunk_of_iki_and_position():
    # 3-3-2: IKIs 1,2 in chunk 0; IKI 3 boundary (starts chunk 1); 4,5 in chunk 1; 6 boundary; 7 in chunk 2
    assert st.chunk_of_iki((3, 3, 2)) == [0, 0, 1, 1, 1, 2, 2]
    pos = st.position_in_chunk((3, 3, 2))
    assert pos[2] == 0.0 and pos[5] == 0.0
    assert pos[0] < pos[1]


def test_invalid_partition():
    with pytest.raises(ValueError):
        st.parse_structure("3-3")


# --- scoring --------------------------------------------------------------------------

def test_scoring_conventions():
    perfect_seg = score_block([3, 4], [4], "segment_cp")
    assert perfect_seg["f1_conv"] == 1.0 and perfect_seg["exact"]
    half_seg = score_block([3], [4], "segment_cp")       # one of the two cps -> boundary still detected
    assert half_seg["recall_conv"] == 1.0 and half_seg["precision_conv"] == 1.0
    assert half_seg["f1_strict"] < 1.0
    iki = score_block([4], [4], "iki_index")
    assert iki["f1_conv"] == 1.0
    miss = score_block([2], [4], "iki_index")
    assert miss["f1_conv"] == 0.0 and miss["f1_tol1"] == 0.0
    null = score_block([], [], "iki_index")
    assert null["f1_conv"] == 1.0 and not null["false_alarm"]
    assert score_block([5], [], "iki_index")["false_alarm"]


def test_ari():
    assert adjusted_rand_index([0, 0, 1, 1], [5, 5, 7, 7]) == pytest.approx(1.0)
    assert adjusted_rand_index([0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    assert adjusted_rand_index([0, 1, 0, 1], [0, 0, 1, 1]) < 0.0


# --- generator ------------------------------------------------------------------------

def _session(calib, seed=1, **seq_kw):
    rng = np.random.default_rng(seed)
    p = base_participant(rng, "t", calib)
    for k, v in seq_kw.items():
        setattr(p.sequences["blue"], k, v)
    return simulate_session(p, 1, 1, calib, rng)


def test_session_format(calib, tmp_path):
    res = _session(calib)
    ev = res.events
    assert list(ev.columns) == ["BlockNumber", "EventNumber", "Time Since Block start", "isHit", "target", "pressed", "sequence"]
    assert (ev.groupby("BlockNumber").size() == 8).all()
    assert ev["EventNumber"].tolist() == list(range(1, len(ev) + 1))
    counts = ev.groupby("BlockNumber")["sequence"].first().value_counts()
    assert counts["blue"] == 80 and counts["green"] == 40 and counts["yellow"] == 40
    blocks = ev.groupby("BlockNumber").agg(seq=("sequence", "first"), t=("target", lambda x: "".join(map(str, x))))
    assert set(blocks[blocks.seq == "blue"].t) == {BLUE}
    assert set(blocks[blocks.seq == "green"].t) == {GREEN}
    assert set(blocks[blocks.seq == "yellow"].t) <= set(YELLOW_POOL)
    assert (ev.groupby("BlockNumber")["Time Since Block start"].apply(lambda s: s.diff().dropna().gt(0).all())).all()
    errs = ev[ev.isHit == 0]
    assert (errs["pressed"] != errs["target"]).all()
    assert (ev[ev.isHit == 1]["pressed"] == ev[ev.isHit == 1]["target"]).all()

    path = tmp_path / "x.csv"
    write_srt_csv(ev, path)
    raw = path.read_bytes()
    assert b"\r\n" in raw and b";" in raw
    df = load_srt_file(path)
    assert len(df) == len(ev)
    assert extract_ikis(df, "blue")


def test_determinism(calib, tmp_path):
    args = ("B_static", 48, 1, 1, 1, 99, str(DEFAULT_CALIBRATION_PATH), str(tmp_path / "a"), dt.date(2025, 1, 6))
    m1, t1 = _generate_participant(args)
    args2 = args[:7] + (str(tmp_path / "b"),) + args[8:]
    m2, t2 = _generate_participant(args2)
    f1 = Path(m1[0]["file"]).read_bytes()
    f2 = Path(m2[0]["file"]).read_bytes()
    assert f1 == f2
    pd.testing.assert_frame_equal(t1[0], t2[0])


def test_filename_parse_unique():
    a = file_stem("B_static", 48, "d050-nmid-spike-r01-p001", 3, dt.date(2025, 1, 6))
    b = file_stem("B_static", 48, "d050-nmid-spike-r01-p002", 3, dt.date(2025, 1, 6))
    ma, mb = _parse_source_file_meta(a + ".csv"), _parse_source_file_meta(b + ".csv")
    assert ma["day_index"] == 3 and ma["session"] == "FRA_3"
    assert ma["participant_id"] != mb["participant_id"]


def test_spike_creates_boundary_profile(calib):
    """With a strong spike the boundary IKI must be slower than its neighbours; without it not."""
    def profile(delta):
        res = _session(calib, seed=5, structure="4-4", delta=delta)
        ev = res.events
        rows = []
        for b, g in ev[ev.sequence == "blue"].groupby("BlockNumber"):
            if g.isHit.all():
                rows.append(np.log(np.diff(g["Time Since Block start"].to_numpy())))
        return np.median(np.vstack(rows), axis=0)

    with_spike = profile(0.5)
    without = profile(0.0)
    contrast = lambda pr: pr[3] - 0.5 * (pr[2] + pr[4])  # noqa: E731
    assert contrast(with_spike) - contrast(without) > 0.3


def test_truth_columns(calib):
    res = _session(calib, structure="3-3-2", delta=0.3)
    t = res.truth
    blue = t[t.sequence_type == "blue"]
    assert set(blue["true_boundaries"]) == {json.dumps([3, 6])}
    assert set(t[t.sequence_type == "yellow"]["true_n_chunks"]) == {1}


def test_all_cells_build(calib):
    cells = build_cells(calib)
    codes = [(c.family, c.code) for c in cells]
    assert len(codes) == len(set(codes))
    rng = np.random.default_rng(0)
    for c in cells:
        p = c.make(rng, "u")
        assert isinstance(p, ParticipantSpec)
        for code_char in c.code:
            assert code_char.isalnum() or code_char == "-"
