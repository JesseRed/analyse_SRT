"""Score chunking-method outputs against the simulation ground truth.

Reads only the standard outputs (``trials.csv``) written by ``run-methods`` or by the
normal chunking CLI, so any future method that follows the output contract can be
evaluated without code changes - only its boundary convention has to be declared in
``METHOD_CONVENTIONS``.

Layout expected:  <results>/<family>/<method>/<sequence>/trials.csv
Ground truth:     <sim_dir>/<family>/ground_truth.csv and manifest.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import structures as st

# How a value k in chunk_boundaries is to be read (see structures.py).
METHOD_CONVENTIONS: Dict[str, str] = {
    "rational_chunking": "iki_index",
    "hcrp_lm": "iki_index",
    "community_network": "segment_cp",
    "change_point_pelt": "segment_cp",
    "hsmm_chunking": "segment_cp",
}
# Methods that estimate one structure per sliding window and copy it to the blocks.
WINDOW_METHODS = {"change_point_pelt", "rational_chunking"}


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------


def parse_list(value: Any) -> List[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text in ("[]", "nan"):
        return []
    try:
        return [int(v) for v in json.loads(text)]
    except (ValueError, TypeError):
        return [int(float(v)) for v in text.strip("[]").replace(" ", "").split(",") if v]


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    """ARI of two labelings (1.0 if both are a single cluster)."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.size
    if n < 2:
        return float("nan")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2.0  # noqa: E731
    sum_ij = comb(table).sum()
    sum_a = comb(table.sum(axis=1)).sum()
    sum_b = comb(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb(n)
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:
        return 1.0
    return float((sum_ij - expected) / (max_index - expected))


def _prf(tp_pred: int, n_pred: int, tp_true: int, n_true: int) -> Dict[str, float]:
    precision = tp_pred / n_pred if n_pred else float("nan")
    recall = tp_true / n_true if n_true else float("nan")
    if n_pred == 0 and n_true == 0:
        f1 = 1.0
    elif np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def score_block(pred: Iterable[int], true_boundaries: Iterable[int], convention: str) -> Dict[str, Any]:
    """All block-level scores for one predicted boundary list.

    * ``*_conv``: convention-aware matching (primary). A predicted segment change point c
      detects the true boundary IKI k if c in {k-1, k}; for iki_index methods c == k.
    * ``*_strict``: exact set match in the method's native convention.
    * ``*_tol1``: native convention, +-1 position tolerance.
    """
    pred = sorted({int(x) for x in pred})
    truth = sorted({int(x) for x in true_boundaries})
    if convention == "segment_cp":
        native_truth = st.boundaries_to_segment_cps(truth)
        matches = lambda c, k: c in (k - 1, k)  # noqa: E731
        labels_pred = st.segment_cps_to_iki_labels(pred)
        labels_true = st.segment_cps_to_iki_labels(native_truth)
    elif convention == "iki_index":
        native_truth = truth
        matches = lambda c, k: c == k  # noqa: E731
        labels_pred = st.boundaries_to_item_labels(pred)
        labels_true = st.boundaries_to_item_labels(truth)
    else:
        raise ValueError(f"unknown convention {convention!r}")

    tp_pred = sum(any(matches(c, k) for k in truth) for c in pred)
    tp_true = sum(any(matches(c, k) for c in pred) for k in truth)
    conv = _prf(tp_pred, len(pred), tp_true, len(truth))

    inter = len(set(pred) & set(native_truth))
    strict = _prf(inter, len(pred), inter, len(native_truth))

    tol_pred = sum(any(abs(c - k) <= 1 for k in native_truth) for c in pred)
    tol_true = sum(any(abs(c - k) <= 1 for c in pred) for k in native_truth)
    tol = _prf(tol_pred, len(pred), tol_true, len(native_truth))

    n_pos = st.N_IKIS - 1 if convention == "segment_cp" else st.N_IKIS
    vec_p = np.zeros(n_pos, dtype=bool)
    vec_t = np.zeros(n_pos, dtype=bool)
    vec_p[[c - 1 for c in pred if 1 <= c <= n_pos]] = True
    vec_t[[c - 1 for c in native_truth if 1 <= c <= n_pos]] = True

    return {
        "n_pred": len(pred),
        "n_true": len(truth),
        "false_alarm": bool(len(truth) == 0 and len(pred) > 0),
        "precision_conv": conv["precision"],
        "recall_conv": conv["recall"],
        "f1_conv": conv["f1"],
        "precision_strict": strict["precision"],
        "recall_strict": strict["recall"],
        "f1_strict": strict["f1"],
        "f1_tol1": tol["f1"],
        "exact": bool(set(pred) == set(native_truth)),
        "hamming": int((vec_p != vec_t).sum()),
        "ari": adjusted_rand_index(labels_pred, labels_true),
        "n_chunks_pred": len(pred) + 1,
        "n_chunks_true_native": len(native_truth) + 1,
    }


# ----------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------


def load_truth(sim_family_dir: Path) -> pd.DataFrame:
    truth = pd.read_csv(sim_family_dir / "ground_truth.csv")
    manifest = pd.read_csv(sim_family_dir / "manifest.csv")
    factors = manifest[["source_stem", "factors"]].copy()
    expanded = factors["factors"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    fac_df = pd.DataFrame(list(expanded)).add_prefix("f_")
    fac_df["source_stem"] = factors["source_stem"].to_numpy()
    keep = ["source_stem", "group"]
    truth = truth.merge(fac_df, on="source_stem", how="left")
    truth = truth.merge(manifest[[c for c in keep if c in manifest.columns]], on="source_stem", how="left")
    return truth


def load_method_trials(method_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(method_dir.glob("*/trials.csv")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["source_stem"] = df["source_file"].map(lambda s: Path(str(s)).stem)
        frames.append(df[["source_stem", "sequence_type", "block_number", "chunk_boundaries"]])
    if not frames:
        return pd.DataFrame(columns=["source_stem", "sequence_type", "block_number", "chunk_boundaries"])
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------


def evaluate_method(method: str, trials: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    convention = METHOD_CONVENTIONS.get(method)
    if convention is None:
        raise ValueError(f"No boundary convention declared for method {method!r} (METHOD_CONVENTIONS).")
    merged = trials.merge(truth, on=["source_stem", "sequence_type", "block_number"], how="inner")
    rows = []
    for rec in merged.itertuples(index=False):
        pred = parse_list(rec.chunk_boundaries)
        true_b = parse_list(rec.true_boundaries)
        row = score_block(pred, true_b, convention)
        if method in WINDOW_METHODS:
            wm = score_block(pred, parse_list(rec.window_modal_boundaries), convention)
            row.update({f"{k}_wm": wm[k] for k in ("f1_conv", "recall_conv", "precision_conv", "exact")})
        row.update({"source_stem": rec.source_stem, "sequence_type": rec.sequence_type,
                    "block_number": rec.block_number, "_pred_list": pred})
        rows.append(row)
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    scores.insert(0, "method", method)
    scores.insert(1, "convention", convention)
    return scores.merge(truth, on=["source_stem", "sequence_type", "block_number"], how="left")


def _cluster_bootstrap_se(df: pd.DataFrame, value: str, cluster: str = "participant_uid",
                          n_boot: int = 200, seed: int = 0) -> float:
    groups = df.groupby(cluster)[value].agg(["sum", "count"])
    groups = groups[groups["count"] > 0]
    if len(groups) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    sums, counts = groups["sum"].to_numpy(), groups["count"].to_numpy()
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    est = sums[idx].sum(axis=1) / np.maximum(counts[idx].sum(axis=1), 1)
    return float(np.std(est, ddof=1))


SUMMARY_METRICS = ["f1_conv", "recall_conv", "precision_conv", "f1_strict", "f1_tol1",
                   "exact", "ari", "hamming", "n_pred", "n_true"]


def summarize_cells(scores: pd.DataFrame) -> pd.DataFrame:
    """Per family x cell x method x sequence: mean metrics + cluster-bootstrap MCSE."""
    out = []
    keys = ["family", "cell", "method", "sequence_type"]
    for key, grp in scores.groupby(keys):
        row: Dict[str, Any] = dict(zip(keys, key))
        row["n_files"] = grp["source_stem"].nunique()
        row["n_participants"] = grp["participant_uid"].nunique()
        row["n_blocks"] = len(grp)
        for m in SUMMARY_METRICS:
            vals = pd.to_numeric(grp[m], errors="coerce")
            row[m] = float(vals.mean())
            tmp = grp.assign(_v=vals).dropna(subset=["_v"])
            row[f"{m}_mcse"] = _cluster_bootstrap_se(tmp, "_v") if len(tmp) else float("nan")
        null_blocks = grp[grp["n_true"] == 0]
        row["false_alarm_rate"] = float(null_blocks["false_alarm"].mean()) if len(null_blocks) else float("nan")
        row["spurious_per_null_block"] = float(null_blocks["n_pred"].mean()) if len(null_blocks) else float("nan")
        row["n_chunks_bias"] = float((grp["n_chunks_pred"] - grp["n_chunks_true_native"]).mean())
        row["n_chunks_rmse"] = float(np.sqrt(((grp["n_chunks_pred"] - grp["n_chunks_true_native"]) ** 2).mean()))
        if "f1_conv_wm" in grp:
            row["f1_conv_window_modal"] = float(grp["f1_conv_wm"].mean())
        for f in ("f_effect", "f_noise", "f_channel", "f_variant", "f_group"):
            if f in grp and grp[f].notna().any():
                row[f[2:]] = grp[f].dropna().iloc[0]
        out.append(row)
    return pd.DataFrame(out)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC (P[pos > neg] + 0.5 P[tie])."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def static_power(scores: pd.DataFrame) -> pd.DataFrame:
    """Family B: threshold-free detectability per effect size (AUC vs. effect 0) and d50."""
    b = scores[scores["family"] == "B_static"]
    if b.empty:
        return pd.DataFrame()
    per_file = (
        b.groupby(["method", "sequence_type", "f_noise", "f_channel", "f_effect", "source_stem"])
        .agg(mean_pred=("n_pred", "mean"), mean_true=("n_true", "mean"), recall=("recall_conv", "mean"))
        .reset_index()
    )
    rows = []
    for key, grp in per_file.groupby(["method", "sequence_type", "f_noise", "f_channel"]):
        null = grp[(grp["f_effect"] == 0) | (grp["mean_true"] == 0)]["mean_pred"].to_numpy()
        effects = sorted(grp["f_effect"].unique())
        rec_curve = []
        for eff in effects:
            pos = grp[(grp["f_effect"] == eff) & (grp["mean_true"] > 0)]
            rec = float(pos["recall"].mean()) if len(pos) else float("nan")
            rec_curve.append((eff, rec))
            rows.append({
                "method": key[0], "sequence_type": key[1], "noise": key[2], "channel": key[3],
                "effect": eff, "auc_vs_null": _auc(pos["mean_pred"].to_numpy(), null) if eff > 0 else float("nan"),
                "recall_conv": rec, "n_files_pos": len(pos), "n_files_null": len(null),
            })
        d50 = float("nan")
        for (e0, r0), (e1, r1) in zip(rec_curve, rec_curve[1:]):
            if not np.isnan(r0) and not np.isnan(r1) and r0 < 0.5 <= r1:
                d50 = e0 + (0.5 - r0) * (e1 - e0) / (r1 - r0)
                break
        for r in rows:
            if (r["method"], r["sequence_type"], r["noise"], r["channel"]) == key:
                r["d50_recall"] = d50
    return pd.DataFrame(rows)


def null_positions(scores: pd.DataFrame) -> pd.DataFrame:
    """Family A: where do spurious boundaries appear (per method, variant, position)?"""
    a = scores[(scores["family"] == "A_null")]
    if a.empty:
        return pd.DataFrame()
    rows = []
    for (method, variant, seq), grp in a.groupby(["method", "f_variant", "sequence_type"]):
        n_blocks = len(grp)
        conv = grp["convention"].iloc[0]
        n_pos = 6 if conv == "segment_cp" else 7
        counts = np.zeros(n_pos)
        for lst in grp["_pred_list"]:
            for c in lst:
                if 1 <= c <= n_pos:
                    counts[c - 1] += 1
        rows.append({"method": method, "variant": variant, "sequence_type": seq, "n_blocks": n_blocks,
                     "false_alarm_rate": float(grp["false_alarm"].mean()),
                     **{f"pos{k + 1}": counts[k] / n_blocks for k in range(n_pos)}})
    return pd.DataFrame(rows)


def dynamics_tracking(scores: pd.DataFrame) -> pd.DataFrame:
    """Family C/E: does the predicted number of boundaries follow the true change over sessions?"""
    sub = scores[scores["family"].isin(["C_dynamics", "E_realistic"])]
    if sub.empty:
        return pd.DataFrame()
    per_session = (
        sub.groupby(["method", "family", "cell", "sequence_type", "participant_uid", "session"])
        .agg(pred=("n_pred", "mean"), true=("n_true", "mean"), f1=("f1_conv", "mean"))
        .reset_index()
    )
    rows = []
    for key, grp in per_session.groupby(["method", "family", "cell", "sequence_type"]):
        cors = []
        for _, g in grp.groupby("participant_uid"):
            if len(g) >= 3 and g["true"].std() > 0 and g["pred"].std() > 0:
                cors.append(pd.Series(g["true"].to_numpy()).corr(pd.Series(g["pred"].to_numpy()), method="spearman"))
        sess = grp.groupby("session")[["pred", "true", "f1"]].mean()
        rows.append({
            "method": key[0], "family": key[1], "cell": key[2], "sequence_type": key[3],
            "n_participants": grp["participant_uid"].nunique(),
            "mean_spearman_true_vs_pred": float(np.mean(cors)) if cors else float("nan"),
            "n_with_change": len(cors),
            **{f"s{int(s)}_pred": float(v) for s, v in sess["pred"].items()},
            **{f"s{int(s)}_true": float(v) for s, v in sess["true"].items()},
            **{f"s{int(s)}_f1": float(v) for s, v in sess["f1"].items()},
        })
    return pd.DataFrame(rows)


def group_power(scores: pd.DataFrame) -> pd.DataFrame:
    """Family D: young vs. old difference in predicted chunk count (Welch t per replicate)."""
    from scipy import stats

    d = scores[scores["family"] == "D_groups"]
    if d.empty:
        return pd.DataFrame()
    per_part = (
        d.groupby(["method", "sequence_type", "f_effect", "rep", "f_group", "participant_uid"])
        .agg(pred=("n_pred", "mean"), true=("n_true", "mean"))
        .reset_index()
    )
    rows = []
    for key, grp in per_part.groupby(["method", "sequence_type", "f_effect", "rep"]):
        old = grp[grp["f_group"] == "old"]["pred"].to_numpy()
        young = grp[grp["f_group"] == "young"]["pred"].to_numpy()
        if len(old) < 2 or len(young) < 2:
            continue
        t, p = stats.ttest_ind(old, young, equal_var=False)
        pooled = np.sqrt((old.var(ddof=1) + young.var(ddof=1)) / 2) or np.nan
        rows.append({"method": key[0], "sequence_type": key[1], "effect": key[2], "rep": key[3],
                     "diff_old_minus_young": float(old.mean() - young.mean()),
                     "true_diff": float(grp[grp["f_group"] == "old"]["true"].mean() - grp[grp["f_group"] == "young"]["true"].mean()),
                     "cohens_d": float((old.mean() - young.mean()) / pooled) if pooled else float("nan"),
                     "p_value": float(p), "significant_correct_sign": bool(p < 0.05 and old.mean() > young.mean())})
    reps = pd.DataFrame(rows)
    if reps.empty:
        return reps
    return (
        reps.groupby(["method", "sequence_type", "effect"])
        .agg(n_reps=("rep", "count"), power=("significant_correct_sign", "mean"),
             mean_diff=("diff_old_minus_young", "mean"), true_diff=("true_diff", "mean"),
             mean_cohens_d=("cohens_d", "mean"), median_p=("p_value", "median"))
        .reset_index()
    )


def _md_table(df: pd.DataFrame, cols: List[str], floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_keine Daten_\n"
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def run_evaluation(
    sim_dir: str | Path,
    results_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    families: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    sim_dir, results_dir = Path(sim_dir), Path(results_dir)
    out = Path(output_dir) if output_dir else results_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    all_scores = []
    fam_dirs = [d for d in sorted(results_dir.iterdir()) if d.is_dir() and (sim_dir / d.name / "ground_truth.csv").exists()]
    for fam_dir in fam_dirs:
        if families and fam_dir.name not in families:
            continue
        truth = load_truth(sim_dir / fam_dir.name)
        for method_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            method = method_dir.name
            if methods and method not in methods:
                continue
            if method not in METHOD_CONVENTIONS:
                print(f"skip {method}: no convention declared")
                continue
            trials = load_method_trials(method_dir)
            if trials.empty:
                continue
            scores = evaluate_method(method, trials, truth)
            all_scores.append(scores)
            print(f"[{fam_dir.name}] {method}: {len(scores)} blocks scored")
    if not all_scores:
        raise FileNotFoundError(f"No method outputs found under {results_dir}")
    scores = pd.concat(all_scores, ignore_index=True)

    cells = summarize_cells(scores)
    power = static_power(scores)
    nullpos = null_positions(scores)
    dyn = dynamics_tracking(scores)
    groups = group_power(scores)

    scores.drop(columns=["_pred_list"]).to_csv(out / "block_scores.csv.gz", index=False, compression="gzip")
    cells.to_csv(out / "evaluation_summary.csv", index=False)
    for name, df in (("static_power", power), ("null_positions", nullpos),
                     ("dynamics_tracking", dyn), ("group_power", groups)):
        if not df.empty:
            df.to_csv(out / f"{name}.csv", index=False)

    report = [
        "# Bewertung der Chunking-Methoden auf simulierten Daten",
        "",
        f"Simulation: `{sim_dir}` · Methoden-Outputs: `{results_dir}`",
        "",
        "Primäres Maß ist `f1_conv`: konventions-bewusster Abgleich (Segment-Methoden treffen eine "
        "wahre Grenz-IKI k mit Change-Point k−1 oder k). MCSE = Cluster-Bootstrap über Probanden.",
        "",
        "## Nullszenarien (Familie A): Fehlalarme",
        _md_table(cells[cells["family"] == "A_null"],
                  ["cell", "method", "sequence_type", "n_files", "false_alarm_rate", "spurious_per_null_block"]),
        "## Statische Strukturen (Familie B): Erkennbarkeit nach Effektstärke",
        _md_table(power[(power["noise"] == "mid")] if not power.empty else power,
                  ["method", "sequence_type", "channel", "effect", "recall_conv", "auc_vs_null", "d50_recall"]),
        "## Alle Zellen",
        _md_table(cells, ["family", "cell", "method", "sequence_type", "n_files", "f1_conv", "f1_conv_mcse",
                          "recall_conv", "precision_conv", "ari", "n_chunks_bias", "false_alarm_rate"]),
        "## Dynamik (Familien C/E)",
        _md_table(dyn, ["method", "cell", "sequence_type", "n_participants", "mean_spearman_true_vs_pred",
                        "s1_f1", "s3_f1", "s5_f1"]),
        "## Gruppenunterschiede (Familie D)",
        _md_table(groups, ["method", "sequence_type", "effect", "n_reps", "power", "mean_diff", "true_diff",
                           "mean_cohens_d", "median_p"]),
    ]
    (out / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    return {"output_dir": str(out), "n_blocks": len(scores), "methods": sorted(scores["method"].unique())}
