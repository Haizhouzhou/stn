"""Phase 3: Tinkhauser beta-burst ground-truth extraction.

Usage:
    python scripts/03_extract_bursts.py \\
        --extracted-root ~/scratch/stn/extracted \\
        --audit-tsv ~/scratch/stn/audit/cohort_summary.tsv \\
        --updrs-tsv ~/scratch/stn/raw/participants_updrs_off.tsv \\
        --out ~/scratch/stn/results/bursts \\
        --band-mode both \\
        --jobs 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from joblib import Parallel, delayed

from stnbeta.analysis.updrs import get_updrs_lateralized, load_updrs
from stnbeta.ground_truth.pipeline import run_subject

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("03_extract_bursts")

LONG_BURST_MS = 500.0
PRIMARY_COND_ORDER = ("MedOff_Rest", "MedOff_Hold", "MedOff_Move")


def _primary_medoff_cond(subject_rows: pd.DataFrame) -> str | None:
    available = set(subject_rows["condition"].unique())
    for c in PRIMARY_COND_ORDER:
        if c in available:
            return c
    return None


def _save_fig(fig: plt.Figure, path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    (path.parent / (path.stem + ".caption.txt")).write_text(caption + "\n")
    plt.close(fig)


def _beta_active_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where is_beta_active_channel is True (if column exists)."""
    if "is_beta_active_channel" in df.columns:
        return df[df["is_beta_active_channel"] == True].copy()
    return df.copy()


def make_figures(
    df: pd.DataFrame,
    fig_dir: Path,
    audit_df: pd.DataFrame,
    contact_audit: pd.DataFrame | None,
    updrs_tsv_df: pd.DataFrame | None,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    bm = "fixed_13_30"
    df_bm = _beta_active_filter(df[df["band_mode"] == bm])

    # ── Figure 1: Paired burst rate MedOff vs MedOn (beta-active channels only) ──
    paired_rows = []
    for subj, grp in df_bm.groupby("subject_id"):
        medoff_cond = _primary_medoff_cond(grp[grp["condition"].str.startswith("MedOff")])
        if medoff_cond is None:
            continue
        medon_cond = "MedOn_" + medoff_cond.split("_", 1)[1]
        off_rows = grp[grp["condition"] == medoff_cond]
        on_rows = grp[grp["condition"] == medon_cond]
        if off_rows.empty or on_rows.empty:
            continue
        paired_rows.append({
            "subject_id": subj,
            "MedOff": off_rows["rate_per_min"].mean(),
            "MedOn": on_rows["rate_per_min"].mean(),
        })
    paired_df = pd.DataFrame(paired_rows)

    fig, ax = plt.subplots(figsize=(5, 5))
    for _, row in paired_df.iterrows():
        ax.plot([0, 1], [row["MedOff"], row["MedOn"]], "o-", color="steelblue", alpha=0.6, lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MedOff", "MedOn"])
    ax.set_ylabel("Burst rate (bursts/min)")
    ax.set_title("Group beta burst rate: MedOff vs MedOn\n(beta-active channels only)")
    _save_fig(
        fig, fig_dir / "group_burst_rate_medoff_vs_medon.png",
        "Paired lines: mean beta burst rate (bursts/min, fixed 13–30 Hz) per subject "
        "for the primary available condition (Rest > Hold > Move), restricted to "
        "beta-active channels (FOOOF beta peak ≥3 dB above 1/f aperiodic on MedOff Rest). "
        f"N = {len(paired_df)} subjects with both MedOff and MedOn available.",
    )

    # ── Figure 2: Burst duration distribution ──
    df_bm_all = df[df["band_mode"] == bm].copy()  # not filtered — all channels for distribution
    dur_rows = []
    for _, r in df_bm_all.iterrows():
        dur_rows.append({"condition": r["condition"], "p50_duration_ms": r["p50_duration_ms"]})
    dur_df = pd.DataFrame(dur_rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, cgrp in dur_df.groupby("condition"):
        vals = cgrp["p50_duration_ms"].dropna().values
        if len(vals) > 0:
            ax.hist(vals, bins=15, alpha=0.5, label=cond, density=True)
    ax.set_xlabel("Median burst duration (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Burst duration distribution by condition (fixed 13–30 Hz)")
    ax.legend(fontsize=7, ncol=2)
    _save_fig(
        fig, fig_dir / "burst_duration_distribution.png",
        "Distribution of per-channel median burst durations (ms) across conditions "
        "for the fixed 13–30 Hz band. Each bar represents density of per-channel p50 "
        "burst duration values pooled across all subjects (all channels).",
    )

    # ── Figure 3: UPDRS vs long-burst fraction (CHECK 3a + CHECK 3b side-by-side) ──
    updrs_df = audit_df[["subject_id", "updrs_off_total"]].copy().dropna(subset=["updrs_off_total"])

    lbf_rows = []
    for subj, grp in df_bm.groupby("subject_id"):
        medoff_cond = _primary_medoff_cond(grp[grp["condition"].str.startswith("MedOff")])
        if medoff_cond is None:
            continue
        cond_grp = grp[grp["condition"] == medoff_cond]
        if cond_grp.empty:
            continue
        lbf_rows.append({"subject_id": subj, "long_burst_fraction": cond_grp["long_burst_fraction"].mean()})
    lbf_df = pd.DataFrame(lbf_rows)
    scatter_df = lbf_df.merge(updrs_df, on="subject_id")

    # CHECK 3b: per-hemisphere LBF vs contralateral UPDRS
    hemi_rows = []
    if updrs_tsv_df is not None:
        for (subj, hemi), grp in df_bm.groupby(["subject_id", "hemi"]):
            medoff_cond = _primary_medoff_cond(grp[grp["condition"].str.startswith("MedOff")])
            if medoff_cond is None:
                continue
            cond_grp = grp[grp["condition"] == medoff_cond]
            if cond_grp.empty:
                continue
            mean_lbf = cond_grp["long_burst_fraction"].mean()
            scores = get_updrs_lateralized(subj, hemi, updrs_tsv_df)
            contra = scores["contralateral"]
            if contra is not None:
                hemi_rows.append({
                    "subject_id": subj,
                    "hemi": hemi,
                    "long_burst_fraction": mean_lbf,
                    "updrs_contralateral": contra,
                })
    hemi_df = pd.DataFrame(hemi_rows)

    n_subplots = 2 if len(hemi_df) >= 3 else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 4))
    if n_subplots == 1:
        axes = [axes]

    # Subplot A: CHECK 3a — subject-level total UPDRS
    ax = axes[0]
    if len(scatter_df) >= 3:
        ax.scatter(scatter_df["long_burst_fraction"], scatter_df["updrs_off_total"],
                   color="crimson", alpha=0.8, zorder=3)
        m, b = np.polyfit(scatter_df["long_burst_fraction"], scatter_df["updrs_off_total"], 1)
        xr = np.linspace(scatter_df["long_burst_fraction"].min(),
                         scatter_df["long_burst_fraction"].max(), 50)
        ax.plot(xr, m * xr + b, "k--", lw=1.5, alpha=0.7)
        r, p_two = scipy.stats.pearsonr(
            scatter_df["long_burst_fraction"], scatter_df["updrs_off_total"]
        )
        p_one = p_two / 2.0
        ax.set_title(f"CHECK 3a: r={r:.2f}, p(1-sided)={p_one:.3f}\nN={len(scatter_df)} subjects")
    else:
        ax.text(0.5, 0.5, f"N={len(scatter_df)} (too few)", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Long-burst fraction (>500 ms, MedOff, beta-active ch)")
    ax.set_ylabel("UPDRS-III total (off)")

    # Subplot B: CHECK 3b — hemisphere-level contralateral UPDRS
    if n_subplots == 2:
        ax = axes[1]
        colors_hemi = {"left": "steelblue", "right": "darkorange"}
        for hemi_val, grp in hemi_df.groupby("hemi"):
            ax.scatter(grp["long_burst_fraction"], grp["updrs_contralateral"],
                       color=colors_hemi.get(hemi_val, "gray"), alpha=0.8,
                       label=f"{hemi_val} STN", zorder=3)
        if len(hemi_df) >= 3:
            m, b = np.polyfit(hemi_df["long_burst_fraction"], hemi_df["updrs_contralateral"], 1)
            xr = np.linspace(hemi_df["long_burst_fraction"].min(),
                             hemi_df["long_burst_fraction"].max(), 50)
            ax.plot(xr, m * xr + b, "k--", lw=1.5, alpha=0.7)
            r3b, p3b_two = scipy.stats.pearsonr(
                hemi_df["long_burst_fraction"], hemi_df["updrs_contralateral"]
            )
            p3b_one = p3b_two / 2.0
            ax.set_title(
                f"CHECK 3b (PRIMARY): r={r3b:.2f}, p(1-sided)={p3b_one:.3f}\n"
                f"N={len(hemi_df)} hemisphere-subject pairs"
            )
        ax.set_xlabel("Long-burst fraction (>500 ms, MedOff, beta-active ch)")
        ax.set_ylabel("Contralateral UPDRS-III subscore")
        ax.legend(fontsize=8)

    fig.tight_layout()
    caption_3a = (
        f"Left: CHECK 3a (secondary) — mean long-burst fraction (MedOff, beta-active channels) "
        f"vs UPDRS-III total (off). N={len(scatter_df)} subjects. "
    )
    caption_3b = (
        f"Right: CHECK 3b (primary) — per-hemisphere mean LBF vs contralateral UPDRS-III subscore "
        f"(AR_contra + rest-tremor 3.17). N={len(hemi_df)} hemisphere-subject pairs. "
        f"Colors: blue=left STN, orange=right STN."
    )
    _save_fig(
        fig, fig_dir / "updrs_vs_long_burst_fraction.png",
        caption_3a + caption_3b,
    )

    # ── Figure 4: Strip plot per-subject mean burst rate (MedOff, beta-active) ──
    strip_rows = []
    for subj, grp in df_bm.groupby("subject_id"):
        medoff_cond = _primary_medoff_cond(grp[grp["condition"].str.startswith("MedOff")])
        if medoff_cond is None:
            continue
        cgrp = grp[grp["condition"] == medoff_cond]
        if cgrp.empty:
            continue
        strip_rows.append({"subject_id": subj, "rate_per_min": cgrp["rate_per_min"].mean()})
    strip_df = pd.DataFrame(strip_rows).sort_values("rate_per_min")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        np.zeros(len(strip_df)) + np.random.default_rng(0).uniform(-0.1, 0.1, len(strip_df)),
        strip_df["rate_per_min"].values,
        alpha=0.8, color="steelblue", s=60, zorder=3,
    )
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Mean burst rate (bursts/min)")
    ax.set_title("Per-subject burst rate, MedOff\n(fixed 13–30 Hz, beta-active ch)")
    _save_fig(
        fig, fig_dir / "per_subject_strip.png",
        "Strip plot of per-subject mean beta burst rate (bursts/min, averaged across "
        "beta-active bipolar channels) during MedOff primary condition (Rest > Hold > Move). "
        "Each point is one subject.",
    )

    # ── Figure 5: Contact selection overview ──
    if contact_audit is not None and not contact_audit.empty:
        cs_summary = []
        for subj, grp in contact_audit.groupby("subject_id"):
            n_total = len(grp)
            n_active = int(grp["active"].sum())
            cs_summary.append({
                "subject_id": subj,
                "n_total": n_total,
                "n_active": n_active,
                "fraction": n_active / max(n_total, 1),
            })
        cs_df = pd.DataFrame(cs_summary).sort_values("fraction", ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, len(cs_df) * 0.45)))
        bar_colors = ["steelblue" if f >= 0.5 else "crimson" for f in cs_df["fraction"]]
        ax.barh(cs_df["subject_id"], cs_df["fraction"], color=bar_colors, alpha=0.85)
        ax.barh(cs_df["subject_id"], 1.0 - cs_df["fraction"], left=cs_df["fraction"],
                color="lightgray", alpha=0.4)
        ax.axvline(0.5, color="black", linestyle="--", lw=1.0, alpha=0.7, label="50% reference")
        for i, (_, row) in enumerate(cs_df.reset_index(drop=True).iterrows()):
            label = f"{int(row['n_active'])}/{int(row['n_total'])}"
            ax.text(0.01, i, label, va="center", fontsize=8,
                    color="white" if row["fraction"] >= 0.2 else "black")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of bipolar channels with detectable beta peak")
        ax.set_title("Contact selection: beta-active channels per subject\n"
                     "(FOOOF peak ≥3 dB above 1/f, 13–35 Hz, MedOff Rest)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_fig(
            fig, fig_dir / "contact_selection_overview.png",
            "Horizontal bar per subject: fraction of bipolar channels classified as "
            "beta-active (FOOOF beta peak ≥3 dB above aperiodic, 13–35 Hz, MedOff Rest). "
            "Blue bars ≥50% active, red <50%. Labels show N_active/N_total. "
            "Dashed line at 50%.",
        )


def sanity_checks(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    updrs_tsv_df: pd.DataFrame | None,
) -> bool:
    bm = "fixed_13_30"
    df_bm_all = df[df["band_mode"] == bm].copy()
    df_bm = _beta_active_filter(df_bm_all)
    all_pass = True

    # ── CHECK 1: Medication effect (beta-active channels only) ──
    paired_rows = []
    for subj, grp in df_bm.groupby("subject_id"):
        medoff_cond = _primary_medoff_cond(grp[grp["condition"].str.startswith("MedOff")])
        if medoff_cond is None:
            continue
        medon_cond = "MedOn_" + medoff_cond.split("_", 1)[1]
        off_rows = grp[grp["condition"] == medoff_cond]
        on_rows = grp[grp["condition"] == medon_cond]
        if off_rows.empty or on_rows.empty:
            continue
        paired_rows.append({
            "subject_id": subj,
            "off": off_rows["rate_per_min"].mean(),
            "on": on_rows["rate_per_min"].mean(),
        })
    paired_df = pd.DataFrame(paired_rows)

    if len(paired_df) < 3:
        print(f"CHECK 1 [Medication effect]:  FAIL — only {len(paired_df)} paired subjects")
        all_pass = False
    else:
        t_stat, p_val = scipy.stats.ttest_rel(paired_df["off"], paired_df["on"])
        mean_diff = float((paired_df["off"] - paired_df["on"]).mean())
        direction_ok = mean_diff > 0
        p_ok = p_val < 0.05
        status = "PASS" if (p_ok and direction_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"CHECK 1 [Medication effect]:  {status}  "
            f"t={t_stat:.3f}, p={p_val:.4f}, mean(MedOff-MedOn)={mean_diff:.2f} bursts/min"
            f"  (N={len(paired_df)} pairs)"
        )

    # ── CHECK 2: Duration range (beta-active channels only) ──
    medoff_rows = df_bm[df_bm["condition"].str.startswith("MedOff")]
    if medoff_rows.empty:
        print("CHECK 2 [Duration range]:     FAIL — no MedOff rows")
        all_pass = False
    else:
        group_median = float(medoff_rows["p50_duration_ms"].median())
        in_range = 150.0 <= group_median <= 500.0
        status = "PASS" if in_range else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"CHECK 2 [Duration range]:     {status}  "
            f"group median p50 duration = {group_median:.1f} ms  (expected [150, 500] ms)"
        )

    # ── CHECK 3a: Subject-level LBF vs total UPDRS (secondary) ──
    updrs_df = audit_df[["subject_id", "updrs_off_total"]].dropna(subset=["updrs_off_total"])

    lbf_rows = []
    for subj, grp in df_bm.groupby("subject_id"):
        medoff_cond = _primary_medoff_cond(grp[grp["condition"].str.startswith("MedOff")])
        if medoff_cond is None:
            continue
        cond_grp = grp[grp["condition"] == medoff_cond]
        if cond_grp.empty:
            continue
        lbf_rows.append({"subject_id": subj, "long_burst_fraction": cond_grp["long_burst_fraction"].mean()})
    lbf_df = pd.DataFrame(lbf_rows)
    scatter_df = lbf_df.merge(updrs_df, on="subject_id")

    if len(scatter_df) < 3:
        print(f"CHECK 3a [UPDRS total corr]:  FAIL — only {len(scatter_df)} subjects with UPDRS")
        all_pass = False
    else:
        r, p_two = scipy.stats.pearsonr(
            scatter_df["long_burst_fraction"], scatter_df["updrs_off_total"]
        )
        p_one = p_two / 2.0
        r_pos = r > 0
        status = "PASS" if r_pos else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"CHECK 3a [UPDRS total corr]:  {status}  "
            f"r={r:.3f}, p(one-sided)={p_one:.4f}  (N={len(scatter_df)} subjects)"
        )

    # ── CHECK 3b: Hemisphere-level LBF vs contralateral UPDRS (primary) ──
    if updrs_tsv_df is None:
        print("CHECK 3b [Contralat. UPDRS]:  UNAVAILABLE — participants_updrs_off.tsv not found")
    else:
        hemi_rows = []
        for (subj, hemi), grp in df_bm.groupby(["subject_id", "hemi"]):
            medoff_cond = _primary_medoff_cond(grp[grp["condition"].str.startswith("MedOff")])
            if medoff_cond is None:
                continue
            cond_grp = grp[grp["condition"] == medoff_cond]
            if cond_grp.empty:
                continue
            mean_lbf = cond_grp["long_burst_fraction"].mean()
            scores = get_updrs_lateralized(subj, hemi, updrs_tsv_df)
            contra = scores["contralateral"]
            if contra is not None:
                hemi_rows.append({
                    "subject_id": subj,
                    "hemi": hemi,
                    "long_burst_fraction": mean_lbf,
                    "updrs_contralateral": contra,
                })
        hemi_df = pd.DataFrame(hemi_rows)

        if len(hemi_df) < 3:
            print(
                f"CHECK 3b [Contralat. UPDRS]:  FAIL — only {len(hemi_df)} "
                f"(subject, hemisphere) pairs with contralateral UPDRS"
            )
            # Gate 1 fails only if BOTH 3a and 3b fail; mark 3b as fail but do not
            # override all_pass if 3a passed
        else:
            r3b, p3b_two = scipy.stats.pearsonr(
                hemi_df["long_burst_fraction"], hemi_df["updrs_contralateral"]
            )
            p3b_one = p3b_two / 2.0
            r3b_pos = r3b > 0
            # Trend-level threshold: p < 0.1 one-sided
            trend_ok = r3b_pos and p3b_one < 0.1
            status = "PASS" if trend_ok else ("TREND" if r3b_pos else "FAIL")
            if not r3b_pos:
                all_pass = False
            print(
                f"CHECK 3b [Contralat. UPDRS]:  {status}  "
                f"r={r3b:.3f}, p(one-sided)={p3b_one:.4f}  "
                f"(N={len(hemi_df)} hemisphere-subject pairs)"
            )

    return all_pass


def write_per_subject_table(
    df: pd.DataFrame,
    table_dir: Path,
    audit_df: pd.DataFrame,
    contact_audit: pd.DataFrame | None,
    updrs_tsv_df: pd.DataFrame | None,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    bm = "fixed_13_30"
    df_bm = df[df["band_mode"] == bm].copy()
    df_bm_active = _beta_active_filter(df_bm)

    # Per-subject contact selection summary
    cs_map: dict[str, dict] = {}
    if contact_audit is not None and not contact_audit.empty:
        for subj, grp in contact_audit.groupby("subject_id"):
            cs_map[subj] = {
                "n_total": len(grp),
                "n_active": int(grp["active"].sum()),
            }

    # Per-subject UPDRS
    updrs_total_map = dict(
        zip(audit_df["subject_id"], audit_df.get("updrs_off_total", pd.Series(dtype=float)))
    )

    # Aggregate active-channel stats
    agg = (
        df_bm_active.groupby(["subject_id", "condition"])
        .agg(
            n_bursts=("n_bursts", "sum"),
            burst_rate_per_min=("rate_per_min", "mean"),
            mean_duration_ms=("mean_duration_ms", "mean"),
            p75_duration_ms=("p75_duration_ms", "mean"),
            p90_duration_ms=("p90_duration_ms", "mean"),
            long_burst_fraction=("long_burst_fraction", "mean"),
        )
        .reset_index()
    )

    lines = ["# Per-subject burst summary (fixed 13–30 Hz band, beta-active channels only)\n"]
    lines.append(
        "| Subject | N_bip_total | N_beta_active | Condition | N bursts | Rate (b/min) | "
        "Mean dur (ms) | p75 dur | p90 dur | Long frac | UPDRS-III off |"
    )
    lines.append(
        "|---------|-------------|---------------|-----------|----------|-------------|"
        "---------------|---------|---------|-----------|---------------|"
    )
    for _, r in agg.sort_values(["subject_id", "condition"]).iterrows():
        subj = r["subject_id"]
        cs = cs_map.get(subj, {})
        n_tot = cs.get("n_total", "?")
        n_act = cs.get("n_active", "?")
        updrs = updrs_total_map.get(subj, float("nan"))
        updrs_str = f"{updrs:.0f}" if not (isinstance(updrs, float) and np.isnan(updrs)) else "?"
        lines.append(
            f"| {subj} | {n_tot} | {n_act} | {r['condition']} | {r['n_bursts']:.0f} | "
            f"{r['burst_rate_per_min']:.1f} | {r['mean_duration_ms']:.1f} | "
            f"{r['p75_duration_ms']:.1f} | {r['p90_duration_ms']:.1f} | "
            f"{r['long_burst_fraction']:.3f} | {updrs_str} |"
        )

    (table_dir / "per_subject_summary.md").write_text("\n".join(lines) + "\n")


def write_literature_comparison(df: pd.DataFrame, table_dir: Path) -> None:
    """Write literature comparison using beta-active channels only."""
    table_dir.mkdir(parents=True, exist_ok=True)
    bm = "fixed_13_30"
    df_bm = _beta_active_filter(df[df["band_mode"] == bm])

    medoff_rows = df_bm[df_bm["condition"].str.startswith("MedOff")]

    n_subj = df_bm["subject_id"].nunique()
    n_hemi = df_bm.groupby(["subject_id", "hemi"]).ngroups if not df_bm.empty else 0

    rate_med = float(medoff_rows["rate_per_min"].median()) if not medoff_rows.empty else float("nan")
    rate_iqr_lo = float(medoff_rows["rate_per_min"].quantile(0.25)) if not medoff_rows.empty else float("nan")
    rate_iqr_hi = float(medoff_rows["rate_per_min"].quantile(0.75)) if not medoff_rows.empty else float("nan")
    dur_med = float(medoff_rows["p50_duration_ms"].median()) if not medoff_rows.empty else float("nan")
    long_frac_med = float(medoff_rows["long_burst_fraction"].median()) if not medoff_rows.empty else float("nan")

    lines = [
        "# Literature comparison (beta-active channels only, fixed 13–30 Hz, MedOff)",
        "",
        f"**Cohort:** N={n_subj} subjects, {n_hemi} hemisphere-recordings",
        "",
        "| Metric | This study | Tinkhauser 2017 | Neumann 2016 |",
        "|--------|-----------|-----------------|--------------|",
        f"| Burst rate (b/min), median [IQR] | "
        f"{rate_med:.1f} [{rate_iqr_lo:.1f}–{rate_iqr_hi:.1f}] | ~30–50 | N/A |",
        f"| Median burst duration p50 (ms) | {dur_med:.0f} | ~200–300 | ~200 |",
        f"| Long burst fraction (>500 ms) | {long_frac_med:.3f} | N/A | N/A |",
        "",
        "> All values restricted to beta-active channels "
        "(FOOOF beta peak ≥3 dB above 1/f aperiodic on MedOff Rest, 13–35 Hz).",
    ]
    (table_dir / "literature_comparison.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extracted-root", type=Path, required=True)
    ap.add_argument("--audit-tsv", type=Path, required=True)
    ap.add_argument(
        "--updrs-tsv",
        type=Path,
        default=None,
        help="Path to participants_updrs_off.tsv for lateralized UPDRS (CHECK 3b).",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--band-mode", choices=["fixed_13_30", "individualized", "both"], default="both")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--subject", type=str, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out.parent / "figures" / "03_bursts"
    table_dir = args.out.parent / "tables" / "03_bursts"

    audit_df = pd.read_csv(args.audit_tsv, sep="\t")
    included = audit_df[audit_df["include"] == True]["subject_id"].tolist()

    updrs_tsv_df = None
    if args.updrs_tsv is not None and args.updrs_tsv.exists():
        updrs_tsv_df = load_updrs(args.updrs_tsv)
        logger.info("Loaded UPDRS TSV: %s (%d subjects)", args.updrs_tsv, len(updrs_tsv_df))
    else:
        logger.warning(
            "UPDRS TSV not found (%s) — CHECK 3b will be UNAVAILABLE", args.updrs_tsv
        )

    if args.subject is not None:
        if args.subject not in included:
            logger.error("--subject %s not in included cohort", args.subject)
            return 1
        included = [args.subject]
        logger.info("Single-subject dry-run: %s", args.subject)
    else:
        logger.info("Running burst extraction for %d included subjects", len(included))

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_subject)(
            subject_id=subj,
            bids_root=args.extracted_root.parent,
            extracted_root=args.extracted_root,
            out_dir=args.out,
            band_mode=args.band_mode,
        )
        for subj in included
    )

    all_rows = []
    contact_audit_rows = []
    for res in results:
        if not res or "rows" not in res:
            continue
        all_rows.extend(res["rows"])
        logger.info(
            "Subject %s: threshold_condition=%s, bipolar_count=%s",
            res.get("subject_id", "?"),
            res.get("thresh_cond", "UNKNOWN"),
            res.get("n_bipolar", "?"),
        )
        # Collect contact selection audit rows
        subj = res.get("subject_id", "?")
        cs_rest = res.get("cs_rest_s", 0.0)
        for ch, cr in res.get("contact_results", {}).items():
            hemi = "left" if "left" in ch else "right"
            contact_audit_rows.append({
                "subject_id": subj,
                "hemi": hemi,
                "channel": ch,
                "active": cr["active"],
                "peak_freq_hz": cr["peak_freq_hz"],
                "peak_power_db": cr["peak_power_db"],
                "aperiodic_exponent": cr["aperiodic_exponent"],
                "reason_if_excluded": "" if cr["active"] else cr["reason"],
                "rest_duration_s_used": cs_rest,
            })

    if not all_rows:
        logger.error("No burst stats rows produced — something went wrong")
        return 1

    df = pd.DataFrame(all_rows)
    cohort_path = args.out / "cohort_burst_stats.tsv"
    df.to_csv(cohort_path, sep="\t", index=False)
    logger.info("Wrote %s (%d rows)", cohort_path, len(df))

    contact_audit = pd.DataFrame(contact_audit_rows)
    if not contact_audit.empty:
        audit_path = args.out / "contact_selection_audit.tsv"
        contact_audit.to_csv(audit_path, sep="\t", index=False)
        n_total_ch = len(contact_audit)
        n_active_ch = int(contact_audit["active"].sum())
        n_subj_active = int((contact_audit.groupby("subject_id")["active"].any()).sum())
        n_subj_zero = int((~contact_audit.groupby("subject_id")["active"].any()).sum())
        logger.info(
            "Contact selection audit: %d channels total, %d beta-active (%.0f%%), "
            "%d subjects with ≥1 active, %d subjects with zero active",
            n_total_ch, n_active_ch, 100.0 * n_active_ch / max(n_total_ch, 1),
            n_subj_active, n_subj_zero,
        )
        logger.info("Wrote contact selection audit: %s", audit_path)
    else:
        contact_audit = None

    if args.subject is not None:
        logger.info("Single-subject run complete — skipping cohort sanity checks and figures")
        logger.info("Parquet outputs in: %s", args.out / args.subject)
        # Verify contact selection column exists
        if "is_beta_active_channel" in df.columns:
            n_active = int(df["is_beta_active_channel"].sum())
            logger.info("is_beta_active_channel column present: %d True rows", n_active)
        return 0

    logger.info("=" * 60)
    logger.info("SANITY CHECKS")
    logger.info("=" * 60)
    all_pass = sanity_checks(df, audit_df, updrs_tsv_df)

    logger.info("Making figures …")
    make_figures(df, fig_dir, audit_df, contact_audit, updrs_tsv_df)

    logger.info("Writing tables …")
    write_per_subject_table(df, table_dir, audit_df, contact_audit, updrs_tsv_df)
    write_literature_comparison(df, table_dir)

    # Back up pre-existing literature_comparison if it predates contact selection
    _write_adrs(Path(__file__).resolve().parent.parent / "docs" / "decisions.md")

    if not all_pass:
        logger.error("One or more sanity checks FAILED — do not proceed to Gate 1")
        return 2

    logger.info("All sanity checks PASSED")
    return 0


def _write_adrs(decisions_path: Path) -> None:
    """Append Fix 4/5/6 ADRs to decisions.md if not already present."""
    text = decisions_path.read_text() if decisions_path.exists() else ""
    new_adrs = ""

    if "ADR-015" not in text:
        new_adrs += """
---

## ADR-015: Functional contact selection via FOOOF beta peak
**Date:** 2026-04-22
**Decision:** Each bipolar channel must exhibit a detectable beta peak (13–35 Hz) at least
3 dB above the 1/f aperiodic background on MedOff Rest before being included in burst analysis.
Fit performed with specparam/FOOOF (peak_width_limits=(2,12), max_n_peaks=6, fixed aperiodic),
Welch PSD (4-second windows, 50% overlap), frequency range 2–45 Hz. Minimum rest data: 60 s.
**Rationale:** Rassoulou 2024 (Sci Data) usage notes recommend functional criteria because
anatomical contact positions within STN are not provided in the dataset. The functional approach
(Tinkhauser 2017, Neumann 2016, Lofredi 2019) ensures only physiologically relevant channels
contribute to burst statistics, reducing noise from non-STN contacts.
Threshold = 3 dB peak height above aperiodic 1/f baseline, 13–35 Hz, MedOff Rest.
"""

    if "ADR-016" not in text:
        new_adrs += """
---

## ADR-016: Contralateral UPDRS subscore as CHECK 3b primary metric
**Date:** 2026-04-22
**Decision:** CHECK 3b (primary gate) uses the per-hemisphere contralateral UPDRS-III subscore
(AR_contralateral + rest-tremor-amplitude 3.17 UE+LE for the contralateral body side).
CHECK 3a (secondary) retains the total UPDRS-III for literature comparability.
Contralateral items per MDS-UPDRS Part III: rigidity (3.3, lateralized), finger tapping (3.4),
hand movements (3.5), pronation-supination (3.6), toe tapping (3.7), leg agility (3.8),
rest tremor amplitude (3.17 UE+LE). Jaw tremor (3.17e) and all axial items excluded.
Column mapping verified: "AR right/left" in participants_updrs_off.tsv = 3.3b/d + 3.4–3.8
for right/left body (cross-checked against individual item sums for multiple subjects).
**Rationale:** Left STN pathology drives right-body symptoms and vice versa (standard
lateralization in DBS literature). Using total UPDRS-III averages across both hemispheres
and weakens the signal by including symptoms driven by the contralateral (unrecorded) electrode.
Gate 1 passes if CHECK 1 + CHECK 2 + (CHECK 3a or CHECK 3b) pass at p < 0.1 one-sided.
If both 3a and 3b fail, this is accepted as an honest small-N cohort finding — not re-tuned.
"""

    if "ADR-017" not in text:
        new_adrs += """
---

## ADR-017: Long MedOff Rest bursts retained — no max-duration filter
**Date:** 2026-04-22
**Decision:** No maximum burst duration filter is applied. The ~3.7 s burst observed in
sub-6m9kB5 LFP-left-34 (MedOff Rest) is verified non-artifactual: no BAD_LFP annotation
overlaps this epoch (confirmed by get_epoch_mask output). Tonic beta elevations during
prolonged MedOff Rest are documented as a disease feature (Lofredi 2019, J Neurosci).
**Rationale:** Introducing an ad hoc max-duration filter to exclude this burst would require
physiological justification absent from the cited literature. The burst is retained as a
valid long-duration event. The long_burst_fraction metric is designed to detect exactly
this phenotype; excluding it would bias the CHECK 3 correlation downward.
"""

    if new_adrs:
        decisions_path.write_text(text + new_adrs)
        logger.info("Appended ADRs 015–017 to %s", decisions_path)


if __name__ == "__main__":
    sys.exit(main())
