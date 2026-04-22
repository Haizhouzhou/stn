"""Phase 3: Tinkhauser beta-burst ground-truth extraction.

Usage:
    python scripts/03_extract_bursts.py \\
        --extracted-root ~/scratch/stn/extracted \\
        --audit-tsv ~/scratch/stn/audit/cohort_summary.tsv \\
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
    """Return the best available MedOff condition for a subject's summary rows."""
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


def make_figures(df: pd.DataFrame, fig_dir: Path, audit_df: pd.DataFrame) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # 1. Paired burst rate: MedOff vs MedOn (fixed_13_30 only, averaged over channels)
    bm = "fixed_13_30"
    df_bm = df[df["band_mode"] == bm].copy()

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
    ax.set_title("Group beta burst rate: MedOff vs MedOn")
    _save_fig(
        fig, fig_dir / "group_burst_rate_medoff_vs_medon.png",
        "Paired lines show mean beta burst rate (bursts/min, fixed 13–30 Hz band) "
        "per subject for the primary available condition (Rest > Hold > Move). "
        f"N = {len(paired_df)} subjects with both MedOff and MedOn available.",
    )

    # 2. Burst duration distribution
    dur_rows = []
    for subj, grp in df_bm.groupby("subject_id"):
        for _, r in grp.iterrows():
            dur_rows.append({
                "condition": r["condition"],
                "band_mode": r["band_mode"],
                "p50_duration_ms": r["p50_duration_ms"],
            })
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
        "for the fixed 13–30 Hz band. Each bar represents the density of per-channel "
        "p50 burst duration values pooled across all subjects.",
    )

    # 3. UPDRS vs long-burst fraction scatter
    updrs_df = audit_df[["subject_id", "updrs_off_total"]].copy()
    updrs_df = updrs_df.dropna(subset=["updrs_off_total"])

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
    if len(scatter_df) >= 3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(scatter_df["long_burst_fraction"], scatter_df["updrs_off_total"],
                   color="crimson", alpha=0.8, zorder=3)
        m, b = np.polyfit(scatter_df["long_burst_fraction"], scatter_df["updrs_off_total"], 1)
        xr = np.linspace(scatter_df["long_burst_fraction"].min(), scatter_df["long_burst_fraction"].max(), 50)
        ax.plot(xr, m * xr + b, "k--", lw=1.5, alpha=0.7)
        r, p = scipy.stats.pearsonr(scatter_df["long_burst_fraction"], scatter_df["updrs_off_total"])
        ax.set_xlabel("Long-burst fraction (>500 ms, MedOff)")
        ax.set_ylabel("UPDRS-III motor (off)")
        ax.set_title(f"UPDRS correlation: r = {r:.2f}, p = {p:.3f}")
        _save_fig(
            fig, fig_dir / "updrs_vs_long_burst_fraction.png",
            f"Scatter plot of mean long-burst fraction (bursts >500 ms, averaged across bipolar "
            f"channels, MedOff primary condition) vs UPDRS-III motor score (off medication). "
            f"Pearson r = {r:.2f}, p = {p:.3f}, N = {len(scatter_df)} subjects.",
        )
    else:
        logger.warning("Too few subjects with UPDRS for scatter plot (%d)", len(scatter_df))

    # 4. Strip plot of per-subject mean burst rate (MedOff Rest, fixed_13_30)
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
    ax.scatter(np.zeros(len(strip_df)) + np.random.default_rng(0).uniform(-0.1, 0.1, len(strip_df)),
               strip_df["rate_per_min"].values,
               alpha=0.8, color="steelblue", s=60, zorder=3)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Mean burst rate (bursts/min)")
    ax.set_title("Per-subject burst rate, MedOff (fixed 13–30 Hz)")
    _save_fig(
        fig, fig_dir / "per_subject_strip.png",
        "Strip plot of per-subject mean beta burst rate (bursts/min, averaged across bipolar "
        "channels) during MedOff. Each point represents one subject. The primary available "
        "condition (Rest > Hold > Move) is used.",
    )


def sanity_checks(df: pd.DataFrame, audit_df: pd.DataFrame) -> bool:
    """Run three sanity checks. Print PASS/FAIL for each. Return True if all pass."""
    bm = "fixed_13_30"
    df_bm = df[df["band_mode"] == bm].copy()
    all_pass = True

    # --- Check 1: Medication effect ---
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

    # --- Check 2: Duration range ---
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

    # --- Check 3: UPDRS correlation ---
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
        print(f"CHECK 3 [UPDRS correlation]:  FAIL — only {len(scatter_df)} subjects with UPDRS")
        all_pass = False
    else:
        r, p_two = scipy.stats.pearsonr(scatter_df["long_burst_fraction"], scatter_df["updrs_off_total"])
        p_one = p_two / 2.0  # one-sided
        r_pos = r > 0
        status = "PASS" if r_pos else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"CHECK 3 [UPDRS correlation]:  {status}  "
            f"r={r:.3f}, p(one-sided)={p_one:.4f}  (N={len(scatter_df)} subjects)"
        )

    return all_pass


def write_per_subject_table(df: pd.DataFrame, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    bm = "fixed_13_30"
    df_bm = df[df["band_mode"] == bm].copy()

    # Average across channels per subject × condition
    agg = (
        df_bm.groupby(["subject_id", "condition"])
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

    lines = ["# Per-subject burst summary (fixed 13–30 Hz band)\n"]
    lines.append(
        "| Subject | Condition | N bursts | Rate (b/min) | Mean dur (ms) | p75 dur | p90 dur | Long frac |"
    )
    lines.append("|---------|-----------|----------|-------------|---------------|---------|---------|-----------|")
    for _, r in agg.sort_values(["subject_id", "condition"]).iterrows():
        lines.append(
            f"| {r['subject_id']} | {r['condition']} | {r['n_bursts']:.0f} | "
            f"{r['burst_rate_per_min']:.1f} | {r['mean_duration_ms']:.1f} | "
            f"{r['p75_duration_ms']:.1f} | {r['p90_duration_ms']:.1f} | "
            f"{r['long_burst_fraction']:.3f} |"
        )

    (table_dir / "per_subject_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extracted-root", type=Path, required=True)
    ap.add_argument("--audit-tsv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--band-mode", choices=["fixed_13_30", "individualized", "both"], default="both")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--subject", type=str, default=None,
                    help="Run a single subject only (dry-run / debug). Skips aggregate "
                         "sanity checks and figures. Default: run all included subjects.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out.parent / "figures" / "03_bursts"
    table_dir = args.out.parent / "tables" / "03_bursts"

    audit_df = pd.read_csv(args.audit_tsv, sep="\t")
    included = audit_df[audit_df["include"] == True]["subject_id"].tolist()

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
    for res in results:
        if res and "rows" in res:
            all_rows.extend(res["rows"])
            logger.info(
                "Subject %s: threshold_condition=%s, bipolar_count=%s",
                res.get("subject_id", "?"),
                res.get("thresh_cond", "UNKNOWN"),
                res.get("n_bipolar", "?"),
            )

    if not all_rows:
        logger.error("No burst stats rows produced — something went wrong")
        return 1

    df = pd.DataFrame(all_rows)
    cohort_path = args.out / "cohort_burst_stats.tsv"
    df.to_csv(cohort_path, sep="\t", index=False)
    logger.info("Wrote %s (%d rows)", cohort_path, len(df))

    if args.subject is not None:
        logger.info("Single-subject run complete — skipping cohort sanity checks and figures")
        logger.info("Parquet outputs in: %s", args.out / args.subject)
        return 0

    logger.info("=" * 60)
    logger.info("SANITY CHECKS")
    logger.info("=" * 60)
    all_pass = sanity_checks(df, audit_df)

    logger.info("Making figures …")
    make_figures(df, fig_dir, audit_df)

    logger.info("Writing tables …")
    write_per_subject_table(df, table_dir)

    if not all_pass:
        logger.error("One or more sanity checks FAILED — do not proceed to Gate 1")
        return 2

    logger.info("All sanity checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
