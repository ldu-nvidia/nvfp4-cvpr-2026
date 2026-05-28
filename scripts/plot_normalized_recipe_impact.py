#!/usr/bin/env python3
"""
QAT Sensitivity Plot
====================
Shows each recipe's AUPRC as % of Baseline (per-seed normalization).
This removes absolute AUPRC differences and isolates the recipe effect.

Usage: python plot_normalized_recipe_impact.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats as sp

D = Path(__file__).resolve().parent
raw_default = D / "comparison_plots" / "multi_seed_raw_results.csv"
raw_15m = D / "comparison_plots" / "multi_seed_raw_results_15m.csv"
raw_path = raw_15m if raw_15m.exists() else raw_default
df = pd.read_csv(raw_path)
print(f"Loaded {len(df)} entries")

# Normalize: for each (arch, size, seed), divide AUPRC by that seed's Baseline
records = []
for (arch, size, seed), grp in df.groupby(["arch", "model_size", "seed"]):
    baseline_val = grp[grp["recipe_id"] == 0]["auprc"].values
    if len(baseline_val) == 0:
        continue
    bv = baseline_val[0]
    for _, row in grp.iterrows():
        records.append({
            "arch": arch, "model_size": size, "seed": seed,
            "recipe_id": row["recipe_id"],
            "recipe_name": row["recipe_name"],
            "auprc_pct": (row["auprc"] / bv) * 100 if bv > 0 else 100,
        })
ndf = pd.DataFrame(records)

RO = [2, 7, 3, 6, 5, 4, 1]
RL = {2: "Fwd-Only", 7: "Fwd+RHT", 3: "Chain Rule",
      6: "SR Only", 5: "2D+RHT+SR", 1: "NVFP4 Full", 4: "2D+RHT"}
RC = {
    1: "#e74c3c",   # NVFP4 Full
    4: "#2ecc71",   # 2D+RHT
    5: "#f39c12",   # 2D+RHT+SR
    6: "#1abc9c",   # SR Only
    2: "#2c3e50",  # Fwd-Only (dark navy)
    3: "#e67e22",  # Chain Rule
    7: "#8e44ad",  # Fwd+RHT (purple) - make distinct from Fwd-Only
}
AL = {"swin": "Swin", "cnn": "CNN"}

YLIM = {
    "swin": (75, 110),
    "cnn": (75, 110),
}

SL = {"matched_500k": "500K", "matched_4m": "4M", "matched_15m": "15M"}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

panels = [
    ("swin", "matched_500k", axes[0, 0]),
    ("swin", "matched_4m", axes[0, 1]),
    ("swin", "matched_15m", axes[0, 2]),
    ("cnn", "matched_500k", axes[1, 0]),
    ("cnn", "matched_4m", axes[1, 1]),
    ("cnn", "matched_15m", axes[1, 2]),
]

rng = np.random.default_rng(42)

for arch, size, ax in panels:
    sub = ndf[(ndf["arch"] == arch) & (ndf["model_size"] == size)]
    is_bottom = (arch == "cnn")
    is_left = (size == "matched_500k")

    for i, rid in enumerate(RO):
        vals = sub[sub["recipe_id"] == rid]["auprc_pct"].values
        if len(vals) == 0:
            continue
        c = RC.get(rid, "#888")
        m = np.mean(vals)
        s = np.std(vals, ddof=1)
        n = len(vals)
        ci = sp.t.ppf(0.975, n - 1) * s / np.sqrt(n)

        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(i + jitter, vals, s=30, color=c, alpha=0.65,
                   edgecolors="white", linewidths=0.4, zorder=5)

        ax.plot([i - 0.28, i + 0.28], [m, m], color=c, lw=2.5, zorder=6)
        ax.plot([i, i], [m - ci, m + ci], color=c, lw=2, zorder=6)
        ax.plot([i - 0.1, i + 0.1], [m - ci, m - ci], color=c, lw=1.5, zorder=6)
        ax.plot([i - 0.1, i + 0.1], [m + ci, m + ci], color=c, lw=1.5, zorder=6)

    ax.axhline(100, color="#3498db", linestyle="--", linewidth=2.75, alpha=0.85, zorder=1)

    ylim_lo, ylim_hi = YLIM[arch]
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_xticks(range(len(RO)))

    HIGHLIGHT_RECIPES = {6, 5, 4}
    HIGHLIGHT_COLOR = "#c0392b"
    if is_bottom:
        ax.set_xticklabels([RL[r] for r in RO], fontsize=13,
                           fontweight="bold", rotation=30, ha="right")
        for i, rid in enumerate(RO):
            if rid in HIGHLIGHT_RECIPES:
                ax.get_xticklabels()[i].set_color(HIGHLIGHT_COLOR)
    else:
        ax.set_xticklabels([])

    ax.set_ylabel("")

    ax.set_title(f"{AL[arch]} | {SL[size]}", fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14, labelleft=is_left)
    if not is_left:
        ax.set_yticklabels([])
    else:
        for lab in ax.get_yticklabels():
            lab.set_fontweight("bold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    HIGHLIGHT_IDX = [i for i, r in enumerate(RO) if r in {6, 5, 4}]
    for hi in HIGHLIGHT_IDX:
        ax.axvspan(hi - 0.45, hi + 0.45, alpha=0.07, color="#c0392b", zorder=0)

fig.supylabel("AUPRC (% of Baseline)", fontsize=16, fontweight="bold", x=0.045)

n_seeds = ndf["seed"].nunique()
fig.text(0.99, 0.005, f"* Each dot represents 1 of {n_seeds} random weight initialization seeds; bars show mean ± 95% CI.",
         ha="right", va="bottom", fontsize=12, color="black")

plt.tight_layout(rect=[0.04, 0.02, 1, 1])
fig.subplots_adjust(wspace=0.06)
out = D / "comparison_plots" / "qat_sensitivity_15m.png"
fig.savefig(out, dpi=900, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
fig.savefig(D / "comparison_plots" / "qat_sensitivity_15m.pdf",
            bbox_inches="tight", facecolor="white")
plt.close()
