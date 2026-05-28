#!/usr/bin/env python3
"""
Statistical Analysis of Multi-Seed QAT Recipe Results
=====================================================

Loads telemetry from all seed runs and performs:
1. Mean ± std AUPRC per recipe
2. 95% confidence intervals
3. Pairwise statistical tests (Wilcoxon signed-rank)
4. Rank stability analysis (Kendall's W)
5. Effect size (Cohen's d)

Usage:
    python analyze_multi_seed.py                    # Analyze all
    python analyze_multi_seed.py --arch swin        # Swin only
    python analyze_multi_seed.py --plot             # Generate plots
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Path setup
MAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN_DIR / "common"))

try:
    from utils import RECIPE_NAMES, RECIPE_COLORS
except Exception:
    # Keep analysis runnable in minimal environments (e.g., no OpenCV).
    RECIPE_NAMES = {
        0: "Baseline",
        1: "NVFP4 Full",
        60031: "NVFP4 Full (Skip Bottleneck)",
        2: "Fwd-Only",
        3: "Chain Rule",
        4: "2D+RHT",
        5: "2D+RHT+SR",
        6: "SR Only",
        7: "Fwd+RHT",
    }
    RECIPE_COLORS = {
        0: "#3498db",
        1: "#e74c3c",
        60031: "#c0392b",
        4: "#2ecc71",
        5: "#f39c12",
        6: "#1abc9c",
        2: "#34495e",
        3: "#e67e22",
        7: "#16a085",
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
MODEL_SIZES = ["matched_500k", "matched_4m", "matched_15m"]
RECIPES = [0, 1, 2, 3, 4, 5, 6, 7]

ARCH_CKPT_DIRS = {
    "swin": MAIN_DIR / "swin" / "swin_ckpts",
    "cnn": MAIN_DIR / "cnn" / "cnn_ckpts",
}


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def load_metrics_json(seed_dir: Path, seed: int) -> List[Dict]:
    """Load per-recipe metrics JSON files and deduplicate by (seed,size,recipe)."""
    metrics_dir = seed_dir / "metrics"
    if not metrics_dir.exists():
        return []

    chosen: Dict[Tuple[int, str, int], Tuple[float, Dict]] = {}
    for jf in sorted(metrics_dir.glob("metrics_r*_*.json")):
        try:
            with open(jf, "r") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Failed to read {jf.name}: {e}")
            continue

        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}

        model_size = cfg.get("model_size") or cfg.get("full_config", {}).get("model_size")
        recipe_id = cfg.get("recipe_id")
        if model_size is None or recipe_id is None:
            continue

        ts = _parse_timestamp(cfg.get("timestamp"))
        sort_key = ts.timestamp() if ts is not None else jf.stat().st_mtime

        key = (seed, str(model_size), int(recipe_id))
        prev = chosen.get(key)
        if prev is None or sort_key > prev[0]:
            chosen[key] = (sort_key, {
                "seed": seed,
                "model_size": str(model_size),
                "recipe_id": int(recipe_id),
                "recipe_name": cfg.get("recipe_name") or RECIPE_NAMES.get(int(recipe_id), str(recipe_id)),
                "auprc": m.get("auprc", None),
                "recall": m.get("recall", None),
                "f2_score": m.get("f2_score", None),
                "precision": m.get("precision", None),
                # The JSON uses "dice" for test-set Dice; keep legacy CSV column name.
                "best_dice": m.get("dice", m.get("best_dice", None)),
                "actual_epochs": m.get("actual_epochs", None),
            })

    return [v[1] for v in chosen.values()]


def collect_all_results(arch: str) -> pd.DataFrame:
    """Collect AUPRC from all seed/size/recipe combinations into a DataFrame."""
    ckpt_base = ARCH_CKPT_DIRS[arch]
    rows = []
    
    for seed in SEEDS:
        seed_dir = ckpt_base / f"seed_{seed}"
        if not seed_dir.exists():
            continue

        seed_records = load_metrics_json(seed_dir, seed)

        for data in seed_records:
            recipe_id = data.get("recipe_id")
            model_size = data.get("model_size", "unknown")
            auprc = data.get("auprc", None)
            recall = data.get("recall", None)
            f2 = data.get("f2_score", None)
            precision = data.get("precision", None)
            best_dice = data.get("best_dice", None)
            actual_epochs = data.get("actual_epochs", None)
            
            rows.append({
                "arch": arch,
                "seed": seed,
                "model_size": model_size,
                "recipe_id": recipe_id,
                "recipe_name": RECIPE_NAMES.get(recipe_id, str(recipe_id)),
                "auprc": auprc,
                "recall": recall,
                "f2_score": f2,
                "precision": precision,
                "best_dice": best_dice,
                "actual_epochs": actual_epochs,
            })
    
    return pd.DataFrame(rows)


def compute_summary_stats(df: pd.DataFrame, metric: str = "auprc") -> pd.DataFrame:
    """Compute mean, std, CI for each (arch, size, recipe) group."""
    grouped = df.groupby(["arch", "model_size", "recipe_id", "recipe_name"])[metric]
    
    summary = grouped.agg(["mean", "std", "count", "min", "max"]).reset_index()
    
    # 95% confidence interval
    summary["ci_95"] = summary.apply(
        lambda row: stats.t.ppf(0.975, row["count"] - 1) * row["std"] / np.sqrt(row["count"])
        if row["count"] > 1 else np.nan, axis=1
    )
    summary["ci_lower"] = summary["mean"] - summary["ci_95"]
    summary["ci_upper"] = summary["mean"] + summary["ci_95"]
    
    return summary


def pairwise_tests(df: pd.DataFrame, arch: str, model_size: str, metric: str = "auprc") -> pd.DataFrame:
    """Run pairwise Wilcoxon signed-rank tests between all recipe pairs."""
    subset = df[(df["arch"] == arch) & (df["model_size"] == model_size)]
    
    # Pivot: rows=seeds, columns=recipes, values=metric
    pivot = subset.pivot_table(index="seed", columns="recipe_id", values=metric)
    
    recipe_ids = sorted(pivot.columns.tolist())
    results = []
    
    for r1, r2 in combinations(recipe_ids, 2):
        vals1 = pivot[r1].dropna()
        vals2 = pivot[r2].dropna()
        
        # Align by seed
        common_seeds = vals1.index.intersection(vals2.index)
        if len(common_seeds) < 3:
            continue
        
        v1 = vals1[common_seeds].values
        v2 = vals2[common_seeds].values
        
        # Wilcoxon signed-rank test
        try:
            stat, p_value = stats.wilcoxon(v1, v2)
        except ValueError:
            stat, p_value = np.nan, np.nan
        
        # Cohen's d (paired effect size, using sample std with ddof=1)
        diff = v1 - v2
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        
        results.append({
            "arch": arch,
            "model_size": model_size,
            "recipe_1": r1,
            "recipe_1_name": RECIPE_NAMES.get(r1, str(r1)),
            "recipe_2": r2,
            "recipe_2_name": RECIPE_NAMES.get(r2, str(r2)),
            "mean_1": np.mean(v1),
            "mean_2": np.mean(v2),
            "mean_diff": np.mean(diff),
            "wilcoxon_stat": stat,
            "p_value": p_value,
            "significant_005": p_value < 0.05 if not np.isnan(p_value) else False,
            "significant_001": p_value < 0.01 if not np.isnan(p_value) else False,
            "cohens_d": cohens_d,
            "n_seeds": len(common_seeds),
        })
    
    results_df = pd.DataFrame(results)
    
    # Holm-Bonferroni multiple comparison correction
    if not results_df.empty and results_df["p_value"].notna().any():
        valid = results_df["p_value"].notna()
        p_vals = results_df.loc[valid, "p_value"].values
        n_tests = len(p_vals)
        
        # Sort p-values ascending
        sorted_idx = np.argsort(p_vals)
        sorted_pvals = p_vals[sorted_idx]
        
        # Holm adjustment: p_(i) * (m - i), where i is 0-indexed rank
        adjusted = sorted_pvals * (n_tests - np.arange(n_tests))
        
        # Enforce non-decreasing (Holm step-down monotonicity)
        adjusted = np.maximum.accumulate(adjusted)
        
        # Clip to [0, 1]
        adjusted = np.clip(adjusted, 0, 1)
        
        # Map back to original test order
        corrected = np.empty(n_tests)
        corrected[sorted_idx] = adjusted
        
        results_df.loc[valid, "p_corrected"] = corrected
        results_df["significant_005_corrected"] = results_df["p_corrected"] < 0.05
        results_df["significant_001_corrected"] = results_df["p_corrected"] < 0.01
    
    return results_df


def rank_stability(df: pd.DataFrame, arch: str, model_size: str, metric: str = "auprc") -> Dict:
    """Compute rank stability across seeds using Kendall's W."""
    subset = df[(df["arch"] == arch) & (df["model_size"] == model_size)]
    pivot = subset.pivot_table(index="seed", columns="recipe_id", values=metric)
    
    if pivot.shape[0] < 3:
        return {"kendalls_w": np.nan, "rank_table": None}
    
    # Rank within each seed (higher AUPRC = rank 1)
    ranks = pivot.rank(axis=1, ascending=False)
    
    # Kendall's W
    n_seeds = ranks.shape[0]
    n_recipes = ranks.shape[1]
    rank_sums = ranks.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    ss = ((rank_sums - mean_rank_sum) ** 2).sum()
    w = 12 * ss / (n_seeds**2 * (n_recipes**3 - n_recipes))
    
    return {
        "kendalls_w": w,
        "mean_ranks": ranks.mean().to_dict(),
        "rank_std": ranks.std().to_dict(),
        "rank_table": ranks,
    }


def plot_multi_seed_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    """Generate multi-seed comparison plots with error bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for arch in df["arch"].unique():
        for model_size in df["model_size"].unique():
            sub = summary[(summary["arch"] == arch) & (summary["model_size"] == model_size)]
            if sub.empty:
                continue
            
            sub = sub.sort_values("mean", ascending=False)
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            x = range(len(sub))
            colors = [RECIPE_COLORS.get(rid, "#888888") for rid in sub["recipe_id"]]
            
            bars = ax.bar(x, sub["mean"], yerr=sub["ci_95"], capsize=5,
                         color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
            
            ax.set_xticks(x)
            ax.set_xticklabels([RECIPE_NAMES.get(r, str(r)) for r in sub["recipe_id"]],
                              rotation=45, ha="right", fontsize=10)
            ax.set_ylabel("AUPRC (mean ± 95% CI)", fontsize=13)
            ax.set_title(f"{arch.upper()} | {model_size} | AUPRC across {sub['count'].iloc[0]:.0f} seeds",
                        fontsize=15, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            
            # Add value labels
            for bar, mean, ci in zip(bars, sub["mean"], sub["ci_95"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 0.002,
                       f"{mean:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            
            plt.tight_layout()
            fname = f"multi_seed_{arch}_{model_size}_auprc.png"
            plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
            print(f"  📊 Saved: {fname}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-seed QAT results")
    parser.add_argument("--arch", choices=["swin", "cnn", "both"], default="both")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output-dir", type=str, default=str(MAIN_DIR / "comparison_plots"))
    args = parser.parse_args()
    
    architectures = ["swin", "cnn"] if args.arch == "both" else [args.arch]
    output_dir = Path(args.output_dir)
    
    print(f"{'═'*70}")
    print(f"📊 MULTI-SEED STATISTICAL ANALYSIS")
    print(f"{'═'*70}")
    
    # Collect all results
    all_dfs = []
    for arch in architectures:
        print(f"\nLoading {arch.upper()} results...")
        df = collect_all_results(arch)
        if not df.empty:
            all_dfs.append(df)
            print(f"  Found {len(df)} result entries")
        else:
            print(f"  ⚠️  No results found for {arch}")
    
    if not all_dfs:
        print("\n❌ No results found. Run 'python run_multi_seed.py' first.")
        return
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA COMPLETENESS CHECK
    # ═══════════════════════════════════════════════════════════════════════════
    expected_per_group = len(SEEDS)
    for arch in architectures:
        for model_size in MODEL_SIZES:
            for recipe_id in RECIPES:
                count = len(df[(df["arch"] == arch) & (df["model_size"] == model_size) & (df["recipe_id"] == recipe_id)])
                if count != expected_per_group:
                    print(f"  ⚠️  {arch.upper()} | {model_size} | recipe {recipe_id}: "
                          f"found {count}/{expected_per_group} seeds")
    
    total_expected = len(architectures) * len(MODEL_SIZES) * len(RECIPES) * len(SEEDS)
    print(f"\n  Data: {len(df)}/{total_expected} entries loaded "
          f"({'✅ complete' if len(df) == total_expected else '⚠️  INCOMPLETE'})")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. SUMMARY STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"1. SUMMARY STATISTICS (AUPRC)")
    print(f"{'─'*70}")
    
    summary = compute_summary_stats(df, "auprc")
    
    for arch in architectures:
        for model_size in MODEL_SIZES:
            sub = summary[(summary["arch"] == arch) & (summary["model_size"] == model_size)]
            if sub.empty:
                continue
            
            print(f"\n  {arch.upper()} | {model_size}:")
            print(f"  {'Recipe':<20s} {'Mean':>8s} {'± Std':>8s} {'95% CI':>16s} {'N':>4s}")
            print(f"  {'─'*58}")
            
            for _, row in sub.sort_values("mean", ascending=False).iterrows():
                ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                print(f"  {row['recipe_name']:<20s} {row['mean']:>8.4f} {row['std']:>7.4f} {ci_str:>16s} {row['count']:>4.0f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. PAIRWISE TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"2. PAIRWISE STATISTICAL TESTS (Wilcoxon signed-rank)")
    print(f"{'─'*70}")
    
    all_tests = []
    for arch in architectures:
        for model_size in MODEL_SIZES:
            tests = pairwise_tests(df, arch, model_size)
            if not tests.empty:
                all_tests.append(tests)
                
                sig_uncorr = tests["significant_005"].sum()
                has_corrected = "significant_005_corrected" in tests.columns
                sig_corr = tests["significant_005_corrected"].sum() if has_corrected else 0
                total = len(tests)
                print(f"\n  {arch.upper()} | {model_size}: "
                      f"{sig_uncorr}/{total} pairs significant (uncorrected p<0.05), "
                      f"{sig_corr}/{total} after Holm-Bonferroni correction")
                
                # Show significant pairs (using corrected p-values)
                p_col = "p_corrected" if has_corrected else "p_value"
                sig_col = "significant_005_corrected" if has_corrected else "significant_005"
                sig_tests = tests[tests[sig_col]].sort_values(p_col)
                if not sig_tests.empty:
                    for _, row in sig_tests.head(10).iterrows():
                        direction = ">" if row["mean_diff"] > 0 else "<"
                        p_raw = row["p_value"]
                        p_corr = row.get("p_corrected", p_raw)
                        print(f"    {row['recipe_1_name']} {direction} {row['recipe_2_name']}: "
                              f"Δ={row['mean_diff']:+.4f}, p={p_raw:.4f} (corrected: {p_corr:.4f}), d={row['cohens_d']:.2f}")
                else:
                    print(f"    No significant differences after correction")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. RANK STABILITY
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"3. RANK STABILITY (Kendall's W)")
    print(f"{'─'*70}")
    
    for arch in architectures:
        for model_size in MODEL_SIZES:
            rs = rank_stability(df, arch, model_size)
            w = rs["kendalls_w"]
            if np.isnan(w):
                continue
            
            interpretation = "Strong" if w > 0.7 else "Moderate" if w > 0.5 else "Weak"
            print(f"\n  {arch.upper()} | {model_size}: W = {w:.4f} ({interpretation} agreement)")
            
            if rs["mean_ranks"]:
                print(f"  {'Recipe':<20s} {'Mean Rank':>10s} {'± Std':>8s}")
                print(f"  {'─'*40}")
                sorted_ranks = sorted(rs["mean_ranks"].items(), key=lambda x: x[1])
                for rid, mean_rank in sorted_ranks:
                    std_rank = rs["rank_std"].get(rid, 0)
                    rname = RECIPE_NAMES.get(rid, str(rid))
                    print(f"  {rname:<20s} {mean_rank:>10.1f} {std_rank:>7.1f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    output_dir.mkdir(parents=True, exist_ok=True)

    def _safe_csv_write(frame: pd.DataFrame, primary: Path, fallback_suffix: str = "_15m") -> Path:
        """
        Write CSV to primary path; if it exists but isn't writable (e.g. owned by root),
        write to a suffixed filename instead.
        """
        try:
            frame.to_csv(primary, index=False)
            return primary
        except PermissionError:
            alt = primary.with_name(primary.stem + fallback_suffix + primary.suffix)
            frame.to_csv(alt, index=False)
            return alt
    
    # Save summary CSV
    saved_summary = _safe_csv_write(summary, output_dir / "multi_seed_summary.csv")
    print(f"\n  💾 Saved: {saved_summary}")
    
    # Save pairwise tests
    if all_tests:
        all_tests_df = pd.concat(all_tests, ignore_index=True)
        saved_tests = _safe_csv_write(all_tests_df, output_dir / "multi_seed_pairwise_tests.csv")
        print(f"  💾 Saved: {saved_tests}")
    
    # Save raw data
    saved_raw = _safe_csv_write(df, output_dir / "multi_seed_raw_results.csv")
    print(f"  💾 Saved: {saved_raw}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. PLOTS
    # ═══════════════════════════════════════════════════════════════════════════
    if args.plot:
        print(f"\n{'─'*70}")
        print(f"5. GENERATING PLOTS")
        print(f"{'─'*70}")
        plot_multi_seed_results(df, summary, output_dir)
    
    print(f"\n{'═'*70}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
