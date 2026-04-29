#!/usr/bin/env python3
"""
Convergence analysis for NEAT runs.

Produces three analyses:
  1. Generations-to-threshold: first gen where rolling best >= T across runs
  2. Per-generation learning curve: mean + 95% CI of best fitness at each gen
  3. AUC of best fitness curve (normalised by max_gen * threshold)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_runs(folder: Path, window: int) -> tuple[list[pd.Series], pd.DataFrame]:
    """
    Returns:
      rolled_series : list of per-run rolling-mean best_fitness Series (indexed by generation)
      raw_concat    : all raw rows concatenated
    """
    csv_files = sorted(folder.glob("run_*.csv"))
    if not csv_files:
        print(f"No run_*.csv files in {folder}", file=sys.stderr)
        sys.exit(1)

    rolled_series = []
    raw_frames    = []

    for f in csv_files:
        df = pd.read_csv(f).sort_values("generation").reset_index(drop=True)
        raw_frames.append(df)
        rolled = df["best_fitness"].rolling(window=window, min_periods=1).mean()
        rolled.index = df["generation"]
        rolled_series.append(rolled)

    return rolled_series, pd.concat(raw_frames, ignore_index=True)


def generations_to_threshold(rolled_series: list[pd.Series], threshold: float) -> np.ndarray:
    """First generation per run where rolling best >= threshold. NaN if never reached."""
    results = []
    for s in rolled_series:
        hit = s[s >= threshold]
        results.append(hit.index[0] if len(hit) > 0 else np.nan)
    return np.array(results)


def print_threshold_stats(gens: np.ndarray, threshold: float, label: str = "") -> None:
    reached  = gens[~np.isnan(gens)]
    n_total  = len(gens)
    n_reached = len(reached)

    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"  Threshold: {threshold:,.0f}")
    print(f"{'='*60}")
    print(f"  Runs reaching threshold : {n_reached} / {n_total}  ({100*n_reached/n_total:.1f}%)")
    if n_reached:
        print(f"  Mean generations        : {reached.mean():.1f}")
        print(f"  Median generations      : {np.median(reached):.1f}")
        print(f"  Std dev                 : {reached.std():.1f}")
        print(f"  Min / Max               : {reached.min():.0f} / {reached.max():.0f}")
        print(f"  95th percentile         : {np.percentile(reached, 95):.1f}")


def plot_learning_curve(rolled_series: list[pd.Series], threshold: float, window: int,
                        label: str = "") -> None:
    # Align all series to a common generation index
    max_gen = max(s.index.max() for s in rolled_series)
    gen_idx = np.arange(0, max_gen + 1)

    matrix = np.full((len(rolled_series), len(gen_idx)), np.nan)
    for i, s in enumerate(rolled_series):
        for g, v in s.items():
            if g < len(gen_idx):
                matrix[i, g] = v

    mean   = np.nanmean(matrix, axis=0)
    ci_lo  = np.nanpercentile(matrix, 2.5,  axis=0)
    ci_hi  = np.nanpercentile(matrix, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(gen_idx, ci_lo, ci_hi, alpha=0.2, color="steelblue", label="95% CI across runs")
    ax.plot(gen_idx, mean, color="steelblue", linewidth=2, label=f"Mean best fitness (rolling w={window})")
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Threshold {threshold:,.0f}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (rolling mean)")
    title = f"Learning Curve — {len(rolled_series)} runs"
    if label:
        title = f"{label} — {title}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def compute_auc(rolled_series: list[pd.Series], threshold: float) -> np.ndarray:
    """Normalised AUC per run: integral of rolling best / (max_gen * threshold)."""
    aucs = []
    for s in rolled_series:
        auc = np.trapezoid(s.values, s.index) / (s.index.max() * threshold)
        aucs.append(auc)
    return np.array(aucs)


def main():
    parser = argparse.ArgumentParser(description="Convergence analysis for NEAT run folders.")
    parser.add_argument("folder", help="Folder containing run_*.csv files")
    parser.add_argument("--threshold", type=float, default=1_200_000,
                        help="Fitness threshold for convergence (default: 1200000)")
    parser.add_argument("--window", type=int, default=25,
                        help="Rolling window size (default: 25)")
    parser.add_argument("--label", default="", help="Label for plot titles (e.g. 'NEAT')")
    args = parser.parse_args()

    folder = Path(args.folder)
    rolled_series, _ = load_runs(folder, args.window)

    print(f"Loaded {len(rolled_series)} runs from {folder}")

    # 1. Generations to threshold
    gens = generations_to_threshold(rolled_series, args.threshold)
    print_threshold_stats(gens, args.threshold, label=args.label or str(folder.name))

    # 2. AUC
    aucs = compute_auc(rolled_series, args.threshold)
    print(f"\n  Normalised AUC — mean: {aucs.mean():.4f}  std: {aucs.std():.4f}  "
          f"median: {np.median(aucs):.4f}")

    # 3. Learning curve plot
    plot_learning_curve(rolled_series, args.threshold, args.window,
                        label=args.label or str(folder.name))


if __name__ == "__main__":
    main()
