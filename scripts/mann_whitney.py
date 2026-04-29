#!/usr/bin/env python3
"""
Mann-Whitney U test comparing two sets of NEAT runs.

Compares:
  1. Generations-to-threshold (only runs that reached threshold)
  2. Normalised AUC (all runs)
  3. Normalised AUC (only runs that reached threshold)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SKIP_COLS = {"seed", "generation"}


def load_rolled(folder: Path, window: int) -> list[pd.Series]:
    csv_files = sorted(folder.glob("run_*.csv"))
    if not csv_files:
        print(f"No run_*.csv files in {folder}", file=sys.stderr)
        sys.exit(1)

    rolled = []
    for f in csv_files:
        df = pd.read_csv(f).sort_values("generation").reset_index(drop=True)
        s  = df["best_fitness"].rolling(window=window, min_periods=1).mean()
        s.index = df["generation"]
        rolled.append(s)
    return rolled


def generations_to_threshold(rolled: list[pd.Series], threshold: float) -> np.ndarray:
    result = []
    for s in rolled:
        hit = s[s >= threshold]
        result.append(hit.index[0] if len(hit) > 0 else np.nan)
    return np.array(result)


def compute_auc(rolled: list[pd.Series], threshold: float) -> np.ndarray:
    return np.array([
        np.trapezoid(s.values, s.index) / (s.index.max() * threshold)
        for s in rolled
    ])


def run_test(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, metric: str) -> None:
    a_clean = a[~np.isnan(a)]
    b_clean = b[~np.isnan(b)]

    if len(a_clean) < 2 or len(b_clean) < 2:
        print(f"  {metric}: insufficient data")
        return

    u_stat, p = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
    n1, n2    = len(a_clean), len(b_clean)
    r         = 1 - (2 * u_stat) / (n1 * n2)

    sig = "significant" if p < 0.05 else "NOT significant"

    print(f"\n  {metric}")
    print(f"    {label_a:25s}  n={n1:3d}  median={np.median(a_clean):.4f}  mean={a_clean.mean():.4f}  std={a_clean.std():.4f}")
    print(f"    {label_b:25s}  n={n2:3d}  median={np.median(b_clean):.4f}  mean={b_clean.mean():.4f}  std={b_clean.std():.4f}")
    print(f"    U={u_stat:.1f}  p={p:.4g}  r={r:.3f}  ({sig} at α=0.05)")


def main():
    parser = argparse.ArgumentParser(
        description="Mann-Whitney U test comparing two run folders."
    )
    parser.add_argument("folder_a",    help="First folder (run_*.csv files)")
    parser.add_argument("folder_b",    help="Second folder (run_*.csv files)")
    parser.add_argument("--label-a",   default="A",         help="Label for first folder")
    parser.add_argument("--label-b",   default="B",         help="Label for second folder")
    parser.add_argument("--threshold", type=float,          help="Fitness convergence threshold")
    parser.add_argument("--window",    type=int, default=25, help="Rolling window size (default: 25)")
    args = parser.parse_args()

    if args.threshold is None:
        print("Error: --threshold is required", file=sys.stderr)
        sys.exit(1)

    rolled_a = load_rolled(Path(args.folder_a), args.window)
    rolled_b = load_rolled(Path(args.folder_b), args.window)

    print(f"Loaded {len(rolled_a)} runs from {args.folder_a}")
    print(f"Loaded {len(rolled_b)} runs from {args.folder_b}")

    gens_a = generations_to_threshold(rolled_a, args.threshold)
    gens_b = generations_to_threshold(rolled_b, args.threshold)
    aucs_a = compute_auc(rolled_a, args.threshold)
    aucs_b = compute_auc(rolled_b, args.threshold)

    reached_a = int(np.sum(~np.isnan(gens_a)))
    reached_b = int(np.sum(~np.isnan(gens_b)))

    print(f"\n  {args.label_a}: {reached_a}/{len(gens_a)} reached threshold ({100*reached_a/len(gens_a):.1f}%)")
    print(f"  {args.label_b}: {reached_b}/{len(gens_b)} reached threshold ({100*reached_b/len(gens_b):.1f}%)")

    print(f"\n{'='*65}")
    print(f"  Mann-Whitney U  |  threshold={args.threshold:,.0f}  window={args.window}")
    print(f"{'='*65}")

    run_test(gens_a, gens_b, args.label_a, args.label_b,
             "Generations-to-threshold (reached only)")

    run_test(aucs_a, aucs_b, args.label_a, args.label_b,
             "Normalised AUC (all runs)")

    # Filter to runs that reached threshold in BOTH groups for a fair within-success comparison
    mask_a = ~np.isnan(gens_a)
    mask_b = ~np.isnan(gens_b)
    run_test(aucs_a[mask_a], aucs_b[mask_b], args.label_a, args.label_b,
             "Normalised AUC (reached-threshold runs only)")


if __name__ == "__main__":
    main()
