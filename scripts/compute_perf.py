#!/usr/bin/env python3
"""
Computational performance comparison between two run folders.

Plots mean timing per generation (with 95% CI) for each phase,
showing how cost evolves over training rather than pooling all generations.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TIMING_COLS = ["time_eval_ms", "time_speciate_ms", "time_reproduce_ms", "time_total_ms"]
TITLES = {
    "time_eval_ms":      "Evaluation time",
    "time_speciate_ms":  "Speciation time",
    "time_reproduce_ms": "Reproduction time",
    "time_total_ms":     "Total time",
}


def load_runs(folder: Path) -> list[pd.DataFrame]:
    csv_files = sorted(folder.glob("run_*.csv"))
    if not csv_files:
        print(f"No run_*.csv files in {folder}", file=sys.stderr)
        sys.exit(1)
    return [pd.read_csv(f).sort_values("generation").reset_index(drop=True)
            for f in csv_files]


def build_matrix(runs: list[pd.DataFrame], col: str) -> np.ndarray:
    """(n_runs x max_gen) matrix of per-generation timing values."""
    max_gen = max(df["generation"].max() for df in runs)
    matrix  = np.full((len(runs), max_gen + 1), np.nan)
    for i, df in enumerate(runs):
        for _, row in df.iterrows():
            g = int(row["generation"])
            matrix[i, g] = row[col]
    return matrix


def plot_timing(runs_a, runs_b, label_a, label_b):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    colors = {"a": "steelblue", "b": "darkorange"}

    for ax, col in zip(axes, TIMING_COLS):
        for key, runs, label in [("a", runs_a, label_a), ("b", runs_b, label_b)]:
            mat   = build_matrix(runs, col)
            gens  = np.arange(mat.shape[1])
            mean  = np.nanmean(mat, axis=0)
            ci_lo = np.nanpercentile(mat, 2.5,  axis=0)
            ci_hi = np.nanpercentile(mat, 97.5, axis=0)

            ax.fill_between(gens, ci_lo, ci_hi,
                            alpha=0.15, color=colors[key])
            ax.plot(gens, mean, color=colors[key], linewidth=1.8, label=label)

        ax.set_title(TITLES[col])
        ax.set_xlabel("Generation")
        ax.set_ylabel("Time (ms)")
        ax.legend()
        ax.grid(True, alpha=0.25)

    plt.suptitle(f"Computational cost per generation — {label_a} vs {label_b}",
                 fontsize=13)
    plt.tight_layout()
    plt.show()


def print_summary(runs_a, runs_b, label_a, label_b):
    """Mean timing over last 100 generations (steady-state)."""
    print(f"\n{'='*65}")
    print(f"  Steady-state timing — mean over last 100 generations")
    print(f"{'='*65}")
    print(f"  {'Metric':<22}  {'':>6}  {'mean':>8}  {'std':>8}")

    for col in TIMING_COLS:
        for label, runs in [(label_a, runs_a), (label_b, runs_b)]:
            vals = []
            for df in runs:
                tail = df.nlargest(100, "generation")[col].values
                vals.extend(tail)
            vals = np.array(vals)
            print(f"  {col:<22}  {label:>6}  {vals.mean():>8.3f}  {vals.std():>8.3f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-generation timing trajectories for two run folders."
    )
    parser.add_argument("folder_a",  help="First folder (run_*.csv files)")
    parser.add_argument("folder_b",  help="Second folder (run_*.csv files)")
    parser.add_argument("--label-a", default="A", help="Label for first folder")
    parser.add_argument("--label-b", default="B", help="Label for second folder")
    args = parser.parse_args()

    runs_a = load_runs(Path(args.folder_a))
    runs_b = load_runs(Path(args.folder_b))

    print(f"Loaded {len(runs_a)} runs from {args.folder_a}")
    print(f"Loaded {len(runs_b)} runs from {args.folder_b}")

    print_summary(runs_a, runs_b, args.label_a, args.label_b)
    plot_timing(runs_a, runs_b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
