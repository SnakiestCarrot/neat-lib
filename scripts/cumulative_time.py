#!/usr/bin/env python3
"""
Plot cumulative wall-clock training time vs generation for two run folders.

Usage:
    python cumulative_time.py <folder_a> <folder_b> --label-a NEAT --label-b FT
    python cumulative_time.py <folder_a> <folder_b> --label-a NEAT --label-b FT --output cumtime.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_runs(folder: Path) -> list[pd.DataFrame]:
    csv_files = sorted(folder.glob("run_*.csv"))
    if not csv_files:
        print(f"No run_*.csv files in {folder}", file=sys.stderr)
        sys.exit(1)
    return [pd.read_csv(f).sort_values("generation").reset_index(drop=True)
            for f in csv_files]


def build_cumulative_matrix(runs: list[pd.DataFrame]) -> np.ndarray:
    """(n_runs x max_gen+1) matrix of cumulative total time in seconds."""
    max_gen = max(df["generation"].max() for df in runs)
    matrix  = np.full((len(runs), max_gen + 1), np.nan)
    for i, df in enumerate(runs):
        cum = 0.0
        for _, row in df.iterrows():
            g = int(row["generation"])
            cum += row["time_total_ms"]
            matrix[i, g] = cum / 1000.0  # ms -> s
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative training time vs generation for two run folders."
    )
    parser.add_argument("folder_a",  help="First folder (run_*.csv files)")
    parser.add_argument("folder_b",  help="Second folder (run_*.csv files)")
    parser.add_argument("--label-a", default="A", help="Label for first folder")
    parser.add_argument("--label-b", default="B", help="Label for second folder")
    parser.add_argument("--output",  help="Save figure to this path instead of displaying")
    args = parser.parse_args()

    runs_a = load_runs(Path(args.folder_a))
    runs_b = load_runs(Path(args.folder_b))
    print(f"Loaded {len(runs_a)} runs from {args.folder_a}")
    print(f"Loaded {len(runs_b)} runs from {args.folder_b}")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"a": "steelblue", "b": "darkorange"}

    for key, runs, label in [("a", runs_a, args.label_a), ("b", runs_b, args.label_b)]:
        mat   = build_cumulative_matrix(runs)
        gens  = np.arange(mat.shape[1])
        mean  = np.nanmean(mat, axis=0)
        ci_lo = np.nanpercentile(mat, 2.5,  axis=0)
        ci_hi = np.nanpercentile(mat, 97.5, axis=0)

        ax.fill_between(gens, ci_lo, ci_hi, alpha=0.15, color=colors[key])
        ax.plot(gens, mean, color=colors[key], linewidth=1.8, label=label)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative wall-clock time (s)")
    ax.set_title(f"Cumulative training time — {args.label_a} vs {args.label_b}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
