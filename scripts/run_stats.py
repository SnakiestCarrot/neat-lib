#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd

SKIP_COLS    = {"seed", "generation"}
FITNESS_COLS = ["best_fitness", "mean_fitness", "worst_fitness"]


def compute_stats(data: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = [c for c in data.columns if c not in SKIP_COLS]
    d = data[cols]
    df = pd.DataFrame({
        "mean":   d.mean(),
        "median": d.median(),
        "std":    d.std(),
        "var":    d.var(),
    })
    df.index.name = label
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean/median/std/var for all run CSVs in a folder, "
                    "plus rolling-mean stats for fitness columns."
    )
    parser.add_argument("folder", help="Folder containing run_*.csv files")
    parser.add_argument("--window", type=int, default=25,
                        help="Rolling window size for fitness smoothing (default: 25)")
    parser.add_argument("--output", help="Optional path to write stats as CSV")
    args = parser.parse_args()

    folder = Path(args.folder)
    csv_files = sorted(folder.glob("run_*.csv"))

    if not csv_files:
        print(f"No run_*.csv files found in {folder}", file=sys.stderr)
        sys.exit(1)

    raw_frames    = []
    rolled_frames = []

    for f in csv_files:
        df = pd.read_csv(f)
        raw_frames.append(df)

        # Rolling mean of fitness cols computed per-run, then collected
        rolled = df[FITNESS_COLS].rolling(window=args.window, min_periods=1).mean()
        rolled_frames.append(rolled)

    raw    = pd.concat(raw_frames,    ignore_index=True)
    rolled = pd.concat(rolled_frames, ignore_index=True)

    raw_stats    = compute_stats(raw,    label="column (raw)")
    rolled_stats = compute_stats(rolled, label=f"column (rolling mean w={args.window})")

    fmt = lambda x: f"{x:.4f}"

    print(f"Runs: {len(csv_files)}  |  Total rows: {len(raw)}\n")

    print("=== Raw values ===")
    print(raw_stats.to_string(float_format=fmt))

    print(f"\n=== Rolling mean (window={args.window}) of fitness columns ===")
    print(rolled_stats.to_string(float_format=fmt))

    if args.output:
        out = Path(args.output)
        raw_stats.to_csv(out.with_suffix(".raw.csv"))
        rolled_stats.to_csv(out.with_suffix(f".rolled_w{args.window}.csv"))
        print(f"\nWrote {out.with_suffix('.raw.csv')} and {out.with_suffix(f'.rolled_w{args.window}.csv')}")


if __name__ == "__main__":
    main()
