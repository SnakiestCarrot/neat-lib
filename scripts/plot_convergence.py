#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot sliding-window best fitness from a NEAT run CSV.")
    parser.add_argument("csv", help="Path to run CSV file")
    parser.add_argument("--window", type=int, default=25, help="Sliding window size (default: 25)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["smoothed_best"] = df["best_fitness"].rolling(window=args.window, min_periods=1).mean()

    seed = df["seed"].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["generation"], df["best_fitness"],
            color="steelblue", alpha=0.3, linewidth=0.8, label="Best fitness (raw)")
    ax.plot(df["generation"], df["smoothed_best"],
            color="steelblue", linewidth=2.0, label=f"Best fitness (rolling mean, window={args.window})")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title(f"Convergence — seed {seed}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
