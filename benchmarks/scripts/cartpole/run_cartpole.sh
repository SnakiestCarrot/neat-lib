#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CartPole benchmark — 30 randomised-seed runs
# Edit the variables below to change the experimental setup.
# =============================================================================

# --- Binary location (relative to this script) -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/../../../build/neat_cartpole"

# --- Output directory --------------------------------------------------------
RESULTS_DIR="$SCRIPT_DIR/results"

# --- Number of runs ----------------------------------------------------------
NUM_RUNS=300

# --- Master seed — change this to get a different (but reproducible) set of seeds
MASTER_SEED=20250421

# --- NEAT parameters (passed as CLI flags, override binary defaults) ---------
POPULATION_SIZE=150
COMPAT_THRESHOLD=3.0
DROPOFF_AGE=15
SURVIVAL_THRESHOLD=0.2
PROB_ADD_NODE=0.03
PROB_ADD_LINK=0.05
PROB_MUTATE_WEIGHT=0.8

# =============================================================================
# Run
# =============================================================================

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found or not executable: $BINARY"
    echo "Build the project first (cd build && make -j\$(nproc))"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# Generate NUM_RUNS seeds deterministically from MASTER_SEED using awk.
mapfile -t SEEDS < <(awk -v n="$NUM_RUNS" -v seed="$MASTER_SEED" '
    BEGIN {
        srand(seed)
        for (i = 0; i < n; i++)
            printf "%d\n", int(rand() * 2147483647) + 1
    }
')

echo "CartPole benchmark — $NUM_RUNS runs"
echo "Master seed : $MASTER_SEED"
echo "Results dir : $RESULTS_DIR"
echo "Binary      : $BINARY"
echo "----------------------------------------------"

for i in $(seq 0 $((NUM_RUNS - 1))); do
    SEED="${SEEDS[$i]}"
    RUN_NUM=$(printf "%02d" $((i + 1)))
    OUT="$RESULTS_DIR/run_${RUN_NUM}_seed_${SEED}.csv"

    echo "Run $RUN_NUM / $NUM_RUNS  (seed=$SEED)"

    "$BINARY" \
        --seed               "$SEED" \
        --population-size    "$POPULATION_SIZE" \
        --compat-threshold   "$COMPAT_THRESHOLD" \
        --dropoff-age        "$DROPOFF_AGE" \
        --survival-threshold "$SURVIVAL_THRESHOLD" \
        --prob-add-node      "$PROB_ADD_NODE" \
        --prob-add-link      "$PROB_ADD_LINK" \
        --prob-mutate-weight "$PROB_MUTATE_WEIGHT" \
        --csv="$OUT"
done

echo "----------------------------------------------"
echo "Done. Results in $RESULTS_DIR"
