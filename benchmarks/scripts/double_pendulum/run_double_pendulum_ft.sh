#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Double Pendulum — fixed-topology baseline.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/../../../build/neat_pendulum"

RESULTS_DIR="$SCRIPT_DIR/results_ft"

NUM_RUNS=300

MASTER_SEED=20250421

# --- NEAT parameters ---------------------------------------------------------
POPULATION_SIZE=150
COMPAT_THRESHOLD=3.0
DROPOFF_AGE=20
SURVIVAL_THRESHOLD=0.2
PROB_MUTATE_WEIGHT=0.8
ACTIVATION=tanh

# --- Fixed-topology overrides ------------------------------------------------
PROB_ADD_NODE=0.0
PROB_ADD_LINK=0.0
PROB_TOGGLE_ENABLE=0.0
C3=0.0

# =============================================================================
# Run
# =============================================================================

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found or not executable: $BINARY"
    echo "Build the project first (cd build && make -j\$(nproc))"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

mapfile -t SEEDS < <(awk -v n="$NUM_RUNS" -v seed="$MASTER_SEED" '
    BEGIN {
        srand(seed)
        for (i = 0; i < n; i++)
            printf "%d\n", int(rand() * 2147483647) + 1
    }
')

echo "Double Pendulum FT benchmark — $NUM_RUNS runs"
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
        --prob-toggle-enable "$PROB_TOGGLE_ENABLE" \
        --prob-mutate-weight "$PROB_MUTATE_WEIGHT" \
        --activation         "$ACTIVATION" \
        --c3                 "$C3" \
        --csv="$OUT"
done

echo "----------------------------------------------"
echo "Done. Results in $RESULTS_DIR"
