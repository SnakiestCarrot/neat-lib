#!/usr/bin/env bash
# =============================================================================
# Fixed-topology sweep — runs all four FT benchmark scripts sequentially.
# Mirrors run_all.sh; results land in each env's results_ft/ directory.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SUMMARY_LOG="$SCRIPT_DIR/run_all_ft.log"

# Same order as run_all.sh: fastest first.
ENVS=(
    "cartpole|$SCRIPT_DIR/cartpole/run_cartpole_ft.sh|$BUILD_DIR/neat_cartpole"
    "pendulum|$SCRIPT_DIR/double_pendulum/run_double_pendulum_ft.sh|$BUILD_DIR/neat_pendulum"
    "platformer|$SCRIPT_DIR/platformer/run_platformer_ft.sh|$BUILD_DIR/neat_platformer"
    "snake|$SCRIPT_DIR/snake/run_snake_ft.sh|$BUILD_DIR/neat_snake"
)

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
echo "Pre-flight checks (fixed-topology sweep)..."
missing=0
for entry in "${ENVS[@]}"; do
    IFS='|' read -r name script binary <<< "$entry"
    if [[ ! -x "$binary" ]]; then
        echo "  MISSING: $binary"
        missing=1
    fi
    if [[ ! -x "$script" ]]; then
        echo "  NOT EXECUTABLE: $script"
        missing=1
    fi
done
if [[ $missing -ne 0 ]]; then
    echo "ERROR: prerequisites missing. Build first:"
    echo "  cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j\$(nproc)"
    exit 1
fi

if [[ -n "$(cd "$REPO_ROOT" && git status --porcelain 2>/dev/null)" ]]; then
    echo
    echo "WARNING: working tree has uncommitted changes."
    echo "         Sidecars do not record a git hash, so the code state that"
    echo "         produced these results will not be recoverable later."
    echo "         Commit before launching if reproducibility matters."
    echo
    read -r -p "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

echo "OK."
echo

# -----------------------------------------------------------------------------
# Run sweeps
# -----------------------------------------------------------------------------
declare -a STATUSES DURATIONS

fmt_duration() {
    local s=$1
    printf '%dh %02dm %02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

total_start=$(date +%s)
total_envs=${#ENVS[@]}
i=0
for entry in "${ENVS[@]}"; do
    i=$((i + 1))
    IFS='|' read -r name script binary <<< "$entry"

    echo "================================================================"
    echo "[$i/$total_envs] $name (FT) — starting at $(date '+%H:%M:%S')"
    echo "================================================================"

    env_start=$(date +%s)
    if bash "$script"; then
        STATUSES[$i]="OK"
    else
        STATUSES[$i]="FAILED (exit $?)"
    fi
    env_end=$(date +%s)
    DURATIONS[$i]=$((env_end - env_start))

    echo
    echo "[$i/$total_envs] $name (FT) — ${STATUSES[$i]} in $(fmt_duration ${DURATIONS[$i]})"
    echo
done
total_end=$(date +%s)
total_duration=$((total_end - total_start))

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
{
    echo "================================================================"
    echo "Summary (fixed-topology) — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    printf '%-12s  %-16s  %s\n' "env" "status" "duration"
    printf '%-12s  %-16s  %s\n' "------------" "----------------" "----------"
    i=0
    for entry in "${ENVS[@]}"; do
        i=$((i + 1))
        IFS='|' read -r name _ _ <<< "$entry"
        printf '%-12s  %-16s  %s\n' "$name" "${STATUSES[$i]}" "$(fmt_duration ${DURATIONS[$i]})"
    done
    printf '%-12s  %-16s  %s\n' "------------" "----------------" "----------"
    printf '%-12s  %-16s  %s\n' "total" "" "$(fmt_duration $total_duration)"
} | tee "$SUMMARY_LOG"

for s in "${STATUSES[@]}"; do
    [[ "$s" == "OK" ]] || exit 1
done
