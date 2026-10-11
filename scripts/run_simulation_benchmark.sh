#!/usr/bin/env bash
# Full simulation benchmark: generate a tier, run all methods (by priority), evaluate.
#
# Usage:   scripts/run_simulation_benchmark.sh [tier] [jobs]
#          tier = full (default) | core ; jobs = parallel processes (default 14)
# Detached (survives closing the terminal):
#          nohup setsid scripts/run_simulation_benchmark.sh full 14 > outputs/sim_full_run.log 2>&1 &
#
# Expected runtime for "full" (10,200 files) with 14 processes on a 16-core machine:
#   generate ~0.5 h | rational + hcrp ~0.5 h | PELT (all sequences) ~4.5 h
#   HSMM (blue) ~6.5 h | community network (blue) ~5 h | evaluate ~0.5 h
# Each step is written to outputs/sim_<tier>/; evaluation can be re-run at any time on the
# results finished so far:  python -m src.simulation evaluate --sim-dir simulations/<tier> --results outputs/sim_<tier>
set -euo pipefail
cd "$(dirname "$0")/.."

TIER="${1:-full}"
JOBS="${2:-14}"
PY="${PYTHON:-.venv/bin/python}"
SIM="simulations/${TIER}"
OUT="outputs/sim_${TIER}"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

if [ ! -f "${SIM}/simulation_meta.json" ]; then
  echo "[$(stamp)] generate ${TIER}"
  "$PY" -m src.simulation generate --config "configs/simulation/${TIER}.yaml" --jobs "$JOBS"
fi

run() {  # run <methods> [sequences]
  echo "[$(stamp)] run-methods $*"
  if [ -n "${2:-}" ]; then
    "$PY" -m src.simulation run-methods --sim-dir "$SIM" --results "$OUT" --methods "$1" --sequences "$2" --jobs "$JOBS"
  else
    "$PY" -m src.simulation run-methods --sim-dir "$SIM" --results "$OUT" --methods "$1" --jobs "$JOBS"
  fi
}

run rational_chunking,hcrp_lm
run change_point_pelt
"$PY" -m src.simulation evaluate --sim-dir "$SIM" --results "$OUT"   # intermediate evaluation
run hsmm_chunking blue
run community_network blue

echo "[$(stamp)] evaluate"
"$PY" -m src.simulation evaluate --sim-dir "$SIM" --results "$OUT"
echo "[$(stamp)] DONE"
