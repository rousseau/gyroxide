#!/bin/bash
# Forward run week by week (8769 steps per week, t = (SG - 21)/15), one restart point per week:
#   tools/forward_weeks.sh <config.toml> <out dir> [w1 = 36]
# writes <out>/w<SG>/{checkpoint/, surface_*.vtk, run.log}; re-running resumes after the last week.
set -euo pipefail
cd "$(dirname "$0")/.."
cfg=$1; D=$2; w1=${3:-36}
BIN=${BIN:-target/release/gyroxide}
MESH=${MESH:-data/meshes/w21_left_610k.mesh}
WEEK=8769
prev=""
for w in $(seq 22 "$w1"); do
  W=$D/w$w
  if [ -f "$W/checkpoint/state.json" ]; then prev=$W/checkpoint; continue; fi
  resume=(); [ -n "$prev" ] && resume=(--resume "$prev")
  t0=$(date +%s)
  "$BIN" run --config "$cfg" --mesh "$MESH" "${resume[@]}" --steps $WEEK --print-every 4000 \
      --vtk-every $WEEK --checkpoint-every $WEEK --out "$W" > "$W.log" 2>&1
  if grep -q NaN "$W.log"; then echo "SG $w: NaN (see $W.log)"; exit 1; fi
  prev=$W/checkpoint
  echo "$(date '+%F %T') SG $w done in $(( $(date +%s) - t0 ))s"
done
