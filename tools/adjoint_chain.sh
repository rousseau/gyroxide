#!/bin/bash
# Week-by-week fit of the per-vertex growth a_t to the dHCP atlas by the equilibrium adjoint
# (examples/adjoint_week.rs, Burn autodiff), chained over the weeks w0..w1.
#
#   tools/adjoint_chain.sh <config.toml> <checkpoint at SG w0> <w0> <w1> <out dir> [adjoint_week options]
#   e.g. tools/adjoint_chain.sh configs/scan_hmu/h2_c2.toml out/scan_hmu/h2_c2/checkpoint_w23 23 28 \
#            out/adjoint_chain/h2_c2 --iters 8 --beta 1e-4
#
# Week w0: fit at the given checkpoint. Each later week w: run the dynamics one week (8769 steps)
# from the fitted state of week w-1 with the fitted table, then fit week w. The fitted rows are
# written into <out>/a_fit.npy (the config's a table is copied there first; the run uses
# <out>/config.toml, pointing to it). Per week: <out>/w<w>/{fwd.log, fwd/ (dynamics, surface
# before the fit), adj.log, adj/ (iterates), ckpt/ (fitted equilibrium)}; fitted surfaces as
# <out>/surface_<step>.vtk (tools/scan_metrics.py, tools/compare_grid.py --rows).
# Re-running the same command resumes after the last completed week.
# Needs data/adjoint_prep/w<w>/ (tools/adjoint_prep.py <w> data/adjoint_prep/w<w>) and the
# binaries: cargo build --release -p gyroxide-cli --bin gyroxide
#           cargo build --release -p gyroxide-cli --example adjoint_week --features "wgpu autodiff"
set -euo pipefail
cd "$(dirname "$0")/.."
cfg=$1; start=$2; w0=$3; w1=$4; D=$5; shift 5
MESH=data/meshes/w21_left_610k.mesh
BIN=target/release/gyroxide
ADJ=target/release/examples/adjoint_week
WEEK=8769
mkdir -p "$D"
a_src=$(sed -n 's/^a *= *"\(.*\)".*/\1/p' "$cfg")
[ -f "$D/a_fit.npy" ] || cp "$a_src" "$D/a_fit.npy"
sed "s#^a *= *\".*\"#a = \"$D/a_fit.npy\"#" "$cfg" > "$D/config.toml"
echo "$(date '+%F %T') chain SG $w0 -> $w1, config $cfg, options: $*" >> "$D/chain.log"

ck=$start
for w in $(seq "$w0" "$w1"); do
  W=$D/w$w
  if [ -f "$W/ckpt/state.json" ]; then ck=$W/ckpt; continue; fi
  mkdir -p "$W"
  if [ "$w" -gt "$w0" ]; then
    t0=$(date +%s)
    if ! "$BIN" run --config "$D/config.toml" --mesh "$MESH" --resume "$ck" --steps $WEEK --print-every 2000 \
        --vtk-every $WEEK --checkpoint-every $WEEK --out "$W/fwd" > "$W/fwd.log" 2>&1 || grep -q NaN "$W/fwd.log"; then
      echo "$(date '+%F %T') SG $w: dynamics failed (see $W/fwd.log)" | tee -a "$D/chain.log"
      exit 1
    fi
    ck=$W/fwd/checkpoint
    echo "$(date '+%F %T') SG $w: dynamics $(( $(date +%s) - t0 ))s" >> "$D/chain.log"
  fi
  t0=$(date +%s)
  "$ADJ" "$D/config.toml" "$MESH" "$ck" "data/adjoint_prep/w$w" "$W/adj" --theta0-table \
      --write-table "$D/a_fit.npy" --save-checkpoint "$W/ckpt.tmp" \
      --save-surface "$D/surface_$(printf %07d $(( (w - 21) * WEEK ))).vtk" "$@" > "$W/adj.log" 2>&1 || {
    echo "$(date '+%F %T') SG $w: adjoint failed (see $W/adj.log)" | tee -a "$D/chain.log"; exit 1; }
  mv "$W/ckpt.tmp" "$W/ckpt"
  ck=$W/ckpt
  echo "$(date '+%F %T') SG $w: fit $(( $(date +%s) - t0 ))s; $(grep -E '^start|^iter [0-9]+: shape' "$W/adj.log" | sed -n '1p;$p' | tr '\n' ' ')" >> "$D/chain.log"
done
echo "$(date '+%F %T') CHAIN DONE" >> "$D/chain.log"
