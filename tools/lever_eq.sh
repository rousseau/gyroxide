#!/bin/sh
# Levers on the folding of the reference (measured growth with core, out/remodel_a0, mu_g/mu_w = 2,
# K = 5): one parameter changed by continuation from the reference equilibrium, values in the order
# given, each equilibrium starting from the previous one.
#   mu   <v>: mu_white = v (mu_gray = 1);
#   core <s>: core growth tensors scaled, U = I + s (U_ref - I) (out/remodel_a0/w<w>/core_eff.npy).
#   tools/lever_eq.sh <out root> <mu|core> <week> <v1> <v2> ...
# writes <out root>/<mode>/w<w>/v<v>/{run.log, eq/x_released.npy}.
set -e
cd "$(dirname "$0")/.."
ROOT=$1; MODE=$2; w=$3; shift 3
BIN=${BIN:-target/mu/adjoint_week_v8}
prev=out/remodel_a0/w$w/eq/x_released.npy
for v in "$@"; do
    d=$ROOT/$MODE/w$w/v$v
    mkdir -p $d/start
    core=out/remodel_a0/w$w/core_eff.npy
    if [ $MODE = mu ]; then
        sed -e "s/^mu_white = .*/mu_white = $v/" configs/mu_test/mu2_s.toml > $d/material.toml
    else
        cp configs/mu_test/mu2_s.toml $d/material.toml
        core=$d/core.npy
    fi
    python3 - "$prev" "$d/start" "$w" "$MODE" "$v" "$d" <<'PY'
import json, sys
import numpy as np
x = np.load(sys.argv[1]); ck = sys.argv[2]; w = int(sys.argv[3])
np.save(f"{ck}/x.npy", x); np.save(f"{ck}/v.npy", np.zeros_like(x))
np.save(f"{ck}/growth_prev.npy", np.zeros(610828))
open(f"{ck}/state.json", "w").write(json.dumps({"step": 0, "t": (w - 21) / 15}))
if sys.argv[4] == "core":
    u = np.load(f"out/remodel_a0/w{w}/core_eff.npy"); s = float(sys.argv[5])
    eye = np.array([1.0, 1, 1, 0, 0, 0])
    np.save(f"{sys.argv[6]}/core.npy", eye + s * (u - eye))
PY
    $BIN $d/material.toml data/meshes/w21_left_610k_s.mesh $d/start data/adjoint_prep_s/w$w $d/eq \
        --rigid --multiscale 0:0 --cumulative --theta0-table --relax0 0 --newton 150 --min-growth 0 \
        --inverse-eq $d/start/x.npy --iters 0 --release-relax 0 \
        --core-override $core --lm --newton-cg 1500 --force-tol 2e-5 > $d/run.log 2>&1
    echo "SG $w $MODE $v: $(grep -E '^released with|^released mm k=  0|^inverted elements:' $d/run.log | sed 's/absolute.*rigid rms/rigid/; s/ p50.*//' | tr '\n' ' ')"
    prev=$d/eq/x_released.npy
    rm -r $d/start
    [ $MODE = core ] && rm $d/core.npy
done
