#!/bin/sh
# Reference chain (measured growth, out/remodel_a0) re-equilibrated under another material, week by
# week: from the equilibrium of the chain material, mu_w switched first, then the bulk modulus K raised
# by continuation, Levenberg-Marquardt equilibria (cf. tools/material_chain.sh for the fitted models).
# Weekly chaining in the new material stalls at SG 24 (fold onset, an element inverted).
#   [MU_W=1] [KS="5 10 20 35 50"] tools/material_ref.sh <out root> <week> ...
# writes <out root>/w<w>/k<K>/{material.toml, eq/x_released.npy, run.log}, the last K as
# <out root>/w<w>/eq/x_released.npy (the layout of tools/remodel_chain.py, for tools/make_full.py).
set -e
cd "$(dirname "$0")/.."
ROOT=$1; shift
BIN=${BIN:-target/mu/adjoint_week_v8}
MW=${MU_W:-1}
for w in "$@"; do
    prev=out/remodel_a0/w$w/eq/x_released.npy
    for K in ${KS:-5 10 20 35 50}; do
        d=$ROOT/w$w/k$K
        mkdir -p $d/start
        sed -e "s/^mu_white = .*/mu_white = $MW/" -e "s/^bulk = .*/bulk = $K/" configs/mu_test/mu2_s.toml > $d/material.toml
        python3 - "$prev" "$d/start" "$w" <<'EOF'
import json, sys
import numpy as np
x = np.load(sys.argv[1]); ck = sys.argv[2]
np.save(f"{ck}/x.npy", x); np.save(f"{ck}/v.npy", np.zeros_like(x))
np.save(f"{ck}/growth_prev.npy", np.zeros(610828))
open(f"{ck}/state.json", "w").write(json.dumps({"step": 0, "t": (int(sys.argv[3]) - 21) / 15}))
EOF
        $BIN $d/material.toml data/meshes/w21_left_610k_s.mesh $d/start data/adjoint_prep_s/w$w $d/eq \
            --rigid --multiscale 0:0 --cumulative --theta0-table --relax0 0 --newton 150 --min-growth 0 \
            --inverse-eq $d/start/x.npy --iters 0 --release-relax 0 \
            --core-override out/remodel_a0/w$w/core_eff.npy --lm --newton-cg 1500 --force-tol 2e-5 > $d/run.log 2>&1
        echo "SG $w mu_w $MW K $K: $(grep -E '^released with|^released mm k=  0|^inverted elements:' $d/run.log | sed 's/absolute.*rigid rms/rigid/; s/ p50.*//' | tr '\n' ' ')"
        prev=$d/eq/x_released.npy
        rm -r $d/start
    done
    cp $prev $ROOT/w$w/eq_x_released.npy
    mkdir -p $ROOT/w$w/eq && mv $ROOT/w$w/eq_x_released.npy $ROOT/w$w/eq/x_released.npy
done
