#!/bin/sh
# Stability of the reference equilibria (measured growth with core, out/remodel_a0): the equilibrium is
# displaced by a smooth random field (random Fourier modes, wavelength L mm, rms amplitude A mm, same
# displacement for every node) and re-equilibrated with the same growth and material.
#   [AMPS="0.3 1.0"] [L=6] [SEED=0] tools/perturb_eq.sh <out root> <week> ...
# writes <out root>/w<w>/a<A>/{run.log, eq/x_released.npy}.
set -e
cd "$(dirname "$0")/.."
ROOT=$1; shift
BIN=${BIN:-target/mu/adjoint_week_v8}
for w in "$@"; do
    for A in ${AMPS:-0.3 1.0}; do
        d=$ROOT/w$w/a$A
        mkdir -p $d/start
        python3 - out/remodel_a0/w$w/eq/x_released.npy "$d/start" "$w" "$A" "${L:-6}" "${SEED:-0}" <<'PY'
import json, sys
import numpy as np
x = np.load(sys.argv[1]); ck = sys.argv[2]; w = int(sys.argv[3]); amp, lam, seed = map(float, sys.argv[4:7])
mm = json.load(open("data/meshes/w21_left_610k_s.json"))["maxd_mm"]
rng = np.random.default_rng(int(seed) + w)
k = rng.normal(size=(64, 3)); k *= 2 * np.pi / (lam / mm) / np.linalg.norm(k, axis=1, keepdims=True)
ph = rng.uniform(0, 2 * np.pi, 64); c = rng.normal(size=(64, 3))
u = np.cos(x @ k.T + ph) @ c
u *= amp / mm / np.sqrt((u ** 2).sum(1).mean())
np.save(f"{ck}/x.npy", x + u); np.save(f"{ck}/v.npy", np.zeros_like(x))
np.save(f"{ck}/growth_prev.npy", np.zeros(610828))
open(f"{ck}/state.json", "w").write(json.dumps({"step": 0, "t": (w - 21) / 15}))
PY
        $BIN configs/mu_test/mu2_s.toml data/meshes/w21_left_610k_s.mesh $d/start data/adjoint_prep_s/w$w $d/eq \
            --rigid --multiscale 0:0 --cumulative --theta0-table --relax0 0 --newton 150 --min-growth 0 \
            --inverse-eq $d/start/x.npy --iters 0 --release-relax 0 \
            --core-override out/remodel_a0/w$w/core_eff.npy --lm --newton-cg 1500 --force-tol 2e-5 > $d/run.log 2>&1
        echo "SG $w amp $A mm: $(grep -E '^released with|^released mm k=  0|^inverted elements:' $d/run.log | sed 's/absolute.*rigid rms/rigid/; s/ p50.*//' | tr '\n' ' ')"
        rm -r $d/start
    done
done
