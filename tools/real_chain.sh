#!/bin/sh
# The whole chain redone from SG 21 with the realistic material (mu_w = mu_g, K = 50 mu_g,
# configs/mu_test/mu2_s_real.toml, mu2_full_real.toml), Levenberg-Marquardt equilibria (--lm):
#  1. reference chain SG 22 -> 36 (measured growth, no fit): the equilibria of out/remodel_a0 taken to
#     the new material by continuation (tools/material_ref.sh), out/real/ref; chaining the weeks in
#     the new material (tools/remodel_chain.py) stalls at SG 24
#  2. whole-hemisphere inputs from it: tools/make_full.py, data/full_real (law maps: data/full/law)
#  3. SG 27 fit (law + lobar core, additive thickening as out/full/fit27_lb40), from its fitted
#     parameters (the fitted growth does not depend on the material) and from its fitted state taken
#     to the new material by continuation (out/real/c27_<K>; from the reference state the start
#     equilibrium fails), out/real/fit27_lbm
#  4. lobar chain SG 28 -> 36 (tools/full_chain.sh, LAW=1, --thick-mult, --keep-best), out/real
#   tools/real_chain.sh [first step, default 1]
set -e
cd "$(dirname "$0")/.."
BIN=${BIN:-target/mu/adjoint_week_v8}
SOLVER="--lm --newton-cg 1500 --force-tol 2e-5"
L=data/full/law
LB=$(ls $L/lb40/basis_lb*_s100.npy | paste -sd,)
S=${1:-1}
if [ "$S" -le 1 ]; then
    BIN=$BIN tools/material_ref.sh out/real/ref $(seq 22 36)
fi
if [ "$S" -le 2 ]; then
    python3 tools/make_full.py --out data/full_real --weeks $(seq 26 36) --full "out/real/ref/w{w}/eq/x_released.npy"
fi
if [ "$S" -le 3 ]; then
    FIT27="--absolute --cumulative --theta0-table --relax 0 --relax0 0 --newton 150 --min-growth 0 --max-step 0.05 \
--cg 600 --w-vertex 1 --w-normal 1 --w-lap 1 --w-area 0.5 --growth-band 0.5 --fit-fields c,k --field-band 50 \
--field-law $L/basis_d30_w27_s100.npy,$L/basis_d3o_w27_s100.npy,$L/basis_one_s100.npy --field-law-k $LB \
--theta-init out/full/fit27_lb40/theta_final.npy $SOLVER"
    x=$(python3 -c "import sys; sys.path.insert(0, 'tools'); from full_eval import last_x; import numpy as np; \
np.save('out/real/x27_lb40.npy', last_x('out/full/fit27_lb40'))" && echo out/real/x27_lb40.npy)
    for K in 5 10 20 35 50; do
        d=out/real/c27_$K
        mkdir -p $d/ckpt
        cp data/full/ckpt_w27/state.json data/full/ckpt_w27/v.npy data/full/ckpt_w27/growth_prev.npy $d/ckpt/
        cp $x $d/ckpt/x.npy
        sed -e "s/^mu_white = .*/mu_white = 1/" -e "s/^bulk = .*/bulk = $K/" configs/mu_test/mu2_full.toml > $d/material.toml
        # shellcheck disable=SC2086
        $BIN $d/material.toml data/meshes/w21_left_610k_s.mesh $d/ckpt data/full/prep/w27 $d \
            --multiscale 0:0 $FIT27 --stress > $d.log 2>&1
        echo "SG 27 K $K: $(grep -E '^stress mm k=  0|stress written' $d.log | sed 's/absolute.*rigid rms/rigid/; s/ p50.*//' | tr '\n' ' ')"
        x=$d/x_stress.npy
    done
    CK=data/full_real/ckpt_w27_fit
    rm -rf $CK && cp -r data/full_real/ckpt_w27 $CK && cp $x $CK/x.npy
    # shellcheck disable=SC2086
    $BIN configs/mu_test/mu2_full_real.toml data/meshes/w21_left_610k_s.mesh $CK data/full_real/prep/w27 \
        out/real/fit27_lbm --multiscale 0:60 $FIT27 --keep-best > out/real/fit27_lbm.log 2>&1
fi
# shellcheck disable=SC2086
LAW=1 DATA=data/full_real OUTD=out/real NEWTON=80 RELAX0=0 BIN=$BIN tools/full_chain.sh lbm out/real/fit27_lbm 27 36 \
    --field-law-k $LB --thick-mult --keep-best $SOLVER
