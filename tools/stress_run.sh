#!/bin/sh
# Stress of a fitted whole-hemisphere model of the lobar chain (tools/full_chain.sh, LAW=1 with
# --field-law-k), at its fitted equilibrium: checkpoint = the fitted state, theta = theta_final.
#   [MAT=mu_w,K] [RELAX0=n] [X0=x.npy] [NEWTON=n] [NEWTON_CG=n] tools/stress_run.sh <name, e.g. lbm> <week> <out dir>
# MAT: re-equilibrate the fitted growth with another material (white-matter shear modulus mu_w and
# bulk modulus K, in units of mu_g), without refitting. X0: start state (.npy), e.g. the x_stress.npy
# of a softer material (continuation in K), instead of the fitted state.
set -e
NAME=$1; n=$2; OUT=$3; w=$((n - 1)); shift 3
L=data/full/law
RUN=out/full/fit${n}_$NAME
CK=$OUT/ckpt
mkdir -p $CK
cp data/full/ckpt_w${n}_from_${NAME}$n/state.json data/full/ckpt_w${n}_from_${NAME}$n/v.npy \
   data/full/ckpt_w${n}_from_${NAME}$n/growth_prev.npy $CK/
python3 -c "
import sys; sys.path.insert(0, 'tools'); import numpy as np
from full_eval import last_x
x0 = '${X0:-}'
np.save('$CK/x.npy', np.ascontiguousarray(np.load(x0) if x0 else last_x('$RUN')))"
CFG=configs/mu_test/mu2_full_carry_${NAME}$n.toml
if [ -n "${MAT:-}" ]; then
    CFG=$OUT/material.toml
    sed -e "s/^mu_white = .*/mu_white = ${MAT%,*}/" -e "s/^bulk = .*/bulk = ${MAT#*,}/" \
        configs/mu_test/mu2_full_carry_${NAME}$n.toml > $CFG
fi
LB=$(ls $L/lb40/basis_lb0*_s100.npy | paste -sd,)
${BIN:-target/release/examples/adjoint_week} $CFG data/meshes/w21_left_610k_s.mesh \
    $CK data/full/prep/w$n $OUT --absolute --multiscale 10:30,0:30 --cumulative --theta0-table \
    --relax 0 --relax0 ${RELAX0:-0} --newton ${NEWTON:-40} --newton-cg ${NEWTON_CG:-600} --min-growth 0 --growth-band 0.5 \
    --fit-fields c,k --field-band 50 --cg 600 --theta-init $RUN/theta_final.npy \
    --field-law $L/basis_d30_w${w}_s100.npy,$L/basis_d3o_w${w}_s100.npy,$L/basis_one_s100.npy \
    --field-law-k $LB --thick-mult --stress "$@"
