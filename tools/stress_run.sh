#!/bin/sh
# Stress of a fitted whole-hemisphere model of the lobar chain (tools/full_chain.sh, LAW=1 with
# --field-law-k), at its fitted equilibrium: checkpoint = the fitted state, theta = theta_final.
#   tools/stress_run.sh <name, e.g. lbm> <week> <out dir>
set -e
NAME=$1; n=$2; OUT=$3; w=$((n - 1))
L=data/full/law
RUN=out/full/fit${n}_$NAME
CK=$OUT/ckpt
mkdir -p $CK
cp data/full/ckpt_w${n}_from_${NAME}$n/state.json data/full/ckpt_w${n}_from_${NAME}$n/v.npy \
   data/full/ckpt_w${n}_from_${NAME}$n/growth_prev.npy $CK/
python3 -c "
import sys; sys.path.insert(0, 'tools'); import numpy as np
from full_eval import last_x
np.save('$CK/x.npy', np.ascontiguousarray(last_x('$RUN')))"
LB=$(ls $L/lb40/basis_lb0*_s100.npy | paste -sd,)
${BIN:-target/release/examples/adjoint_week} configs/mu_test/mu2_full_carry_${NAME}$n.toml data/meshes/w21_left_610k_s.mesh \
    $CK data/full/prep/w$n $OUT --absolute --multiscale 10:30,0:30 --cumulative --theta0-table \
    --relax 0 --relax0 0 --newton 40 --newton-cg 600 --min-growth 0 --growth-band 0.5 \
    --fit-fields c,k --field-band 50 --cg 600 --theta-init $RUN/theta_final.npy \
    --field-law $L/basis_d30_w${w}_s100.npy,$L/basis_d3o_w${w}_s100.npy,$L/basis_one_s100.npy \
    --field-law-k $LB --thick-mult --stress
