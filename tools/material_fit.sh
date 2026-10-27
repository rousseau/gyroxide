#!/bin/sh
# Refit of a fitted whole-hemisphere model of the lobar chain (tools/full_chain.sh, LAW=1 with
# --field-law-k) under another material, from its fitted fields and an equilibrium of that material
# (tools/stress_run.sh with MAT=..., X0=...): same options as the chain, Levenberg-Marquardt
# equilibria (--lm), best iterate kept.
#   MAT=mu_w,K X0=x_stress.npy tools/material_fit.sh <name, e.g. lbm> <week> <out dir> [options]
set -e
NAME=$1; n=$2; OUT=$3; w=$((n - 1)); shift 3
L=data/full/law
RUN=out/full/fit${n}_$NAME
CK=$OUT/ckpt
mkdir -p $CK
cp data/full/ckpt_w${n}_from_${NAME}$n/state.json data/full/ckpt_w${n}_from_${NAME}$n/v.npy \
   data/full/ckpt_w${n}_from_${NAME}$n/growth_prev.npy $CK/
cp $X0 $CK/x.npy
CFG=$OUT/material.toml
sed -e "s/^mu_white = .*/mu_white = ${MAT%,*}/" -e "s/^bulk = .*/bulk = ${MAT#*,}/" \
    configs/mu_test/mu2_full_carry_${NAME}$n.toml > $CFG
LB=$(ls $L/lb40/basis_lb0*_s100.npy | paste -sd,)
${BIN:-target/release/examples/adjoint_week} $CFG data/meshes/w21_left_610k_s.mesh \
    $CK data/full/prep/w$n $OUT --absolute --multiscale 10:30,0:30 --cumulative --theta0-table \
    --relax 0 --relax0 0 --newton 80 --newton-cg 1500 --min-growth 0 --max-step 0.05 \
    --w-vertex 1 --w-normal 1 --w-lap 1 --w-area 0.5 --growth-band 0.5 \
    --fit-fields c,k --field-band 50 --cg 600 --theta-init $RUN/theta_final.npy \
    --field-law $L/basis_d30_w${w}_s100.npy,$L/basis_d3o_w${w}_s100.npy,$L/basis_one_s100.npy \
    --field-law-k $LB --thick-mult --lm --force-tol 2e-5 --keep-best "$@"
