#!/bin/sh
# P05, hypothèse 5 : un équilibre du SG 36 (chaîne estimée à μ_g/μ_w = 0,857) à partir de champs et d'un point
# de reprise éventuellement perturbés (tools/h5_perturb.py), sans itération de l'ajustement :
#   tools/h5_run.sh <étiquette> <fields.npy> <point de reprise> [options supplémentaires]
# Écrit out/E12/h5/<étiquette>/{x_0.npy, ...} et out/E12/h5/<étiquette>.log. BIN, ROOT_DATA : voir plus bas.
set -e
cd "$(dirname "$0")/.."
TAG=$1; FIELDS=$2; CKPT=$3; shift 3
G=${G:-$HOME/Exp/oxide/gyroxide}
BIN=${BIN:-$G/target/mu/adjoint_week_sel}
L=data/full/law
LB=$(ls $L/lb40/basis_lb*_s100.npy | paste -sd,)
mkdir -p out/E12/h5
# shellcheck disable=SC2086
$BIN $G/configs/mu_test/mu2_full_paper_r0857_carry_lbm36.toml data/meshes/w21_left_610k_s.mesh "$CKPT" \
    data/full_paper_r0857/prep/w36 out/E12/h5/$TAG --absolute --multiscale 10:0,0:0 --cumulative --theta0-table \
    --relax 300 --relax0 3000 --newton 40 --min-growth 0 --max-step 0.05 --w-vertex 1 --w-normal 1 --w-lap 1 \
    --w-area 0.5 --growth-band 0.5 --fit-fields c,k --field-band 50 --cg 600 --fields-init "$FIELDS" \
    --field-law $L/basis_d30_w35_s100.npy,$L/basis_d3o_w35_s100.npy,$L/basis_one_s100.npy --field-law-k $LB \
    --thick-mult --keep-best --select-by loss --newton-cg 600 --force-tol 5e-5 --ls-tol 3e-3 "$@" > out/E12/h5/$TAG.log 2>&1
grep -E "^start|^start mm" out/E12/h5/$TAG.log | cut -c1-200
