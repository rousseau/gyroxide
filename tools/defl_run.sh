#!/bin/sh
# P05, hypothèse 5, déflation : un équilibre du SG 36 (même configuration que tools/h5_run.sh : chaîne estimée à
# μ_g/μ_w = 0,857, sans itération de l'ajustement) par Newton déflaté des équilibres déjà trouvés.
#   tools/defl_run.sh <étiquette> <point de reprise> <équilibres à déflater, séparés par des virgules ; vide : témoin sans déflation> [options]
# Écrit out/E47/<étiquette>/{x_0.npy, ...} et out/E47/<étiquette>.log. Le Newton part du point de reprise sans
# relaxation (--relax0 0 : la dynamique amortie retomberait sur l'équilibre connu). BIN : binaire (défaut : celui
# du worktree, `cargo build --release --example adjoint_week --features autodiff`).
set -e
cd "$(dirname "$0")/.."
TAG=$1; CKPT=$2; DEFL=$3; shift 3
BIN=${BIN:-target/release/examples/adjoint_week}
G=${G:-$HOME/Exp/oxide/gyroxide}
F=${FIELDS:-out/paper_r0857/fit36_lbm/theta_final.npy}
L=data/full/law
LB=$(ls $L/lb40/basis_lb*_s100.npy | paste -sd,)
mkdir -p out/E47
# shellcheck disable=SC2086
$BIN $G/configs/mu_test/mu2_full_paper_r0857_carry_lbm36.toml data/meshes/w21_left_610k_s.mesh "$CKPT" \
    data/full_paper_r0857/prep/w36 out/E47/$TAG --absolute --multiscale 10:0,0:0 --cumulative --theta0-table \
    --relax 300 --relax0 ${RELAX0:-0} --newton ${NEWTON:-200} --min-growth 0 --max-step 0.05 --w-vertex 1 --w-normal 1 --w-lap 1 \
    --w-area 0.5 --growth-band 0.5 --fit-fields c,k --field-band 50 --cg 600 --fields-init "$F" \
    --field-law $L/basis_d30_w35_s100.npy,$L/basis_d3o_w35_s100.npy,$L/basis_one_s100.npy --field-law-k $LB \
    --thick-mult --keep-best --select-by loss --newton-cg 600 --force-tol 5e-5 --ls-tol 3e-3 \
    ${DEFL:+--defl-from "$DEFL"} "$@" > out/E47/$TAG.log 2>&1
grep -E "^start|^start mm|inverted" out/E47/$TAG.log | cut -c1-200
