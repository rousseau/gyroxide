#!/bin/sh
# Répétition de la semaine SG 32 -> 33 de la chaîne estimée (mêmes entrées que tools/paper_chain.sh), pour mesurer la
# variabilité d'une exécution à l'autre (P02, E33). Sélection de l'itéré par la perte (--select-by loss).
#   tools/variabilite_sg33.sh <numéro de répétition>      sortie : out/variabilite/rep<n>/fit33_lbm
set -e
cd "$(dirname "$0")/.."
BIN=${BIN:-target/mu/adjoint_week_sel}
D=data/full_paper; L=data/full/law
LB=$(ls $L/lb40/basis_lb*_s100.npy | paste -sd,)
SOLVER="--newton-cg 600 --force-tol 5e-5 --ls-tol 3e-3"
RUN=out/paper/fit32_lbm; w=32; n=33
OUT=out/variabilite/rep$1/fit33_lbm
mkdir -p "$(dirname "$OUT")"
# shellcheck disable=SC2086
$BIN configs/mu_test/mu2_full_paper_carry_lbm33.toml data/meshes/w21_left_610k_s.mesh \
    $D/ckpt_w${n}_from_lbm$n $D/prep/w$n $OUT \
    --absolute --multiscale 10:30,0:30 --cumulative --theta0-table --relax 300 --relax0 3000 --newton 40 \
    --min-growth 0 --max-step 0.05 --w-vertex 1 --w-normal 1 --w-lap 1 --w-area 0.5 --growth-band 0.5 \
    --fit-fields c,k --field-band 50 --cg 600 --fields-init $RUN/theta_final.npy \
    --field-law $L/basis_d30_w${w}_s100.npy,$L/basis_d3o_w${w}_s100.npy,$L/basis_one_s100.npy \
    --field-law-k $LB --thick-mult --keep-best --select-by loss $SOLVER > "$OUT.log" 2>&1
