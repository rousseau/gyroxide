#!/bin/sh
# E40 : une exécution de la semaine SG 32 -> 33 de la chaîne estimée (mêmes entrées que tools/variabilite_sg33.sh), avec la somme du
# contact historique ou déterministe, pour comparer deux exécutions au bit près (tools/comparer_runs.py).
#   tools/determinisme_contact.sh <sortie> <niveaux multiscale> <historique|deterministe>
#   ex. : tools/determinisme_contact.sh out/E40/hist_a "10:2,0:2" historique
set -e
cd "$(dirname "$0")/.."
OUT=$1
LEVELS=$2
MODE=$3
BIN=${BIN:-target/release/examples/adjoint_week}
D=data/full_paper; L=data/full/law
LB=$(ls $L/lb40/basis_lb*_s100.npy | paste -sd,)
SOLVER="--newton-cg 600 --force-tol 5e-5 --ls-tol 3e-3"
w=32; n=33
RUN=${RUN_FIT32:-$HOME/Exp/oxide/gyroxide/out/paper/fit32_lbm}
CFG=configs/mu_test/mu2_full_paper_carry_lbm33.toml
mkdir -p "$OUT"
if [ "$MODE" = deterministe ]; then
    sed 's/^midplane_restoring = true$/midplane_restoring = true\ndeterministic_sum = true/' $CFG > "$OUT/config.toml"
    grep -q '^deterministic_sum = true' "$OUT/config.toml"
    CFG="$OUT/config.toml"
fi
# shellcheck disable=SC2086
$BIN $CFG data/meshes/w21_left_610k_s.mesh \
    $D/ckpt_w${n}_from_lbm$n $D/prep/w$n $OUT/fit \
    --absolute --multiscale "$LEVELS" --cumulative --theta0-table --relax 300 --relax0 3000 --newton 40 \
    --min-growth 0 --max-step 0.05 --w-vertex 1 --w-normal 1 --w-lap 1 --w-area 0.5 --growth-band 0.5 \
    --fit-fields c,k --field-band 50 --cg 600 --fields-init $RUN/theta_final.npy \
    --field-law $L/basis_d30_w${w}_s100.npy,$L/basis_d3o_w${w}_s100.npy,$L/basis_one_s100.npy \
    --field-law-k $LB --thick-mult --keep-best --select-by loss $SOLVER $EXTRA > "$OUT/fit.log" 2>&1
