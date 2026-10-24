#!/bin/sh
# Week-by-week chain of whole-hemisphere fits (adjoint_week), from a fit of week W0 to week W1:
# each week starts from the previous fitted state moved by the reference-chain displacement, with
# the growth correction carried (tools/chain_week.py), and is fitted with the same options.
# With LAW=1 the field law of the previous fit is carried too: its coefficients start the next
# week (--fields-init) and its depth maps are those of the atlas one week earlier (predictive);
# --freeze-fields among the extra options keeps the coefficients fixed.
#   [LAW=1] tools/full_chain.sh <name> <first fitted run (week W0)> W0 W1 [extra adjoint_week options]
# (extra options are split on spaces: no spaces inside an option value)
set -e
NAME=$1; RUN=$2; W0=$3; W1=$4; shift 4
EXTRA="$*"
BIN=${BIN:-target/mu/adjoint_week_law2}
L=data/full/law
PREV=""
w=$W0
while [ $w -lt $W1 ]; do
    n=$((w + 1))
    python3 tools/chain_week.py data/full $w $RUN ${NAME}$n $PREV
    OUT=out/full/fit${n}_$NAME
    LAWOPT=""
    if [ "${LAW:-0}" = 1 ]; then
        LAWOPT="--fit-fields c,k --field-band 50 --cg 600 --fields-init $RUN/theta_final.npy \
--field-law $L/basis_d30_w${w}_s100.npy,$L/basis_d3o_w${w}_s100.npy,$L/basis_one_s100.npy"
    fi
    # shellcheck disable=SC2086
    $BIN configs/mu_test/mu2_full_carry_${NAME}$n.toml data/meshes/w21_left_610k_s.mesh \
        data/full/ckpt_w${n}_from_${NAME}$n data/full/prep/w$n $OUT \
        --absolute --multiscale 10:30,0:30 --cumulative --theta0-table --relax 0 --relax0 3000 \
        --newton 40 --min-growth 0 --max-step 0.05 --w-vertex 1 --w-normal 1 --w-lap 1 --w-area 0.5 \
        --growth-band 0.5 $LAWOPT $EXTRA 2>&1 | grep -E "^start mm k=  0|^final mm k=  0|panicked|inverted|coefficients"
    RUN=$OUT; PREV=${NAME}$n; w=$n
done
echo "chain done"
