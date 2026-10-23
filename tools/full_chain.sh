#!/bin/sh
# Week-by-week chain of whole-hemisphere fits (adjoint_week), from a fit of week W0 to week W1:
# each week starts from the previous fitted state moved by the reference-chain displacement, with
# the growth correction carried (tools/chain_week.py), and is fitted with the same options.
#   tools/full_chain.sh <name> <first fitted run (week W0)> W0 W1 [extra adjoint_week options]
set -e
NAME=$1; RUN=$2; W0=$3; W1=$4; shift 4
BIN=${BIN:-target/mu/adjoint_week_law}
PREV=""
w=$W0
while [ $w -lt $W1 ]; do
    n=$((w + 1))
    python3 tools/chain_week.py data/full $w $RUN ${NAME}$n $PREV
    OUT=out/full/fit${n}_$NAME
    $BIN configs/mu_test/mu2_full_carry_${NAME}$n.toml data/meshes/w21_left_610k_s.mesh \
        data/full/ckpt_w${n}_from_${NAME}$n data/full/prep/w$n $OUT \
        --absolute --multiscale 10:30,0:30 --cumulative --theta0-table --relax 0 --relax0 3000 \
        --newton 40 --min-growth 0 --max-step 0.05 --w-vertex 1 --w-normal 1 --w-lap 1 --w-area 0.5 \
        --growth-band 0.5 "$@" 2>&1 | grep -E "^start mm k=  0|^final mm k=  0|panicked|inverted"
    RUN=$OUT; PREV=${NAME}$n; w=$n
done
echo "chain done"
