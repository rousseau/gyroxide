#!/bin/sh
# Start equilibrium of week W1 reached from a fitted equilibrium of week W0 by continuation in time:
# N sub-steps t = (W0 + s/N - 21)/15, each a Newton equilibrium (adjoint_week, no fit, --relax0 0)
# from the previous one: a interpolated from the W0 knot to the W1 knot of the W1 config (carried
# table), law coefficients of the W0 fit (--fields-init among the options).
# The weekly jump from a fitted state is too large for both Newton and the damped relaxation
# once the field law is strong (SG 30 -> 31).
#   tools/time_ramp.sh <config W1> <ckpt of the W0 equilibrium> W0 W1 N <out ckpt> [adjoint_week options]
set -e
CFG=$1; CK0=$2; W0=$3; W1=$4; N=$5; OUT=$6; shift 6
BIN=${BIN:-target/mu/adjoint_week_diag}
TMP=$(mktemp -d)
PREV=$CK0
s=1
while [ $s -le $N ]; do
    CK=$TMP/ck_$s
    mkdir -p $CK
    cp $PREV/x.npy $CK/x.npy
    cp $CK0/v.npy $CK0/growth_prev.npy $CK/
    python3 -c "import json; print(json.dumps({'step': 0, 't': ($W0 + $s / $N - 21) / 15}))" > $CK/state.json
    # shellcheck disable=SC2086
    $BIN $CFG data/meshes/w21_left_610k_s.mesh $CK data/full/prep/w$W1 $TMP/run_$s \
        --absolute --multiscale 0:0 --theta0-table --relax 0 --relax0 0 --newton 60 --min-growth 0 \
        "$@" | grep -E "^start:|^start mm k=  0|panicked"
    cp $TMP/run_$s/x_0.npy $CK/x.npy
    PREV=$CK
    s=$((s + 1))
done
mkdir -p $OUT
cp $PREV/x.npy $PREV/v.npy $PREV/growth_prev.npy $PREV/state.json $OUT/
echo "ramp done: $OUT"
