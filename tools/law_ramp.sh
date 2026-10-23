#!/bin/sh
# Start equilibrium of a field law imposed at full strength (tools: adjoint_week --field-law), by
# continuation: the coefficients are scaled by 0.25, 0.5, 0.75, each equilibrium starting the next.
#   tools/law_ramp.sh <slab name> <coefficients.npy> <bases a,b,c> <out ckpt dir> [week]
set -e
R=$1; INIT=$2; BASES=$3; CK=$4; W=${5:-27}
BIN=${BIN:-target/mu/adjoint_week_law}
PREV=data/slab_$R/ckpt_w$W
TMP=$(mktemp -d)
for f in 0.25 0.5 0.75; do
    python3 -c "import numpy as np, sys; t = np.load('$INIT'); na = len(t) - 6; t[na:] *= $f; np.save('$TMP/init.npy', t)"
    $BIN configs/mu_test/mu2_slab_$R.toml data/slab_$R/slab.mesh $PREV data/slab_$R/prep/w$W $TMP/run_$f \
        --absolute --multiscale 0:0 --theta0-table --relax 0 --relax0 3000 --newton 40 --min-growth 0 \
        --growth-band 0.5 --fit-fields c,k --field-band 5 --freeze-fields \
        --dirichlet data/slab_$R/fixed.npy --dirichlet-pos data/slab_$R/fixed_pos_w$W.npy \
        --field-law $BASES --fields-init $TMP/init.npy | grep -E "^start:|start mm k=  0"
    mkdir -p $TMP/ck_$f
    cp data/slab_$R/ckpt_w$W/v.npy data/slab_$R/ckpt_w$W/growth_prev.npy data/slab_$R/ckpt_w$W/state.json $TMP/ck_$f/
    cp $TMP/run_$f/x_0.npy $TMP/ck_$f/x.npy
    PREV=$TMP/ck_$f
done
rm -rf $CK; cp -r $PREV $CK
echo "ramp done: $CK"
