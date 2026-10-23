#!/bin/sh
# Start equilibrium of a field law imposed at full strength (adjoint_week --field-law), by
# continuation: the coefficients are scaled by 0.25, 0.5, 0.75, each equilibrium starting the next.
#   tools/law_ramp.sh <slab name | full> <coefficients.npy> <bases a,b,c> <out ckpt dir> [week]
# "cs" uses data/slab_cs and configs/mu_test/mu2_slab_cs.toml; "full" the whole hemisphere
# (data/full, tools/make_full.py, configs/mu_test/mu2_full.toml).
set -e
R=$1; INIT=$2; BASES=$3; CK=$4; W=${5:-27}
BIN=${BIN:-target/mu/adjoint_week_law}
if [ "$R" = full ]; then
    D=data/full; CFG=configs/mu_test/mu2_full.toml; MESH=data/meshes/w21_left_610k_s.mesh; FIX=""
else
    D=data/slab_$R; CFG=configs/mu_test/mu2_slab_$R.toml; MESH=$D/slab.mesh
    FIX="--dirichlet $D/fixed.npy --dirichlet-pos $D/fixed_pos_w$W.npy"
fi
PREV=$D/ckpt_w$W
TMP=$(mktemp -d)
for f in ${RAMP:-0.25 0.5 0.75}; do
    python3 -c "import numpy as np; t = np.load('$INIT'); na = len(t) - 6; t[na:] *= $f; np.save('$TMP/init.npy', t)"
    $BIN $CFG $MESH $PREV $D/prep/w$W $TMP/run_$f \
        --absolute --multiscale 0:0 --theta0-table --relax 0 --relax0 3000 --newton 40 --min-growth 0 \
        --growth-band 0.5 --fit-fields c,k --field-band 5 --freeze-fields $FIX \
        --field-law $BASES --fields-init $TMP/init.npy | grep -E "^setup|^start:|start mm k=  0"
    mkdir -p $TMP/ck_$f
    cp $D/ckpt_w$W/v.npy $D/ckpt_w$W/growth_prev.npy $D/ckpt_w$W/state.json $TMP/ck_$f/
    cp $TMP/run_$f/x_0.npy $TMP/ck_$f/x.npy
    PREV=$TMP/ck_$f
done
rm -rf $CK; cp -r $PREV $CK
echo "ramp done: $CK"
