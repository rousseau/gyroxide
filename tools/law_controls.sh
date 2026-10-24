#!/bin/sh
# Field-law controls on a slab (tools/law_controls.py maps): the six coefficients and a fitted from
# zero, as out/slab_<name>/fit27_lawzero, with the depth maps of other weeks or spun maps.
#   tools/law_controls.sh <cs|calc> <tag> [<tag> ...]      (tags: w24 w26 spin1 spin2 ...)
R=$1; shift
BIN=${BIN:-target/mu/adjoint_week_lawk}
L=out/slab_$R/law
for T in "$@"; do
    $BIN configs/mu_test/mu2_slab_$R.toml data/slab_$R/slab.mesh data/slab_$R/ckpt_w27 data/slab_$R/prep/w27 \
        out/slab_$R/fit27_ctl_$T --absolute --multiscale 10:30,0:30 --cumulative --theta0-table --relax 0 \
        --relax0 3000 --newton 40 --min-growth 0 --max-step 0.05 --w-vertex 1 --w-normal 1 --w-lap 1 \
        --w-area 0.5 --growth-band 0.5 --fit-fields c,k --field-band 5 \
        --dirichlet data/slab_$R/fixed.npy --dirichlet-pos data/slab_$R/fixed_pos_w27.npy \
        --field-law $L/ctl_${T}_d30_s10.npy,$L/ctl_${T}_d3o_s10.npy,$L/basis_one_s10.npy \
        --fields-init $L/init_zero.npy > out/slab_${R}_fit27_ctl_$T.log 2>&1
    echo "$R $T EXIT $?"
done
