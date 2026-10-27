#!/bin/sh
# Fitted models of the lobar chain re-equilibrated under another material (no refit): per week, from
# the fitted state, mu_w switched first, then the bulk modulus K raised by continuation (each K from
# the equilibrium of the previous one), Levenberg-Marquardt equilibria.
#   [MU_W=1] [KS="5 10 14 20 28 40 50"] tools/material_chain.sh <out root> <week:name> ...
#   e.g. tools/material_chain.sh out/material/chain 28:lbm 29:lbm 32:lbmf
set -e
ROOT=$1; shift
MW=${MU_W:-1}
for wk in "$@"; do
    n=${wk%%:*}; NAME=${wk#*:}
    prev=""
    for K in ${KS:-5 10 14 20 28 40 50}; do
        d=$ROOT/sg${n}_$(echo $MW | tr '.' 'p')_$K
        X0=$prev MAT=$MW,$K NEWTON=80 NEWTON_CG=1500 tools/stress_run.sh $NAME $n $d --lm --force-tol 2e-5 > $d.log 2>&1
        echo "SG $n mu_w $MW K $K: $(grep -E '^stress mm k=  0|stress written' $d.log | sed 's/absolute.*rigid rms/rigid/; s/ p50.*//; s/610828 tetrahedra, //' | tr '\n' ' ')"
        prev=$d/x_stress.npy
    done
done
