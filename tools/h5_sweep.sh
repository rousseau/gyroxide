#!/bin/sh
# P05, hypothèse 5 : balayage de perturbations lisses (croissance, géométrie) autour de l'état ajusté du SG 36
# (chaîne estimée à μ_g/μ_w = 0,857), un équilibre par exécution, sans itération de l'ajustement.
#   tools/h5_sweep.sh   (reprend où il s'est arrêté : une exécution terminée a son x_0.npy et son journal)
set -e
# (première version : g* identiques à la référence car a ignoré, x0.3 et x1.0 : éléments retournés, NaN ; corrigé le 14 novembre)
cd "$(dirname "$0")/.."
F=out/paper_r0857/fit36_lbm/theta_final.npy
C=data/full_paper_r0857/ckpt_w36_from_lbm36
I=out/E12/h5/inputs
mkdir -p $I
for seed in 1 2 3; do
  for eps in 0.02 0.05 0.10; do
    t=g${eps}_s$seed
    [ -f out/E12/h5/$t/x_0.npy ] && grep -q "^final mm k=  0" out/E12/h5/$t.log 2>/dev/null && continue
    python3 tools/h5_perturb.py growth $eps $seed $F $I/$t.npy
    # --fields-init ignore le champ a (il vient de --theta0-table) : la perturbation de a passe par --theta-init
    tools/h5_run.sh $t $F $C --theta-init $I/$t.npy || echo "échec $t" >> out/E12/h5/echecs.txt
  done
  for sig in 0.05 0.1 0.2; do
    t=x${sig}_s$seed
    [ -f out/E12/h5/$t/x_0.npy ] && grep -q "^final mm k=  0" out/E12/h5/$t.log 2>/dev/null && continue
    python3 tools/h5_perturb.py geom $sig $seed $C $I/ckpt_$t
    tools/h5_run.sh $t $F $I/ckpt_$t || echo "échec $t" >> out/E12/h5/echecs.txt
  done
done
