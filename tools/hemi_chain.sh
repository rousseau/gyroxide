#!/bin/sh
# Chaîne de l'article ISBI pour un hémisphère (E34, P02 tâche 4) : maillage SG 21, tables, croissance mesurée et
# estimée à μ_g/μ_w = 0,857, tableau de bord. Le côté gauche existe déjà (tools/table1_repro.sh) ; ce script refait
# la même recette pour le droit, en réutilisant les mêmes outils (nom du côté dans le nom du maillage, tools/hemi.py).
#
#   HEMI=right tools/hemi_chain.sh [étape ...]       étapes : inputs configs remodel paper eval (défaut : toutes)
#
#   inputs   maillage TetGen de l'atlas SG 21 (tools/atlas_to_mesh.py, même densité en mm³ que le gauche : volume
#            maximal 0,11305 mm³ = 3,15e-6 en unités normalisées du gauche, 610 000 tétraèdres ; mur médial à 11 mm du point
#            le plus médial (8 mm pour le gauche : même part des sommets, 12,4 % contre 12,5 %) ; plan médian au point le plus
#            médial de SG 21 à 36 : le droit de l'atlas dérive de 2,7 mm au-delà de celui de SG 21), tables nodales
#            a et h (lissage 200), tenseurs du cœur, tétraèdres frontière scindés (tools/split_boundary_tets.py),
#            entrées par semaine (tools/adjoint_prep.py, SG 22 à 36)
#   configs  configurations du côté (tools/hemi_config.py : nom du maillage, plan médian, arête moyenne)
#   remodel  croissance mesurée à μ_g/μ_w = 0,857, SG 22 à 36 (tools/remodel_chain.py --alpha 0), out/remodel_<côté>_r0857
#   paper    croissance estimée, tools/paper_chain.sh (sélection loss), out/paper_<côté>_r0857
#   eval     gyroxide eval (SG 26, 28, 30, 33, 36 ; mesurée et estimée ; persistance SG 26 à 36), tableau dans
#            out/scorecard/t1_<côté>_r0857 ; cibles : celles de la chaîne mesurée du côté (data/full_paper_<côté>_r0857/prep)
# Reprises : FROM=<SG> pour `paper` (paper_chain.sh), RESUME=<SG> pour `remodel`.
set -e
cd "$(dirname "$0")/.."
HEMI=${HEMI:?HEMI=right}
[ "$HEMI" = left ] && { echo "le côté gauche : tools/table1_repro.sh" >&2; exit 2; }
M=data/meshes/w21_${HEMI}_610k
TAG=_${HEMI}_r0857
STEPS=${*:-"inputs configs remodel paper eval"}
A=${OXIDE_DHCP_ATLAS:-data/atlas_dhcp}

step_inputs() {
    [ -f $M.mesh ] || {
        python3 -I tools/atlas_to_mesh.py $A/fetal.week21.$HEMI.pial.surf.vtk $M.mesh --wm $A/fetal.week21.$HEMI.wm.surf.vtk \
            --maxvol-mm3 0.11305303406 --medial-dist 11 --medial-weeks 21 36
        mv $M.mesh.json $M.json
        mv $M.mesh.gr.txt $M.gr.txt
    }
    [ -f $M.nodal_a.npy ] || python3 -I tools/atlas_nodal_tables.py $M --smooth 200
    [ -f $M.nodal_core.npy ] || python3 -I tools/atlas_core_tensors.py $M
    for w in $(seq 22 36); do
        [ -f data/adjoint_prep_$HEMI/w$w/target.npy ] || python3 -I tools/adjoint_prep.py $w data/adjoint_prep_$HEMI/w$w $M
    done
    [ -f ${M}_s.mesh ] || (cd data/meshes && python3 -I ../../tools/split_boundary_tets.py w21_${HEMI}_610k.mesh w21_${HEMI}_610k_s.mesh \
        --node-npy w21_${HEMI}_610k.nodal_a.npy:1 w21_${HEMI}_610k.nodal_h.npy:1 w21_${HEMI}_610k.nodal_core.npy:1 \
        --node-txt w21_${HEMI}_610k.gr.txt --prep ../adjoint_prep_$HEMI:../adjoint_prep_s_$HEMI)
    python3 -I tools/mesh_quality.py ${M}_s.mesh ${M}_s.nodal_h.npy
}

step_configs() {
    python3 -I tools/hemi_config.py configs/mu_test/mu_r0857_s.toml configs/mu_test/mu_r0857_s_$HEMI.toml $M.json
    python3 -I tools/hemi_config.py configs/mu_test/mu2_full_r0857.toml configs/mu_test/mu2_full_r0857_$HEMI.toml $M.json
}

step_remodel() {
    python3 tools/remodel_chain.py out/remodel_${HEMI}_r0857 --alpha 0 --w1 36 --resume "${RESUME:-0}" \
        --config configs/mu_test/mu_r0857_s_$HEMI.toml --mesh ${M}_s.mesh --prep data/adjoint_prep_s_$HEMI
}

step_paper() {
    HEMI=$HEMI TAG=$TAG REMODEL=out/remodel_${HEMI}_r0857 CFG_BASE=configs/mu_test/mu2_full_r0857_$HEMI.toml SELECT=loss \
        tools/paper_chain.sh "${FROM:-1}"
}

step_eval() {
    O=out/scorecard/t1_${HEMI}_r0857
    T="data/full_paper${TAG}/prep/w{w}/target.npy"
    GX=${GX:-target/release/gyroxide}
    mkdir -p $O
    for w in ${WEEKS:-26 28 30 33 36}; do
        $GX eval out/remodel_${HEMI}_r0857/w$w/eq/x_released.npy --week $w --label measured --atlas-npy "$T" \
            --mesh-name w21_${HEMI}_610k_s --out $O/measured_$w.json
        $GX eval out/paper$TAG/fit${w}_lbm --week $w --label estimated --atlas-npy "$T" \
            --mesh-name w21_${HEMI}_610k_s --out $O/estimated_$w.json
    done
    $GX eval --persistence --weeks 26-36 --atlas-npy "$T" --mesh-name w21_${HEMI}_610k_s --out $O/persistence
    python3 tools/scorecard_table.py $O --out $O/table
}

for s in $STEPS; do
    case $s in
    inputs | configs | remodel | paper | eval) echo "== $s $(date -u +%FT%TZ)"; "step_$s" ;;
    *) echo "étape inconnue : $s (inputs configs remodel paper eval)" >&2; exit 2 ;;
    esac
done
echo "== fini $(date -u +%FT%TZ)"
