#!/bin/sh
# Convergence en maillage (P03, tâche 3) : chaîne de référence à croissance mesurée (tools/remodel_chain.py --alpha 0,
# SG 22 à $W1, défaut 30), puis `gyroxide eval` de chaque semaine, sur chacun des maillages donnés.
#   [OUTD=out/E38] [W1=30] tools/convergence_chain.sh g m f
# Chaque maillage est un dossier de tools/mesh_family.py : $OUTD/<nom>/{mesh_s.mesh, prep/, config.toml}. Sorties :
# $OUTD/<nom>/chain/ (une chaîne), $OUTD/<nom>/scorecard/measured_<SG>.json. Reprise : relancer la même commande
# (la chaîne repart après la dernière semaine finie).
set -e
cd "$(dirname "$0")/.."
OUTD=${OUTD:-out/E38}
W1=${W1:-30}
BIN=${BIN:-target/release/examples/adjoint_week}
GX=${GX:-target/release/gyroxide}
for m in "$@"; do
    D=$OUTD/$m
    last=21
    resume=""
    [ -f "$D/chain/chain.json" ] && last=$(python3 -c "import json,sys; print(max([e['sg'] for e in json.load(open('$D/chain/chain.json'))['weeks']] or [21]))")
    [ "$last" -gt 21 ] && resume="--resume $((last + 1))"
    if [ "$last" -lt "$W1" ]; then
        python3 tools/remodel_chain.py "$D/chain" --alpha 0 --w1 "$W1" $resume --config "$D/config.toml" \
            --mesh "$D/mesh_s.mesh" --prep "$D/prep" --bin "$BIN" >> "$D/chain.log" 2>&1
    fi
    mkdir -p "$D/scorecard"
    w=22
    while [ "$w" -le "$W1" ]; do
        [ -f "$D/chain/w$w/eq/x_released.npy" ] && [ ! -f "$D/scorecard/measured_$w.json" ] &&
            "$GX" eval "$D/chain/w$w/eq/x_released.npy" --week "$w" --label measured --atlas-npy "$D/prep/w$w/target.npy" \
                --meshes-dir "$D" --mesh-name mesh_s --out "$D/scorecard/measured_$w.json" >> "$D/eval.log" 2>&1 || true
        w=$((w + 1))
    done
done
