#!/bin/sh
# Bras « modèle réinitialisé sur l'atlas w-1 », croissance mesurée (persistance équitable, P02 tâche 5) :
# pour chaque semaine précédente donnée, maillage de l'atlas w-1 (tools/week_reset.py), une semaine de croissance
# mesurée (adjoint_week, équilibre par Newton à pas de Levenberg-Marquardt, sans ajustement), tableau de bord face à
# l'atlas w, et la persistance simple dans le même repère (atlas w-1 brut contre atlas w brut).
#   [OUTD=out/reset_r0857] [MU_WHITE=1.167] tools/reset_arm.sh 25 27 29 32 35
# sorties : $OUTD/w<prev>/ (maillage, prep, eq/), $OUTD/scorecard/measured_<w>.json et persistence_<w>.json
set -e
cd "$(dirname "$0")/.."
OUTD=${OUTD:-out/reset_r0857}
BIN=${BIN:-target/mu/adjoint_week_sel}
GX=${GX:-target/release/gyroxide}
mkdir -p "$OUTD/scorecard"
for p in "$@"; do
    w=$((p + 1)); R=$OUTD/w$p
    [ -f "$R/mesh_s.mesh" ] || python3 tools/week_reset.py "$p" "$R" --mu-white "${MU_WHITE:-1.167}"
    python3 tools/mesh_quality.py "$R/mesh_s.mesh" "$R/mesh_s.nodal_h.npy" > "$R/qualite.txt"
    if [ ! -f "$R/eq/x_0.npy" ]; then
        $BIN "$R/config.toml" "$R/mesh_s.mesh" "$R/ckpt" "$R/prep" "$R/eq" --rigid --multiscale 200:0,50:0,10:0,0:0 \
            --cumulative --theta0-table --relax0 0 --newton 120 --lm --newton-cg 800 --min-growth 0 --iters 0 > "$R/run.log" 2>&1
    fi
    python3 - "$R" <<'PY'
import sys
import numpy as np
R = sys.argv[1]
np.save(R + "/x_surf.npy", np.load(R + "/eq/x_0.npy")[:32492])
np.save(R + "/x_prev_surf.npy", np.load(R + "/ckpt/x.npy")[:32492])
PY
    $GX eval "$R/x_surf.npy" --week "$w" --label measured --atlas-npy "$R/prep/target.npy" --mesh-name w21_left_610k_s \
        --out "$OUTD/scorecard/measured_$w.json"
    $GX eval "$R/x_prev_surf.npy" --week "$w" --label persistence --atlas-npy "$R/prep/target.npy" --mesh-name w21_left_610k_s \
        --out "$OUTD/scorecard/persistence_$w.json"
    grep -E "^start:|^inverted" "$R/run.log" > "$R/resume.txt" || true
done
