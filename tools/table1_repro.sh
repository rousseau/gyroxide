#!/bin/sh
# Table 1 de l'article ISBI 2027 depuis zéro, en un seul script (P02, tâche 1) : trois lignes de modèle
# (loi d'origine, croissance mesurée, croissance estimée), persistance, tableau de bord, comparaison à l'article.
#
#   tools/table1_repro.sh [étape ...]        étapes : build tab remodel paper eval check (défaut : toutes)
#   TABLE1_METRICS=<table_metrics.json> tools/table1_repro.sh check
#
# Étapes (chaque étape écrit sa durée dans out/table1_repro/duree.tsv : étape, secondes, heure UTC de fin) :
#   build    cargo build --release, puis l'exemple adjoint_week (--features autodiff) copié dans target/mu/ sous les
#            noms attendus par les outils
#   tab      « Original growth law » : gyroxide run, loi tabulée de Tallinen, 43 845 pas (SG 21 → 26), une surface
#            tous les 8 769 pas, out/atlas_w21_tab (GPU wgpu/CUDA, environ 45 min sur la PGX)
#   remodel  croissance mesurée, tools/remodel_chain.py --alpha 0 --core, SG 22 → 36, out/remodel_a0 ;
#            puis tools/make_full.py (data/full), entrée des outils de mesure
#   paper    croissance estimée, tools/paper_chain.sh (SG 22 → 36 ; 10 à 30 min par semaine jusqu'à SG 31, puis
#            3 à 6 h par semaine) ; la durée de chaque semaine est celle de son journal (fit<SG>_lbm.log)
#   eval     gyroxide eval des semaines 26, 28, 30, 33, 36 (mesurée, estimée), persistance SG 26 à 36,
#            tableau (tools/scorecard_table.py) dans out/scorecard/t1
#   check    tools/table1_check.py contre TABLE1_METRICS (écart relatif par cellule, 1 % de tolérance)
#
# Prérequis (hors du script) : data/atlas_dhcp (atlas dHCP, voir CLAUDE.md d'oxide), data/meshes/w21_left_610k*
# et data/adjoint_prep_s (maillage fendu : tools/split_boundary_tets.py, voir docs/validation.md), tables nodales.
# Reproductibilité : le commit, le binaire et les configurations sont figés dans out/table1_repro/provenance.txt.
# Aucune graine aléatoire : la chaîne est déterministe à l'ordre de réduction près des fils (voir docs/adjoint_slab.md,
# « Version de référence »). Un run complet dure plus d'une journée sur la PGX (machine partagée) : tmux ou nohup.
# Les reprises : l'étape `paper` accepte FROM=<SG> (paper_chain.sh) ; `remodel` reprend avec RESUME=<SG>.
set -e
cd "$(dirname "$0")/.."
OUT=out/table1_repro
WEEKS=${WEEKS:-"26 28 30 33 36"}
STEPS=${*:-"build tab remodel paper eval check"}
mkdir -p "$OUT"

chrono() { # chrono <étape> <commande...> : exécute et note la durée
    name=$1; shift
    t0=$(date +%s)
    "$@"
    printf '%s\t%s\t%s\n' "$name" "$(($(date +%s) - t0))" "$(date -u +%FT%TZ)" >> "$OUT/duree.tsv"
}

provenance() {
    {
        echo "commit: $(git rev-parse HEAD)$(git diff --quiet HEAD -- . 2> /dev/null || echo ' (arbre modifié)')"
        echo "date: $(date -u +%FT%TZ)"
        echo "machine: $(uname -mn)"
        echo "toolchain: $(rustc --version 2> /dev/null)"
        [ -f target/mu/adjoint_week_regen ] && echo "adjoint_week: $(sha256sum target/mu/adjoint_week_regen | cut -c1-16)"
        echo "configs:"
        sha256sum configs/atlas_w21_tabulated.toml configs/mu_test/mu2_s.toml configs/mu_test/mu2_full.toml 2> /dev/null
    } > "$OUT/provenance.txt"
}

step_build() {
    cargo build --release --bins
    cargo build --release --example adjoint_week --features autodiff
    mkdir -p target/mu
    cp target/release/examples/adjoint_week target/mu/adjoint_week_regen
    # noms attendus par remodel_chain.py, paper_chain.sh et full_chain.sh
    for n in core law2 lawk v8; do ln -sf adjoint_week_regen "target/mu/adjoint_week_$n"; done
    provenance
}

step_tab() {
    target/release/gyroxide run --config configs/atlas_w21_tabulated.toml --mesh data/meshes/w21_left_610k.mesh \
        --steps 43845 --surface-every 8769 --out out/atlas_w21_tab > "$OUT/tab.log" 2>&1
}

step_remodel() {
    python3 tools/remodel_chain.py out/remodel_a0 --alpha 0 --core --w1 36 --resume "${RESUME:-0}" \
        > "$OUT/remodel.log" 2>&1
    # shellcheck disable=SC2046
    python3 tools/make_full.py --out data/full --weeks $(seq 22 36) --full 'out/remodel_a0/w{w}/eq/x_released.npy'
}

step_paper() {
    tools/paper_chain.sh "${FROM:-1}" > "$OUT/paper.log" 2>&1
    # durée par semaine : du premier au dernier fichier du dossier de chaque ajustement
    : > "$OUT/duree_semaines.tsv"
    for w in $(seq 22 36); do
        d=out/paper/fit${w}_lbm
        [ -d "$d" ] || continue
        a=$(find "$d" -type f -printf '%T@\n' | sort -n | head -1)
        b=$(find "$d" -type f -printf '%T@\n' | sort -n | tail -1)
        awk -v w="$w" -v a="$a" -v b="$b" 'BEGIN { printf "%d\t%.0f\n", w, b - a }' >> "$OUT/duree_semaines.tsv"
    done
}

step_eval() {
    mkdir -p out/scorecard/t1
    BIN=target/release/gyroxide
    for w in $WEEKS; do
        $BIN eval "out/remodel_a0/w$w/eq/x_released.npy" --week "$w" --label measured \
            --atlas-npy 'data/full/prep/w{w}/target.npy' --mesh-name w21_left_610k_s --out "out/scorecard/t1/measured_$w.json"
        $BIN eval "out/paper/fit${w}_lbm" --week "$w" --label estimated \
            --atlas-npy 'data/full/prep/w{w}/target.npy' --mesh-name w21_left_610k_s --out "out/scorecard/t1/estimated_$w.json"
    done
    $BIN eval --persistence --weeks 26-36 --atlas-npy 'data/full/prep/w{w}/target.npy' \
        --mesh-name w21_left_610k_s --out out/scorecard/t1/persistence
    python3 tools/scorecard_table.py out/scorecard/t1 --out out/scorecard/t1/table
}

step_check() {
    : "${TABLE1_METRICS:?TABLE1_METRICS : chemin de table_metrics.json (article ISBI)}"
    python3 tools/table1_check.py out/scorecard/t1 "$TABLE1_METRICS"
}

for s in $STEPS; do
    case $s in
    build | tab | remodel | paper | eval | check) echo "== $s"; chrono "$s" "step_$s" ;;
    *) echo "étape inconnue : $s (build tab remodel paper eval check)" >&2; exit 2 ;;
    esac
done
