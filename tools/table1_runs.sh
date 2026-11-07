#!/bin/sh
# Transfert par Garage des seuls fichiers dont le rejeu de la Table 1 ISBI a besoin (tools/paper_table.py,
# docs/validation.md) : positions finales des trois lignes de modèle, cibles à l'échelle, maillage _s.
#
#   tools/table1_runs.sh list    # la liste des fichiers présents (relatifs à la racine du dépôt)
#   tools/table1_runs.sh push    # sur la machine qui a les runs : envoie vers garage:dev-gyroxide
#   tools/table1_runs.sh pull    # sur la machine qui évalue (PGX) : rapatrie dans le dépôt
#
# Semaines de la Table 1 : 26 28 30 33 36 (WEEKS=... pour changer) et leur semaine précédente (persistance).
# Lignes : « original » (out/atlas_w21_tab, SG 26), « measured » (out/remodel_a0/w<w>/eq/x_released.npy),
# « estimated » (out/paper/fit<w>_lbm, plus grand x_<k>.npy ; à défaut la chaîne out/full/fit<w>_lbm[f]).
# REMOTE=... change la destination (défaut garage:dev-gyroxide). Rien n'est supprimé, ni ici ni sur Garage.
set -e
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-garage:dev-gyroxide}
WEEKS=${WEEKS:-"26 28 30 33 36"}

MODE=local        # pull : les dossiers de runs se lisent sur Garage, pas ici
ls_dir() { if [ "$MODE" = pull ]; then rclone lsf "$REMOTE/$1" 2> /dev/null; else ls "$1"; fi; }
has_dir() { [ "$MODE" = pull ] || [ -d "$1" ]; }

list() {
    M=w21_left_610k_s
    for f in $M.json $M.mesh $M.gr.txt $M.nodal_h.npy $M.nodal_a.npy; do echo "data/meshes/$f"; done
    for d in data/full/prep data/adjoint_prep_s; do
        has_dir "$d" || continue
        for w in $WEEKS; do
            for v in $w $((w - 1)); do
                for f in target cortex faces edges owner; do echo "$d/w$v/$f.npy"; done
            done
        done
    done
    for w in $WEEKS; do
        echo "out/remodel_a0/w$w/eq/x_released.npy"
        printf 'out/atlas_w21_tab/surface_%07d.vtk\n' $(((w - 21) * 8769))
        for run in out/paper/fit${w}_lbm out/full/fit${w}_lbm out/full/fit${w}_lbmf out/full/fit${w}_lb40; do
            has_dir "$run" || continue
            echo "$run/theta_final.npy"
            last=$(ls_dir "$run" | grep -E '^x_[0-9]+\.npy$' | sort -t_ -k2 -n | tail -1)
            [ -n "$last" ] && echo "$run/$last"
        done
    done
}

present() { list | while read -r f; do [ -f "$f" ] && echo "$f"; done; true; }

case "${1:-}" in
list) present ;;
push)
    present > /tmp/table1_files.$$
    [ -s /tmp/table1_files.$$ ] || { echo "aucun fichier de la Table 1 ici : mauvaise machine ?" >&2; exit 1; }
    echo "$(wc -l < /tmp/table1_files.$$) fichiers → $REMOTE"
    rclone copy . "$REMOTE" --files-from /tmp/table1_files.$$ --progress
    list | while read -r f; do [ -f "$f" ] || echo "absent : $f"; done
    rm -f /tmp/table1_files.$$ ;;
pull)
    MODE=pull
    # fichier par fichier : --files-from s'arrête aux fichiers absents de Garage
    list | while read -r f; do
        [ -f "$f" ] || rclone copyto "$REMOTE/$f" "$f" 2> /dev/null || true
        # copyto renvoie 0 même si la source n'existe pas : on teste le fichier
        if [ -f "$f" ]; then echo "ok      $f"; else echo "absent  $f"; fi
    done ;;
*) sed -n '2,12p' "$0"; exit 2 ;;
esac
