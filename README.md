# gyroxide

Simulation du plissement cortical par croissance tangentielle (modèle de
Tallinen et al., *Nature Physics* 2016), en Rust sur [Burn](https://burn.dev),
avec tout l'état de la simulation résident sur le GPU.

Version 1 : modèle de Tallinen à l'identique (néo-Hookéen avec J̄ nodal,
branche des éléments inversés, croissance tangentielle totale, dynamique
explicite amortie, auto-contact point–triangle, plan médian), implémenté en
opérations de tenseurs Burn. Deux préréglages : `tallinen2016` (code C de
l'article) et `braingrowth`.

```sh
cargo test                                   # tests en f64 sur CPU
cargo run --release --bin gyroxide -- run --ball 8          # petite boule sur GPU (wgpu)
cargo run --release --bin gyroxide -- run --preset braingrowth \
    --mesh ~/Sync/Code/BrainGrowth/data/sphere5.mesh --surface-every 5000 --out out/s5
cargo run --release --bin gyroxide -- preset tallinen2016   # paramètres en TOML
```

Options utiles de `run` : `--backend ndarray` (CPU f64), `--steps N`,
`--surface-every N` (surfaces GIfTI), `--hemi L|R`, `--trace-every N` (traces NPY),
`--vtk-every N` (surfaces VTK de débogage, ParaView), `--no-contact`, `--no-midplane`,
`--uniform-growth`, `--config params.toml`.

## Surfaces écrites

Le calcul se fait sur le maillage normalisé (demi-taille du cerveau ~ 1). Les surfaces sont
écrites en GIfTI (`*.surf.gii`, via surfoxide-io) après la transformée inverse de la
normalisation : coordonnées RAS en mm, dans le repère de l'image d'où vient l'état initial.
Seuls les nœuds de surface sont écrits, faces orientées vers l'extérieur,
`AnatomicalStructurePrimary` = `CortexLeft` ou `CortexRight` selon `--hemi`.

Avec `normalization = "none"` (maillages déjà normalisés : maillages de Tallinen, maillage
de l'atlas `w21_left_610k`), il n'y a pas de transformée à inverser : les coordonnées
écrites sont celles du maillage, en unités normalisées, **pas en mm**. Les surfaces VTK de
`--vtk-every` restent dans le repère normalisé du calcul.

## Maillage tétraédrique d'une session

```sh
gyroxide mesh --bids <racine BIDS> --participant X [--session Y] [--hemi L|R] [--tetgen <exécutable>]
```

lit les sorties de `surfx run --bids` (`…_hemi-H_white.surf.gii`, `…_pial.surf.gii`,
`…_cortex.shape.gii`) et écrit `derivatives/gyroxide/sub-X[/ses-Y]/anat/
sub-X[_ses-Y]_hemi-H_desc-tetgen_tet.vtu` : VTK XML `UnstructuredGrid` de tétraèdres, en mm dans le
repère de l'image, avec par nœud `gray_matter` (1 substance grise, 0 substance blanche : nœud
intérieur à la surface blanche) et `cortex` (masque du cortex du sommet de piale le plus proche).
TetGen est appelé comme **programme externe** (licence AGPL : jamais lié à gyroxide) ; avec les
options par défaut (`pq1.5/10YQ`) il garde exactement les triangles de la piale, ses sommets en tête.
Prévu ensuite : isosurface stuffing (Labelle et Shewchuk 2007) en Rust dans `gyroxide-mesh`.

`run --mesh` accepte ce `.vtu` (lu par `gyroxide-mesh`) ; c'est l'entrée par défaut de
`run --bids` (`--mesh-desc` pour en choisir un autre). Un `.vtu` est en mm : il est toujours
normalisé pour le calcul (`brain_growth_whole`), et l'espacement `a` (pas de temps explicite, peaux
de contact) vaut par défaut la plus courte arête de surface, pour la stabilité.

## Un run = un dossier BIDS

```sh
gyroxide run --bids <racine BIDS> --participant X [--session Y] [--output-dir D] \
    --label <run> [--hemi L|R] [--mesh <état initial>] [--config params.toml] [--steps N]
```

Pour un hémisphère de session (état initial par défaut : le `.vtu` de `gyroxide mesh`), ajouter
**`--uniform-growth --no-midplane`** : le masque ellipsoïdal et le plan médian des préréglages
sont faits pour le maillage de Tallinen (le plan médian de `tallinen2016`, y′ = −0,004, couperait
l'hémisphère normalisé en son milieu). L'espacement `a` vaut par défaut la **plus courte arête de
surface** : les maillages TetGen gardent des tétraèdres plats contre la piale et `a` = arête
moyenne donne des NaN en quelques centaines de pas. C'est environ 4 fois plus lent ; l'isosurface
stuffing (gyroxide#5) réglera ces tétraèdres. Exemple, sur la session d'essai :

```sh
gyroxide mesh --bids /tmp/surfx-e2e/sourcedata --participant 0avbc --session 02 --hemi L --tetgen <tetgen>
gyroxide run --bids /tmp/surfx-e2e/sourcedata --participant 0avbc --session 02 --label essai \
    --hemi L --steps 20000 --uniform-growth --no-midplane
```

écrit dans `<étude>/derivatives/gyroxide/sub-X[/ses-Y]/anat/run-<label>/`, `<étude>` étant
`--output-dir` s'il est donné, sinon le parent de la racine BIDS si elle s'appelle `sourcedata/`,
sinon la racine elle-même (même règle que medoxide et surfoxide) :

- `config.toml` : paramètres complets (préréglage + surcharges) et `initial_state`, chemin de
  l'état initial relatif au dossier du run ;
- `sequence.tsv` : `step time ga_weeks file gi energy n_contacts`, une ligne ajoutée et vidée sur
  disque à chaque surface écrite (`--surface-every`, 1000 pas par défaut, plus le pas 0 et l'état
  final) ; `ga_weeks` est interpolé depuis `schedule.ga_weeks` (`n/a` sans ce paramètre),
  `gi` = aire de la surface / aire de son enveloppe convexe, `energy` = énergie élastique `Ue`,
  `n_contacts` = paires en contact (point dans la peau de répulsion d'un triangle) ;
- `status.json` : `{"state": "running" | "done" | "failed", "step", "total_steps", "started",
  "updated", "message"}` (dates ISO 8601 UTC), réécrit de façon atomique à chaque diagnostic ;
  `failed` avec le message en cas d'erreur ou d'état non fini (NaN) ;
- `sub-X[_ses-Y]_hemi-L_desc-sim_step-<pas sur 7 chiffres>_pial.surf.gii` ;
- `derivatives/gyroxide/dataset_description.json` (`GeneratedBy` : `Name`, `Version`, `Commit`,
  `CodeURL`, `CommandLine`).

Le crate `gyroxide-io` (sans Burn) lit ces fichiers pour guioxide : `lire_status`,
`lire_sequence`, `lister_runs`, et calcule `gi`.

Documentation : [modèle](docs/model.md), [architecture](docs/architecture.md),
[validation](docs/validation.md).
