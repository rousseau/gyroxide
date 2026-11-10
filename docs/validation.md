# Validation

## Tests unitaires (`cargo test`)

Tous en f64 sur le backend NdArray :

- maillage non déformé, sans croissance → forces et énergie nulles ;
- un tétraèdre placé exactement à G·X → forces nulles (croissance libre) ;
- tétraèdre retourné → forces finies, sommet rappelé, somme des forces nulle ;
- décomposition propre symétrique (cas aléatoires et dégénérés) ;
- continuité des forces au passage du plancher λ_min = ε² entre les deux branches ;
- forces de contact du GPU = évaluation scalaire de `closestPointTriangle` ;
- grille du plus proche nœud de surface = force brute ; NPY aller-retour ;
  Δt et nombre de pas des préréglages.

## Traces de référence (hook)

`Simulation::run(max_steps, &mut sink)` appelle un `TraceSink` aux pas qu'il
demande. `NpyTraceWriter` (crate `gyroxide-validate`) écrit :

```
trace/meta.json
trace/diagnostics.jsonl          step, t, Uk, Ue, aire, volume, étendues, ninv
trace/step_0000500/x.npy         positions   [N,3] f64
trace/step_0000500/v.npy         vitesses    [N,3] f64
trace/step_0000500/f.npy         force totale [N,3] f64, avant amortissement
```

L'état enregistré est celui du *début* du pas (forces assemblées, mise à jour
de Newton pas encore appliquée), comme le bloc de sortie du code C. Indices de
nœuds 0-based, dans l'ordre du fichier `.mesh`.

```
gyroxide run --preset tallinen2016 --mesh sphere5.mesh --backend ndarray \
             --steps 2000 --trace-every 500 --out out/s5
gyroxide compare out/s5/trace <dossier de référence> \
             --ref-positions 'pos_{step:07}.npy' --ref-forces 'force_{step:07}.npy'
```

`compare` donne, pour chaque pas commun, l'écart max, RMS et relatif des
positions et des forces. Les motifs `--ref-*` s'adaptent à la disposition des
traces du code C instrumenté (fil « Analyse des codes et des modèles ») ; il
suffira de fixer ces motifs une fois leur format connu.

`Simulation::set_state` permet aussi de repartir d'un état de l'oracle (x, v,
`growth_prev`) pour comparer un seul pas.

## Résultats actuels

- Boule voxelisée (1 680 tétraèdres), BrainGrowth, 1 000 pas : GPU f32 et CPU
  f64 identiques à 6,6e-7 près sur les positions.
- sphere5 (35 497 tétraèdres), BrainGrowth, 500 pas : écart GPU/CPU de 2,3e-6
  sur les positions.
- sphere5, BrainGrowth jusqu'à t = 1 sur GPU (44 722 pas, 227 s, 5,1 ms/pas) :
  avec le plan médian du code d'origine, divergence vers t = 0,49 (un nœud de
  surface traverse le plan et s'emballe) ; avec `--midplane-restoring`, la
  simulation va jusqu'au bout, l'aire passe de 8,63 à 20,8 et un seul élément
  est inversé à la fin.
- **Contre le code C** (traces locales `data/traces/tallinen/sphere5_braingrowth`, hors Git ;
  harnais et README dans `reference/tallinen/`,
  plan médian désactivé, maillage lu depuis `static/` avec `--ref-static`) :
  au pas 1000, CPU f64 = code C à 1,5e-12 près sur les positions et 8e-14 sur
  les forces ; GPU f32 à 3,2e-6 près sur les positions.

```
gyroxide run --preset braingrowth --ref-static <run> --no-midplane --backend ndarray \
             --steps 1001 --trace-every 1000 --out out/cmp
gyroxide compare out/cmp/trace <run> \
             --ref-positions 'step_{step:07}/x.npy' --ref-forces 'step_{step:07}/f_total.npy'
```

## Cerveau week23 (maillage de l'article)

- `week23-3M` (2 782 075 tétraèdres), préréglage `tallinen2016`, comparé au code
  C : écart de 4e-6 sur les positions après 500 pas (GPU f32).
- Avec le plan médian d'origine : stable au moins jusqu'à t = 0,034 (6 000 pas).
- Jusqu'à t = 1 avec `--midplane-restoring` : 178 886 pas en 4 h 13 (85 ms/pas,
  Mac, Metal), aire 8,5 → 40,8, 4 éléments inversés à la fin.
- Un remaillage plus grossier de la même surface (314 000 tétraèdres,
  `tools/stl_to_mesh.py`) plisse aussi mais avec des gyri trop larges : la
  résolution volumique de l'article est nécessaire.

```
gyroxide run --preset tallinen2016 --ref-static <trace week23> --midplane-restoring \
             --vtk-every 17888 --checkpoint-every 10000 --out out/week23_3M
python3 tools/render_surfaces.py week23.png out/week23_3M/surface_0000000.vtk \
             out/week23_3M/surface_final.vtk --labels "t = 0" "t = 1"
```

`--checkpoint-every N` écrit un point de reprise ; `--resume <dossier>` repart
de là (reprise vérifiée exacte au bit près).

## Tableau de bord d'un run (`gyroxide eval`)

Un point d'entrée unique écrit le tableau de bord du plan de travail pour n'importe quel run, atlas ou
sujet, et sa ligne de référence de persistance. Tout changement de modèle se juge sur ce tableau.

### Commandes

```sh
# un run : dossier (x_stress.npy, checkpoint/x.npy, x.npy, x_<k>.npy, ou run BIDS avec *_pial.surf.gii),
# .npy de positions ou surface .surf.gii ; la semaine est lue dans le run (state.json, sequence.tsv) sinon --week
gyroxide eval out/atlas_fit/w30 --week 30 --label tallinen          # écrit out/atlas_fit/w30/scorecard.json
gyroxide eval derivatives/gyroxide/sub-X/ses-Y/anat/run-essai --week 30 --units mm

# persistance : atlas w-1 aligné rigidement contre l'atlas w, SG 22 à 36 (un fichier par semaine)
gyroxide eval --persistence --out out/scorecard/persistence

# tableau Markdown + CSV + figure (une courbe par métrique, persistance en pointillé)
python3 tools/scorecard_table.py out/scorecard --out out/scorecard/tableau

# deux scorecard.json égaux à 1e-6 près (code de sortie 1 sinon) : deux exécutions, deux machines
gyroxide eval --compare a.json b.json
```

Prérequis : `python3` avec numpy, scipy, nibabel (matplotlib pour la figure) ; `GYROXIDE_PYTHON` choisit
l'interpréteur, `GYROXIDE_TOOLS` le dossier `tools/`. Données : surfaces de l'atlas dHCP (`$OXIDE_DHCP_ATLAS`,
défaut `data/atlas_dhcp/`) et maillages de l'atlas (masque du cortex, épaisseurs, tétraèdres : `data/meshes/`,
`rclone copy garage:dev-gyroxide/data/meshes data/meshes`).

**Pourquoi Python.** `gyroxide eval` est une sous-commande Rust mince qui lance `tools/scorecard.py`. Les
métriques sont celles des tables de l'article (Laplacien cotangent, bandes de courbure par noyau de chaleur,
enveloppe convexe, Mota) : elles existent déjà dans `tools/` et serviront aussi aux figures. Les recoder en Rust
ferait deux définitions à tenir d'accord ; une seule, appelée de partout, est ce qui garantit que le tableau et
l'article parlent de la même grandeur. Le prix : `python3` et trois paquets sur la machine qui évalue (la CI les
installe).

### Ce que `scorecard.json` contient

Pour chaque grandeur : valeur du modèle, de l'atlas de la même semaine, rapport modèle / atlas. Les écarts sont
calculés sur le cortex (masque de croissance du maillage, mur médian exclu). Aucune métrique existante n'a changé
de définition ; les fonctions ont seulement été déplacées pour être importables (voir plus bas).

| clé | définition | source (inchangée) |
|---|---|---|
| `dev_rigid_mm.{s200,s50,vertex}` | RMS et p95 de \|S^k x − S^k y\| (mm) après la meilleure rotation + translation, sans échelle, recalculée à chaque échelle ; S = moyenne sur les voisins, k = 200, 50, 0 | `deviation_mm.py` → `surface_metrics.deviation_levels` |
| `curvature_corr` | corrélation de Pearson de la courbure moyenne lissée (S²) avec l'atlas, sur le cortex | `model_gallery.metrics` |
| `fold_depth_mm` | p95 − p5 de la profondeur le long des normales de la surface S³⁰, sur le cortex | `model_gallery.metrics` |
| `area_mm2` | aire du cortex (aires de sommets) | `mota_law.vertex_areas`, `fold_quick.py` |
| `gi` | IG = aire de la surface entière / aire de son enveloppe convexe | `fold_standard.py` |
| `folding_index` | A / V^(2/3) de la surface entière (colonne « folding index » de la Table 1) | `paper_table.py` → `surface_metrics.folding_index` |
| `thickness_mm` | convention de `mota_law.py` : volume de la couche corticale (tétraèdres pondérés par g) / aire externe, étalonné sur SG 21 (× 1,079 ; 1,320 mm contre 1,425 mm) ; atlas : épaisseur pial-blanc mesurée, pondérée par l'aire | `mota_law.model_T` |
| `curvature_bands` | puissance de la courbure par bandes 1-2, 2-4, 4-8, 8-16 mm (H filtrée à l'échelle s moins à 2s, noyau de chaleur) | `fold_standard.py` |
| `diagnostics.n_inverted` | tétraèdres dont le volume signé a changé de signe depuis le maillage de référence, **recompté** sur les positions | `scorecard.count_inverted` |
| `diagnostics.residual` | lu dans les journaux du run, jamais recalculé : `trace/diagnostics.jsonl`, sinon dernier tableau de `run.log` (énergie cinétique Uk, énergie élastique, éléments retournés *au dernier pas journalisé*), sinon `sequence.tsv` ; `force_residual` si `run.log` contient « released: \|F\| a -> b » (équilibre `adjoint_week`) | |

Deux « IG » coexistent : `gi` (enveloppe convexe, celui de `sequence.tsv` et de `fold_standard.py`) et
`folding_index` (A / V^(2/3), celui de la Table 1). Les deux sont donnés ; ne pas les confondre.
`n_inverted` (positions finales) peut différer du `ninv` du journal (dernier pas journalisé) : à SG 30,
1 contre 0 sur `out/atlas_fit/w30`.

**Cible.** Par défaut, la surface piale dHCP de la semaine (repère de calcul : x' = −(y − cy), y' = x − cx,
z' = −(z − cz), qui est un miroir : un run dans le repère de l'atlas brut ne s'aligne pas par rotation). Si la
surface a les sommets de l'atlas (même nombre, mêmes triangles), les métriques sommet à sommet (écarts,
corrélation) sont calculées ; sinon (sujet) elles valent `null` et seules les grandeurs intrinsèques sont
comparées à l'atlas, avec le masque du cortex fourni par `--cortex-mask` (sans lui, tous les sommets des deux
côtés : `inputs.mask = "all"`). L'épaisseur et les éléments retournés demandent le maillage tétraédrique de
l'atlas (runs sur le maillage de l'atlas) ; sans lui, `null`.

**Différence avec la Table 1 de l'article ISBI.** `tools/make_full.py` écrit, pour chaque semaine, une cible
`data/full/prep/w<w>/target.npy` qui est l'atlas **mis à l'échelle par similitude** sur le cortex de l'état du
modèle (« la perte voit les plis, pas la croissance globale »). `paper_table.py` évalue contre cette cible. Le
défaut de `gyroxide eval` est l'atlas brut, rotation + translation seules (définition de `deviation_mm.py`) :
l'erreur de taille globale compte. Même run SG 30 (`out/atlas_fit/w30`), même code, deux cibles : écart rigide
aux sommets 6,01 mm (atlas brut) contre 2,56 mm (atlas mis à l'échelle, facteur 0,788). Pour rejouer la Table 1,
`--atlas-npy 'data/full/prep/w{w}/target.npy'` prend ces cibles (la ligne « atlas w−1 » de la Table 1 est alors
l'écart entre deux cibles à l'échelle de deux états différents, pas entre deux atlas bruts).

### Persistance (ligne de référence de chaque tableau)

L'atlas de la semaine w − 1, aligné rigidement, contre l'atlas de la semaine w, dans les mêmes métriques
(`reference/scorecard/persistence_sg<w>.json`, atlas brut, maillage `w21_left_610k`). RMS en mm ; rapports
« atlas w − 1 / atlas w ».

| SG | S²⁰⁰ | S⁵⁰ | sommets (p95) | corr. | profondeur | aire | IG | épaisseur | bandes 1-2 / 2-4 / 4-8 / 8-16 |
|---|---|---|---|---|---|---|---|---|---|
| 22 | 0.15 | 0.15 | 0.16 (0.24) | 0.998 | 0.984 | 0.985 | 1.000 | 0.999 | 1.00 / 1.00 / 1.01 / 1.00 |
| 23 | 0.36 | 0.37 | 0.38 (0.58) | 0.993 | 0.964 | 0.963 | 0.999 | 0.996 | 0.99 / 0.99 / 1.01 / 0.99 |
| 24 | 0.64 | 0.66 | 0.67 (1.02) | 0.982 | 0.931 | 0.936 | 0.997 | 0.990 | 0.95 / 0.97 / 1.00 / 0.99 |
| 25 | 0.76 | 0.79 | 0.80 (1.25) | 0.973 | 0.918 | 0.923 | 0.994 | 0.987 | 0.89 / 0.93 / 0.96 / 0.97 |
| 26 | 0.95 | 0.99 | 1.02 (1.60) | 0.976 | 0.894 | 0.901 | 0.990 | 0.989 | 0.77 / 0.83 / 0.89 / 0.92 |
| 27 | 1.19 | 1.24 | 1.28 (2.03) | 0.957 | 0.850 | 0.879 | 0.984 | 1.026 | 0.69 / 0.74 / 0.86 / 0.91 |
| 28 | 0.87 | 0.93 | 0.98 (1.53) | 0.962 | 0.862 | 0.902 | 0.978 | 1.052 | 0.75 / 0.74 / 0.83 / 0.95 |
| 29 | 0.89 | 0.95 | 1.01 (1.57) | 0.957 | 0.861 | 0.899 | 0.974 | 1.042 | 0.87 / 0.81 / 0.82 / 0.96 |
| 30 | 0.86 | 0.92 | 0.99 (1.58) | 0.961 | 0.850 | 0.892 | 0.965 | 1.030 | 0.87 / 0.82 / 0.80 / 0.94 |
| 31 | 0.71 | 0.78 | 0.86 (1.36) | 0.960 | 0.857 | 0.904 | 0.962 | 1.047 | 0.91 / 0.87 / 0.83 / 0.95 |
| 32 | 0.66 | 0.71 | 0.78 (1.26) | 0.968 | 0.896 | 0.921 | 0.970 | 1.033 | 0.91 / 0.92 / 0.90 / 0.96 |
| 33 | 0.64 | 0.69 | 0.77 (1.26) | 0.964 | 0.918 | 0.929 | 0.975 | 1.025 | 0.92 / 0.94 / 0.94 / 0.97 |
| 34 | 0.54 | 0.59 | 0.68 (1.12) | 0.967 | 0.933 | 0.941 | 0.978 | 1.025 | 0.95 / 0.96 / 0.95 / 0.98 |
| 35 | 0.44 | 0.47 | 0.53 (0.88) | 0.982 | 0.952 | 0.956 | 0.984 | 1.019 | 0.99 / 0.98 / 0.97 / 0.98 |
| 36 | 0.31 | 0.33 | 0.39 (0.66) | 0.986 | 0.965 | 0.969 | 0.989 | 1.007 | 1.00 / 0.98 / 0.98 / 0.98 |

Lecture : un modèle n'apporte quelque chose que s'il fait mieux que « l'atlas de la semaine d'avant ». La
persistance ne vaut pas 1 en profondeur et en aire (0,85 à SG 27-30) : l'atlas lui-même plisse vite à ces
semaines ; les bandes 1-2 et 2-4 mm tombent à 0,7 à SG 27.

### Validation de l'outil

- **Recoupement, outil par outil** (run `out/atlas_fit/w30`, mêmes entrées) : écart rigide S²⁰⁰ / S⁵⁰ / sommets =
  `deviation_mm.json` à 4e-15 ; corrélation de courbure, profondeur, rapport d'aire, rapport d'IG et rapports de
  bandes = `model_gallery.metrics`, `vertex_areas`, `fold_standard.Surface` à 1e-9 (arrondi de la mise en
  repère) ; épaisseur : étalonnage SG 21 de `mota_law.py` retrouvé (1,320 / 1,425 mm, × 1,079 ; atlas 1,43 →
  1,10 mm de SG 21 à 36). La surface piale GIfTI de l'atlas est identique bit à bit à `target.npy` de
  `adjoint_prep`.
- **Logique d'une cellule de `paper_table.py`** rejouée avec ses propres fonctions sur `out/atlas_fit` (SG 30,
  cible mise à l'échelle comme `make_full.py`) : RMS, p95, corrélation, profondeur, rapport d'indice de plissement
  et persistance coïncident avec `scorecard.py` à 1e-9 près au pire.
- **Déterminisme** : `scorecard.json` octet à octet identique entre deux exécutions et avec 1 ou plusieurs fils
  BLAS (SG 30, et les 15 fichiers de `reference/scorecard/` régénérés). Aucune date ni chemin dans `metrics` ;
  `inputs` (chemins) est ignoré par `--compare`.
- **Non-régression des outils déplacés** : `deviation_mm.json` de `out/atlas_fit` identique octet à octet avant et
  après le déplacement de la boucle dans `surface_metrics.py`.
- **CI** : `tools/tests/test_scorecard.py` (sphère à plis de forme connue : surface identique → écarts nuls,
  corrélation 1 ; invariance aux mouvements rigides ; ordre des écarts selon l'amplitude des plis ; maillage
  différent → métriques intrinsèques seules ; coque de tétraèdres → épaisseur étalonnée = 1 à SG 21, éléments
  retournés comptés ; persistance, tableau, `--compare`, `--atlas-npy`) et `crates/gyroxide-cli/tests/eval.rs`
  (vraie boule `run --bids`, `eval`, deux exécutions identiques, persistance).

### Reproduire entre machines

Sur chaque machine : `rclone copy garage:dev-gyroxide/data/meshes data/meshes`, puis
`gyroxide eval --persistence --out /tmp/p` et, pour chaque semaine,
`gyroxide eval --compare reference/scorecard/persistence_sg$w.json /tmp/p/persistence_sg$w.json`.
**Pas encore fait hors de la PGX** (aarch64, Ubuntu) : le Mac et la RTX n'étaient pas joignables au moment
de l'écriture.

### Rejouer la Table 1 de l'article ISBI

**Fait sur la PGX le 8 novembre 2026, en régénérant les runs** (le transfert par Garage, `tools/table1_runs.sh`,
est resté possible mais les fichiers n'étaient pas dans le bucket) : maillage `w21_left_610k_s`
(`split_boundary_tets.py`), `adjoint_prep_s`, croissance mesurée `out/remodel_a0` (SG 22 à 36, 18 cœurs
partagés avec d'autres sessions : la boucle CPU est le facteur limitant), puis `tools/paper_chain.sh` (inverse
ajusté chaque semaine, SG 22 à 36, environ 15 min par semaine). Entrées de `paper_table.py` :
`tools/make_full.py --out data/full --weeks $(seq 22 36) --full 'out/remodel_a0/w{w}/eq/x_released.npy'`.

```sh
for w in 26 28 30 33 36; do
  gyroxide eval "out/remodel_a0/w$w/eq/x_released.npy" --week $w --label measured \
      --atlas-npy 'data/full/prep/w{w}/target.npy' --mesh-name w21_left_610k_s --out out/scorecard/t1/measured_$w.json
  gyroxide eval out/paper/fit${w}_lbm --week $w --label estimated \
      --atlas-npy 'data/full/prep/w{w}/target.npy' --mesh-name w21_left_610k_s --out out/scorecard/t1/estimated_$w.json
done
gyroxide eval --persistence --weeks 26-36 --atlas-npy 'data/full/prep/w{w}/target.npy' \
    --mesh-name w21_left_610k_s --out out/scorecard/t1/persistence
python3 tools/scorecard_table.py out/scorecard/t1 --out out/scorecard/t1/table
python3 tools/table1_check.py out/scorecard/t1 <table_metrics.json>     # écart relatif par cellule
```

**Outil contre `paper_table.py`** : sur les mêmes runs, mêmes valeurs à 1e-9 (RMS, corrélation, profondeur,
indice de plissement, SG 36 estimé compris). Les écarts ci-dessous viennent donc des runs, pas de l'outil.

**Résultats** (60 cellules : RMS, p95, corrélation, profondeur, indice de plissement ; persistance RMS et p95) :

- **Croissance mesurée** : 24 cellules sur 25 à 0,2 % ou moins ; SG 28 RMS 1,193 contre 1,172 (+1,85 %, p95
  +0,7 %), reste de l'écart du transitoire de `remodel_a0` à SG 27-28 (+1,1 % et +1,8 % à la régénération).
- **Croissance estimée, SG 26 à 30** : RMS 0,343 / 0,687 / 1,136 contre 0,345 / 0,693 / 1,147 (≤ 1 %) ; autres
  colonnes ≤ 0,5 %.
- **Croissance estimée, SG 33** : RMS 1,833 contre 1,839, p95 +0,5 %, mais corrélation 0,845 contre 0,859
  (−1,7 %) et profondeur 1,407 contre 1,434 (−1,8 %).
- **Croissance estimée, SG 36 : hors des 1 %** (RMS 2,152 contre 2,096, corrélation 0,827 contre 0,874,
  profondeur 1,535 contre 1,677). Cause identifiée : la Table 1 de l'article lit SG 36 dans
  `out/full/fit36_lbmf` (chaîne lobaire antérieure, `fitted()` de `paper_figure.py` ne prend la chaîne
  `paper_chain` que si elle est finie), pas dans `out/paper/fit36_lbm`. La chaîne `paper_chain` finie donne un
  SG 36 plus lissé que la ligne publiée. À trancher pour l'article : régénérer la ligne avec la chaîne finie, ou
  garder `lbmf` et le dire. SG 33 (écart de 1,7 à 1,8 % sur deux colonnes) : non reproductible à 1 % à partir de
  la chaîne régénérée ; causes possibles, non isolées : GPU et pilote, f32, chemin de relaxation des semaines
  précédentes.
- **Persistance** : SG 28 à 36 à 0,3 % ou moins. **SG 26 : 1,008 contre 0,907 mm** (p95 1,591 contre 1,448).
  La cellule publiée mélange un atlas SG 25 brut (hypothèse : pas de `data/full/prep/w25` à l'époque, d'où le repli sur le GIfTI
  de `atlas_xyz`) et un atlas SG 26 mis à l'échelle ; ce mélange donne 0,917 mm (à 1 % près de 0,907), la
  définition cohérente (les deux à l'échelle) 1,008 mm. La cellule SG 26 de la Table 1 est donc à corriger.
- La chaîne ajustée n'améliore pas SG 31 (départ = arrivée, 1,427 mm, `--keep-best` garde le départ) ; sans
  conséquence pour la table (SG 26, 28, 30, 33, 36).
- Non rejoué : la ligne « Original growth law » (SG 26), qui lit `out/atlas_w21_tab/surface_*.vtk` (régénéré,
  mais non évalué ici).

Vérifications de configuration : `scorecard` lit pour un dossier `x_stress.npy` avant les `x_<k>.npy` ;
`paper_table.py` prend le plus grand `x_<k>` (ici aucun `x_stress.npy` dans ces dossiers, `inputs.positions`
du scorecard donne le fichier lu). Masque du cortex : `cortex.npy` est le même pour les semaines 22 à 36 de `data/full/prep` (vérifié) ; sa
concordance avec `w21_left_610k_s.gr.txt`, lu par le scorecard, n'a pas été comparée directement (les RMS
rejoués coïncident à 1e-9 avec `paper_table.py`, qui prend le masque de `cortex.npy`).

### Outils touchés

Pour rendre les briques importables sans effet de bord, sans changer aucune définition : `surface_metrics.py`
(nouveau : `Surf`, `rigid_to`, `metrics` de `model_gallery.py`, la boucle d'échelles de `deviation_mm.py`,
`folding_index` de `paper_table.py`) ; `mota_law.model_T` extraite de `main()` ; import de `full_figure`
(matplotlib) rendu paresseux dans `fold_standard.main()`.

## Vérification et convergence (P03, branche `l1-verification`, 9 novembre 2026)

Première tranche : tâches 1 et 2 de P03. Tâches 3 (convergence en maillage de l'hémisphère), 4 (indépendance au
solveur) et 5 (gradients de l'adjoint) non faites ; voir la fin de la section.

### Tests en f64 (`crates/gyroxide-sim/tests/verification.rs`, dans la CI, 2,3 s)

- **Patch test** : cube en 4³ cellules (384 tétraèdres), déformation homogène F_el = [[1,10 ; 0,20 ; −0,10], [0,05 ;
  0,90 ; 0,15], [−0,20 ; 0,10 ; 1,20]] (J = 1,135) et croissance anisotrope uniforme G = I + 0,6 T, positions x =
  F_el G X. Forces nulles aux nœuds intérieurs (< 1e−13 de l'échelle des forces de bord), somme des forces nulle
  (1e−12), contrainte de Cauchy égale dans tous les tétraèdres à μ J^(−5/3) dev B + K (J − 1) I analytique, J exact à
  3e−15, énergie totale égale à W V₀ analytique. **Exposant 2/3 corrigé (10 novembre, E36)** : `powf_scalar(2.0 /
  3.0)` (`elastic.rs`, `pow_j23`, `force_tangent`, branche inversée) arrondissait l'exposant en f32 (0,666 666 69),
  même sur le dispositif f64 ; il est remplacé par `exp(ln x · 2/3)` (fonction `pow23`). **Écart relatif au patch test :
  2e−10 → 4e−15 sur la contrainte** (arrondi), **2e−8 → moins de 1e−13 sur l'énergie** (seuils du test resserrés à
  1e−13). Effet sur un run court (boule générée de 1 680 tétraèdres, 2 000 pas, sortie au pas 1 000 ; ancien et
  nouveau binaire comparés) : positions **à 1,3e−10 près** (relatif, ndarray f64, le 10ᵉ chiffre, comme attendu) ; **à
  1,2e−7 près en wgpu f32**, de l'ordre de l'arrondi f32 (un exposant f32 sur dispositif f32 n'est pas faux, seule
  l'évaluation change). Les diagnostics imprimés (cinq chiffres) sont identiques. Les résultats publiés ne sont pas
  refaits : ils sont en f32 sur GPU, l'écart est de l'ordre du bruit d'exécution mesuré (E33).
- **Force = −gradient de l'énergie** (différences finies centrées, maillage perturbé, G anisotrope, 18 composantes) :
  écart relatif **1,8e−10** (3,4e−8 avant la correction de l'exposant ; seuil du test resserré à 1e−8). La pression nodale J̄ du code C est donc cohérente avec l'énergie.
- **Solution manufacturée** (cube, u = 0,05 (sin πX sin πY, sin πY sin πZ, sin πZ sin πX), μ = 1, K = 5, u imposé au
  bord, force volumique = −Div P calculée sur la solution exacte, équilibre par Newton-Krylov sur la tangente
  exacte, 4 itérations de Newton) :

  | h | erreur max | erreur L2 | CG |
  |---|---|---|---|
  | 1/3 | 5,58e−2 | 4,80e−2 | 32 |
  | 1/6 | 1,53e−2 | 8,83e−3 | 115 |
  | 1/12 | 3,73e−3 | 1,83e−3 | 249 |

  Ordres mesurés : **L2 2,44 puis 2,27, max 1,87 puis 2,03**, soit l'ordre 2 attendu des éléments P1 sur une solution
  lisse (seuils du test : L2 > 1,8 et max > 1,5 entre les deux maillages fins).

### Bicouche plane (`crates/gyroxide-sim/tests/bilayer.rs`, ignoré, environ 150 s)

**Théorie** (formules vérifiées dans la littérature : Allen 1969 ; Chen et Hutchinson 2004 ; Cao et Hutchinson 2012,
« Wrinkling phenomena in neo-Hookean film/substrate bilayers », J. Appl. Mech. 79, 031019, que j'ai retrouvé sans y
lire la formule même, la page citant seulement Allen par les modules de déformation plane) : film mince sur substrat
épais, λ = 2π H (Ē_f / 3 Ē_s)^(1/3) et ε_c = ¼ (3 Ē_s / Ē_f)^(2/3), Ē = E / (1 − ν²) module de déformation plane ;
pour deux matériaux incompressibles Ē = 4 μ et l'on retrouve λ = 2π H (μ_f / 3 μ_s)^(1/3). Ici K = 5 μ_g : le film
est compressible (ν = 0,406, Ē_f = 3,37 μ_g), on prend les Ē du matériau du code. Pour μ_f/μ_s = 50 : Ē_f/Ē_s = 42,3,
λ = 15,18 H, ε_c = 0,0429 (croissance critique s_c = 0,0448).

**Méthode** : brique d'une cellule d'épaisseur (déformation plane, u_y = 0), rouleaux sur les côtés, bas encastré à
0,8 λ, croissance selon x seulement (`forces_tensor`, G = I + g_m s diag(1,0,0)) ; état plat calculé sur une colonne
(Newton-Krylov) puis recopié ; la croissance critique est celle où la plus petite valeur propre de K = −dF/dx (Lanczos
sur K par différences finies centrées de la force, f64) change de signe (bissection) ; domaine de longueur L : le premier
mode cos(π x / L) a pour longueur d'onde 2 L ; on balaie λ/λ_th = 0,8 ; 1 ; 1,25 et on ajuste une parabole.
Maillage grossier : 3 éléments par H dans le film, 0,5 H en x, 20 couches sous le film.

| μ_f/μ_s = 50, maillage grossier | s_c (λ/λ_th = 0,8 ; 1 ; 1,25) | λ* | ε_c |
|---|---|---|---|
| théorie (bicouche net) | | 15,18 H | 0,0429 |
| transition de la loi du code (sigmoïde, 0,4 H entre 10 et 90 %) | 0,0728 ; 0,0643 ; 0,0634 | 17,4 H (+14,6 %) | 0,0591 (+38 %) |
| transition resserrée 10 fois (0,04 H, d' = H + 10 (d − H)) | 0,0281 ; 0,0265 ; 0,0288 | 14,9 H (−2,0 %) | 0,0258 (−40 %) |

**Lecture, critère non tenu en l'état** : avec une transition nette, le premier flambage a la longueur d'onde
théorique (−2 %, dans les 5 %) ; avec la loi de cortex du code, 14,6 % au-dessus, parce que la transition graduelle
raidit la surface du substrat (μ à 0,12 H sous le film : 7 fois μ_s). La croissance critique n'est pas à 5 % dans
aucun des deux cas : +38 % avec la loi du code, −40 % avec la transition nette, à ce maillage ; l'écart du cas net
est une erreur de discrétisation à 3 éléments par H (la transition de 0,04 H est plus fine que le maillage), pas
une mesure de la loi. **Convergence en maillage non atteinte** : sur la loi du code, 3 → 6 éléments par H (h_x de
0,5 à 0,25 H) donne s_c = 0,0643 → 0,0600 (Lanczos à 400 pas) → 0,0573 (800 pas, encore non convergé : le spectre de K
est dense, la plus petite valeur propre converge lentement quand le maillage s'affine). Il faut un solveur propre
meilleur (inverse itération avec préconditionneur, ou factorisation creuse de K, que le dépôt n'a pas) pour conclure
à la résolution de travail ; non fait. Cas non balayés : μ_f/μ_s = 20 et 100 (la même fonction `scan`).

### Indépendance au solveur (tâche 4, E36, 10 novembre)

**Explicite contre quasi statique, SG 21 → 24.** Chaîne explicite de `gyroxide run` (hémisphère `w21_left_610k_s`,
configuration `mu_test/mu2_s.toml`, 8 769 pas par semaine, GPU wgpu f32) sur trois semaines, avec trois réglages :
référence (pas = 0,025 sqrt(ρ a²/K), amortissement 0,5), pas / 2 (17 538 pas par semaine) et amortissement × 2 (1,0).
Équilibres quasi statiques de la chaîne `remodel_a0` (Newton, `adjoint_week`, mêmes tables) en regard. Écart de position
des sommets de surface, en mm (moyenne / RMS / p95 / max), fin de semaine :

| SG | pas / 2 contre référence | amortissement × 2 contre référence | référence contre quasi statique |
|---|---|---|---|
| 22 | 0,0005 / 0,0006 / 0,0012 / 0,0019 | 0,016 / 0,017 / 0,023 / 0,032 | 0,044 / 0,047 / 0,075 / 0,097 |
| 23 | 0,0007 / 0,0008 / 0,0014 / 0,0020 | 0,055 / 0,058 / 0,086 / 0,113 | 0,150 / 0,165 / 0,280 / 0,358 |
| 24 | 0,0009 / 0,0010 / 0,0018 / 0,0024 | 0,121 / 0,128 / 0,200 / 0,260 | 0,346 / 0,386 / 0,651 / 0,825 |

Le pas est sans effet (1 µm à SG 24, soit 0,3 % de l'écart au quasi statique). L'amortissement double l'écart au
quasi statique de +30 % (SG 24 : RMS 0,386 → 0,498 mm) : l'explicite est en retard sur l'équilibre, et d'autant plus que
l'amortissement est grand ; le retard grandit avec la croissance (0,05, 0,16, 0,39 mm) et non avec le pas. Tableau de
bord à SG 24 (`gyroxide eval`) : écart rigide aux sommets 0,43 mm (référence et pas / 2, identiques à la précision
affichée), 0,53 mm (amortissement × 2), 0,20 mm (quasi statique) ; corrélation de courbure 0,950 dans les quatre cas ;
profondeur ×0,949 / ×0,949 / ×0,945 / ×0,948. **L'écart de l'explicite au quasi statique (0,39 mm RMS à SG 24) tient
à l'amortissement, pas à la discrétisation en temps ; les métriques de plissement n'en dépendent pas** (à SG 24, avant
les plis profonds : des semaines plus tardives, où le retard peut se reporter sur l'instabilité, ne sont pas testées).

**f32 contre f64.** Même configuration, hémisphère entier (3 M de tétraèdres), 2 001 pas depuis SG 21 (t = 0,0152 de la
chaîne, soit 0,23 semaine), wgpu f32 (GPU) contre `--backend ndarray` (CPU f64, 47 min). Écart de position (mm) :

| pas | tous les nœuds, RMS / max | région de 20 mm autour du nœud le plus déplacé, RMS / max | déplacement f64, RMS / max |
|---|---|---|---|
| 1 000 | 4,6e-5 / 2,1e-4 | 3,8e-5 / 2,0e-4 | 0,006 / 0,013 |
| 2 000 | 7,8e-5 / 3,2e-4 | 6,4e-5 / 3,2e-4 | 0,018 / 0,037 |

L'écart f32 / f64 vaut 0,4 % du déplacement (RMS) et croît avec le nombre de pas (×1,7 de 1 000 à 2 000). **Limite** : une
fenêtre de 0,23 semaine en début de chaîne, sans pli ; elle ne dit rien de l'amplification de l'écart quand les plis
s'installent (SG 26 et après), où le bruit d'exécution de la chaîne estimée (E33, 1 à 2 % sur la corrélation et la
profondeur) est de toute façon plus grand. Les calculs complets en f64 (8 769 pas par semaine à 1,4 s par pas) coûtent
3,4 h par semaine : hors de portée d'un lancement.

### Gradients de l'adjoint en f64 (`crates/gyroxide-cli/tests/adjoint_gradients.rs`, dans la CI, 4 s)

`adjoint_week` choisit son dispositif par `--backend wgpu|ndarray` (défaut `wgpu`, f32, inchangé) ; `ndarray` est le CPU
en f64. Trois options de précision ajoutées, de défaut inchangé : `--newton-rel` (tolérance relative de |F| du Newton
d'équilibre, 1e-3), `--cg-tol` (résidu relatif des CG, 1e-4) et `--check-steps` (nombre de pas de `--check`, 3). Le test
bâtit une boule synthétique (251 nœuds, 816 tétraèdres, nœuds de surface en tête, tables a, h et tenseur du cœur,
point de contrôle par `gyroxide run --backend ndarray`, cible, arêtes, faces), lance l'exemple avec `--check` et exige
que l'adjoint égale la différence finie centrée de la perte totale (équilibre recalculé à |F| < 1e-11 à chaque point) :

| terme de perte (seul) ou champ | rapport adjoint / différence finie, h = 0,5 | tolérance |
|---|---|---|
| distance des sommets (`--rigid`), a | 1,000004 (h = 1 : 1,000015 ; h = 2 : 1,000061) | 2e-4 |
| indice de plissement (`--fi-weight`) | 0,999987 | 2e-4 |
| aires (`--w-area`) | 1,000010 | 2e-4 |
| retour vers la croissance mesurée (`--alpha`) | 1,000004 | 2e-4 |
| champs c, k, r ensemble, régularisation des arêtes (`--beta`) | 0,999997 | 2e-4 |
| champs c, k, r séparément, et c multiplicatif (`--thick-mult`) | 1,000000 à 1,000004 | 2e-4 |
| normales (`--w-normal`) | 1,0004 | 5e-3 |
| laplacien (`--w-lap`) | 1,0019 (0,9992 sur la boule à 6 cubes) | 5e-3 |

Normales et laplacien ne sont pas invariants par rotation et la boule libre a un mode de rotation mou. Hypothèse, non
démontrée : l'équilibre y dérive (bruit d'arrondi amplifié), ce que l'adjoint ne voit pas ; l'écart du laplacien varie
de 1e-5 à 1e-3 d'une exécution à l'autre sur la boule à 6 cubes, d'où leur tolérance de 5e-3. Sensibilité du test : avec un décalage de K (`--shift`) de 1e-2 au lieu de 1e-12, les
rapports passent à 1,006 (indice de plissement) jusqu'à 1,059 (normales) : une régression de l'adjoint de l'ordre du
pour cent est vue par chaque test.

**Trouvaille.** Avec la perte `--absolute` (positions brutes, sans retrait du mouvement rigide), l'adjoint de la boule
libre est faux de 1 à 2 % : 1,009 avec le décalage de K par défaut (1e-3 du Rayleigh), 1,016 sans décalage. La perte
absolue n'est pas invariante par translation : son gradient par rapport aux positions a une composante dans le noyau de
K (6 modes rigides) et la solution de l'adjoint y est mal posée (le décalage la régularise, au prix de ce biais). Les
ajustements de l'article utilisent `--absolute` ; l'erreur de leurs gradients (hémisphère entier, contact, plan médian)
n'est pas mesurée ici, et on ne sait pas si ces contraintes lèvent la mal-position. `--rigid` (perte invariante) donne 1 ± 1e-5.

Le test n'est lancé qu'avec `--features autodiff` (l'exemple l'exige) : la CI passe `--features autodiff` à clippy (qui
vérifie aussi l'exemple, dont quatre lints de style sont autorisés en tête de fichier) et aux tests.

### Reste à faire pour P03

- Tâche 2 : solveur propre pour les maillages fins, et balayage de μ_f/μ_s (20, 50, 100).
- Tâche 3 : convergence en maillage de l'hémisphère (remaillage TetGen SG 21 à 0,3 / 0,6 / 1,6 M tétraèdres, au moins
  3 éléments dans l'épaisseur corticale, tables nodales transportées, chaîne de référence SG 21 → 30 par maillage, soit
  de 3 à 30 h par maillage).
- Tâche 4 : faite sur SG 21 → 24 et 0,23 semaine en f64 (ci-dessus) ; f32 contre f64 sur les semaines à plis non testé.
- Tâche 5 : faite, voir « Gradients de l'adjoint en f64 » ci-dessus.