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

**Non fait sur la PGX** : les runs de la Table 1 (`out/remodel_a0`, `out/paper/fit*_lbm`, `data/full/prep`,
maillage `w21_left_610k_s`) ne sont ni sur la PGX ni dans le bucket `dev-gyroxide` (seul `out/atlas_fit/` y est).
Pour les amener sur la PGX par Garage (seuls les fichiers nécessaires : positions finales, cibles à l'échelle,
maillage `_s`), `tools/table1_runs.sh push` sur la machine qui a les runs, puis `tools/table1_runs.sh pull` ici ;
`list` montre ce qui serait transféré, `pull` termine par la liste des fichiers encore absents. Puis, avec
`--atlas-npy` et `--mesh-name w21_left_610k_s` :

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
```

à comparer à `table_metrics.json` (colonnes : `dev_vertex_rms_mm` = RMS, `dev_vertex_p95_mm` = p95,
`curvature_corr` = corr, `fold_depth_mm.model` / `.atlas` = profondeur, `folding_index.ratio` = indice de
plissement). Écarts attendus, à vérifier : (1) `scorecard` lit pour un dossier `x_stress.npy` avant les
`x_<k>.npy`, alors que `paper_table.py` prend le plus grand `x_<k>` (`inputs.positions` dit quel fichier a été
lu : passer le `.npy` en cas de doute) ; (2) le masque du cortex vient de `w21_left_610k_s.gr.txt` (supposé
identique à `data/full/prep/w<w>/cortex.npy`, dont `paper_table.py` prend celui de SG 30 pour toutes les semaines).

### Outils touchés

Pour rendre les briques importables sans effet de bord, sans changer aucune définition : `surface_metrics.py`
(nouveau : `Surf`, `rigid_to`, `metrics` de `model_gallery.py`, la boucle d'échelles de `deviation_mm.py`,
`folding_index` de `paper_table.py`) ; `mota_law.model_T` extraite de `main()` ; import de `full_figure`
(matplotlib) rendu paresseux dans `fold_standard.main()`.
