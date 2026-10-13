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
