# Traces de référence Tallinen 2016 (code C instrumenté)

Ces traces servent d'oracle au simulateur Rust+Burn **gyroxide**. Elles sont produites par
`src/tallinen_ref.cpp`, copie instrumentée de `Brain_code_and_meshes/Brains.cpp` (Tallinen et al.,
Nature Physics 2016). L'original n'est pas modifié.

## Ce qui change par rapport à Brains.cpp

La mécanique (croissance, élasticité, branche SVD, contact, plan médian, intégration) est copiée
expression par expression. Seuls changent :

- paramètres en ligne de commande (maillage, sortie, scénario, nombre de pas) ;
- normalisation optionnelle `--normalize tallinen`, la transformation du bloc commenté
  `geom_norm.mesh` de l'original : x' = -(y-cy)/d, y' = (x-cx)/d, z' = -(z-cz)/d ;
- réorientation automatique : cette transformation est une réflexion, elle retourne tous les
  tétraèdres. Le harnais échange n3/n4 (tétraèdres) et n2/n3 (triangles) si tous les volumes sont
  négatifs ou si la surface est orientée vers l'intérieur ; c'est consigné dans `manifest.json` ;
- tableaux par nœud sur le tas (les VLA de pile de l'original débordent sur le maillage 3M) ;
- lecture des nombres avec `>>` (l'original lit dans des `char[10]`) ;
- pas d'OpenMP : l'original accumule les forces en parallèle sans atomiques (courses de données) ;
- sorties NPY/CSV/JSON au lieu de POV-Ray et Amira.

**Équivalence vérifiée :** sur la sphère normalisée, 5 001 pas du scénario actif de l'original
(`tallinen2016-unfold`), les 9 colonnes de `Brains.dat` (pas, t, Uk, Ue, aire, volume, largeurs,
ninv, toutes les 100 itérations) sont identiques au bit près entre l'original compilé sans OpenMP
et le harnais (même compilateur, `-O2`). Voir `orig/` pour le patch minimal appliqué à l'original
pour ce test (nom de fichier, nombre de pas, précision d'écriture).

## Bug du plan médian (`--midplane`)

La pénalité de plan médian de l'original (`Brains.cpp:419-424`, identique dans BrainGrowth
`geometry.py:729-731`) a le mauvais signe : un nœud de surface qui traverse y = -0.004 est poussé
plus loin, avec une force qui croît avec la distance. Sur un vrai cerveau les deux hémisphères ne
traversent presque jamais le plan et le défaut reste caché ; sur la sphère, qui est à cheval sur le
plan, le calcul diverge (`runs/sphere5_braingrowth_blowup`, explosion à partir du pas ~18 000,
t ≈ 0.4). Option `--midplane original` (défaut, fidèle à l'original) | `off` | `restoring` (signe
corrigé). Les traces de la sphère utilisent `off` ; celles de week23 utilisent `original`.

## Traces disponibles (`runs/`)

| trace | maillage | preset | plan médian | pas | dumps |
|---|---|---|---|---|---|
| `check_unfold` | sphère | tallinen2016-unfold | original | 5 001 | 0, 5000 (test d'équivalence) |
| `sphere5_tallinen2016` | sphère | tallinen2016 | off | 178 886 (t = 1) | 0,1,2,10,100,1000, tous les 10 000, final, G et J |
| `sphere5_braingrowth` | sphère | braingrowth | off | 44 722 (t = 1) | 0,1,2,10,100,1000, tous les 5 000, final, G et J |
| `sphere5_braingrowth_blowup` | sphère | braingrowth | original | 20 500 (arrêt) | divergence due au plan médian |
| `week23_tallinen2016_short` | week23-3M (article) | tallinen2016 | original | 501 | 0,1,2,10,100,500 |

Le maillage week23 n'est pas normalisé dans le zip (coordonnées en mm, 22 à 243) alors que le code
suppose un cerveau dans [-1.6, 1.6]³ : il faut `--normalize tallinen`. Le `Brains.dat` fourni dans
le zip vient d'un calcul sur ce maillage non normalisé et n'est pas une référence valable.

## Scénarios (`--preset`)

| preset | at(t) | H(t) | rho | dt | t |
|---|---|---|---|---|---|
| `tallinen2016` | 1.85 t | 0.084 - 0.042 t | 0.0025 | 0.025 sqrt(rho a²/K) | 0 → 1 |
| `tallinen2016-unfold` | repli puis dépli (bloc actif de l'original) | | 0.0025 | idem | -0.25 → 1 |
| `braingrowth` | 1.829 t | 0.042 + 0.01 t | 0.01 | 0.05 sqrt(rho a²/K) | 0 → 1 |

Paramètres communs : mu_g = 1, mu_w = 1.167, K = 5, a = 0.01, gamma = 0.5, eps = 0.1, k = 0,
contact kc = 10K, hc = 0.2a, hs = 0.6a, cellules bw = 3.2, mw = 8a, plan médian y = -0.004.

Dans gyroxide, ces traces sont dans `data/traces/tallinen/<trace>/` (ignoré par git).

## Format d'une trace

```
<run>/
  manifest.json           preset, maillage, normalisation, paramètres, dt, orientation, conventions
  scalars.csv             step,t,at,H,Uk,Ue,Area,Volume,width_x,width_y,ninv (comme Brains.dat)
  static/
    X0.npy                (nn,3) f8  positions de référence (après normalisation)
    tets.npy              (ne,4) i4  indices 0-based, ordre interne du code C (droitier)
    faces.npy             (nf,3) i4  triangles de surface, normales sortantes
    surface_nodes.npy     (nsn,) i4  SN : surface -> nœud
    nearest_surface_node.npy (nn,) i4  csn : indice *dans la surface* du nœud de surface le plus proche
    dist_to_surface.npy   (nn,) f8  d2s
    surface_normals0.npy  (nsn,3) f8 N0, normales de référence normalisées
    gr.npy                (nn,) f8  facteur de croissance (ellipsoïde sans croissance)
  step_NNNNNNN/           état au début du pas N
    x.npy, v.npy          (nn,3) positions et vitesses avant intégration
    f_contact.npy         forces après le contact
    f_after_elastic.npy   + élasticité
    f_total.npy           + plan médian (force intégrée, avant amortissement)
    Vn.npy, Vn0.npy       (nn,) volumes nodaux déformé / référence (avec G du pas précédent)
    G_used.npy, J.npy     (option --dump-tets) G (ne,3,3) utilisé à ce pas, det F (ne,)
    step.json, contact.json  t, at, H, Ue, ninv ; reconstruction de la liste de proximité, nb de paires
  final/                  x, v après le dernier pas
```

Intégration (pour reproduire un pas à partir de `step_N`) :
`F = f_total - gamma*Vn0*v ; v += F/(rho*Vn0)*dt ; x += v*dt`.
Attention : G est mis à jour à la fin de la boucle des tétraèdres, il s'applique au pas suivant.

## Reconstruire

```
g++ -O2 -std=c++17 src/eig3.cpp src/tallinen_ref.cpp -o tallinen_ref
./tallinen_ref --mesh week23-3M-tets.mesh --normalize tallinen --out runs/x --preset tallinen2016 --max-steps 501 --dump-steps 0,1,10,100,500
python3 tools/check_trace.py runs/x
```
