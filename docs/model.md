# Modèle v1 : Tallinen et al. 2016

gyroxide v1 reproduit le modèle du code C de l'article (Tallinen et al., *Nature
Physics* 12, 588, 2016 ; `Brain_code_and_meshes/Brains.cpp`), tel que repris par
BrainGrowth (Python/Numba). Toutes les grandeurs sont normalisées : longueurs en
demi-taille du cerveau, raideurs relatives à μ_g, temps adimensionnel t.

Code correspondant : `crates/gyroxide-sim/src/elastic.rs` (élasticité),
`contact.rs` (contact, plan médian), `lib.rs` (pas de temps),
`crates/gyroxide-model` (paramètres, lois de croissance).

## Équations, par tétraèdre

**Couche corticale et croissance.** d = distance du nœud à la surface de référence
(nœud de surface le plus proche), moyennée sur les 4 sommets ; gr = masque de
croissance, moyenné de même ; n = normale de croissance, somme normalisée des
normales de référence N0 des nœuds de surface les plus proches des 4 sommets
(calculée une fois).

```
g_m = gr / (1 + exp(10 (d / H(t) - 1)))
μ   = μ_w (1 - g_m) + μ_g g_m
G   = I + (I - n nᵀ) s,     s = g_m a_t   (valeur du pas précédent)
```

Comme dans le code C, G est mis à jour en fin de boucle et ne sert qu'au pas
suivant. On stocke donc un seul scalaire par tétraèdre, `growth_prev = g_m a_t`.

**Cinématique.** A_r = G · [x1−x0, x2−x0, x3−x0]_réf, A = mêmes arêtes actuelles.

```
F = A A_r⁻¹,   J = det F,   B = F Fᵀ
V_n0, V_n = volumes nodaux (¼ du volume de chaque tétraèdre voisin), référence
            grandie et actuelle ;  J̄ = moyenne des V_n / V_n0 des 4 sommets
```

**Matériau (branche régulière).** Si λ_min(FᵀF) ≥ ε² et J > 0 :

```
σ = μ J^(-5/3) (B - tr(B)/3 I) + K (J̄ - 1) I,      P = J σ F⁻ᵀ = σ adj(F)ᵀ
W = μ/2 (tr B J^(-2/3) - 3) + K/2 · moyenne des (J_i - 1)²
```

**Éléments inversés ou très comprimés.** Sinon, on diagonalise C = FᵀF
(λ₁ ≤ λ₂ ≤ λ₃, vecteurs V), U = F V diag(1/l), l = √λ, et on prolonge l'énergie
linéairement sous l'étirement plancher ε = 0.1 (formules `Pd.a`, `Pd.e`, `Pd.i`
de Brains.cpp, recopiées à l'identique) ; si det F < 0, l₁ et la première colonne
de U changent de signe. P = U diag(Pd) Vᵀ.

La diagonalisation est faite en forme close (valeurs propres de Kopp, comme
`Matrix::EV()` de vema.h, puis vecteurs propres par produits vectoriels et
problème 2×2 dans le complément orthogonal, méthode d'Eberly), pour rester en
opérations élément par élément sur le GPU. Le code C utilise une
tridiagonalisation de Householder + QL (eig3.cpp) : les vecteurs peuvent différer
d'un signe, ce qui ne change pas P.

**Forces nodales.** Avec N1 = x_r3 × x_r1, N2 = x_r2 × x_r3, N3 = x_r1 × x_r2,
N4 = (x_r2 − x_r3) × (x_r1 − x_r3) (colonnes de A_r) :

```
f0 = P (N1+N2+N3)/6,  f1 = P (N1+N3+N4)/6,  f2 = P (N2+N3+N4)/6,  f3 = P (N1+N2+N4)/6
```

**Contact.** Pénalité point de surface / triangle de surface : si la distance r au
point le plus proche est < h_c = 0.2 a et gr[p] + gr[t₁] > 0,
f = ĉ (r − h_c)/h_c · k_c a², k_c = 10 K, réfléchie si elle pointe sous le
triangle ; −f u, −f v, −f w sur les sommets du triangle (coordonnées
barycentriques). Listes de candidats à h_s = 0.6 a, cellules de 8a,
reconstruites dès qu'un nœud de surface a bougé de plus de (h_s − h_c)/2.

**Plan médian (optionnel).** Nœuds de surface partis à y < y_m − a/2 et passés à
y > y_m (ou l'inverse) : `F_y -= (y_m - y)/h_c · a² K`. Voir « Écarts » ci-dessous.

**Dynamique.** Euler semi-implicite amorti, masse ρ V_n0 :

```
f ← f - γ V_n0 v,   v ← v + f / (ρ V_n0) Δt,   x ← x + v Δt,   Δt = s √(ρ a² / K)
```

## Préréglages

| | Tallinen 2016 (`tallinen2016`) | BrainGrowth (`braingrowth`) |
|---|---|---|
| Source | Brains.cpp, scénario de plissement seul (en commentaire dans le fichier) | simulation.py, valeurs de la synthèse |
| a_t | 1.85 t | 1.829 t |
| H(t) | 0.084 − 0.042 t | 0.042 + 0.01 t |
| ρ | 0.0025 | 0.01 |
| s (Δt) | 0.025 → Δt = 5.59e-6, 178 886 pas | 0.05 → Δt = 2.24e-5, 44 722 pas |
| μ_g, μ_w, K, γ, ε | 1, 1.167, 5, 0.5, 0.1 | idem |
| a, h_c, h_s, k_c | 0.01, 0.2a, 0.6a, 10K | idem |
| Plan médian | y = −0.004 | idem |
| Masque gr | ellipsoïde de Brains.cpp | idem (`mark_nogrowth`) |
| Normalisation | aucune (maillage week23 déjà normalisé) | `normalise_coord(..., "whole")` |

`gyroxide preset <nom>` imprime le TOML complet ; `--config fichier.toml` le
remplace. Le scénario actif de Brains.cpp (plissement puis déplissement, t de
−0.25 à 1, lois par morceaux) n'est pas encore un préréglage ; la loi tabulée
ci-dessous permet de l'écrire.

Remarque : la version de BrainGrowth présente dans `~/Sync/Code/BrainGrowth` a
d'autres valeurs par défaut en ligne de commande (`--stepcontrol 0.1`,
`--massdensity 0.1`, `bounding_box = 19.2`). Le préréglage suit la synthèse
(ρ = 0.01, s = 0.05) ; à confirmer contre la version utilisée pour les traces.

## Loi de croissance tabulée

Outre la loi linéaire (`law = "linear"`, a_t = a0 + a1 t, H = h0 + h1 t), la
section `[schedule]` accepte une loi tabulée : a_t et H(t) (et, en option, μ_g
et μ_w) sont donnés en des nœuds t₀ < t₁ < … < t_{n−1}, interpolés linéairement
entre deux nœuds et gardés constants hors de [t₀, t_{n−1}].

```toml
[schedule]
law = "tabulated"
t = [0.0, 0.066667, 0.133333]      # nœuds, strictement croissants (n ≥ 2)
a = [0.0, 0.123333, 0.246667]      # a_t aux nœuds
h = [0.043958, 0.043958, 0.043958] # H aux nœuds (> 0)
mu_gray  = [1.0, 1.0, 1.0]         # optionnel (> 0), remplace [material].mu_gray
mu_white = [1.167, 1.167, 1.167]   # optionnel (> 0), remplace [material].mu_white
```

Tous les tableaux ont la longueur de `t` ; les erreurs (longueurs, t non
strictement croissant, H ou μ ≤ 0, valeurs non finies) sont signalées au
chargement. K, ρ et γ restent constants (Δt dépend de K). Comme la croissance est
totale (G recalculé à partir de a_t à chaque pas), a_t doit rester continu : un
saut de a_t entre deux nœuds rapprochés impose un saut de croissance au tissu.
Avec deux nœuds, la loi tabulée redonne la loi linéaire (à l'arrondi près) ; la
loi linéaire et les préréglages sont inchangés au bit près.

**Convention t ↔ semaine pour l'atlas dHCP.** t = (SG − 21)/15 : t = 0 à 21
semaines de gestation, t = 1 à 36 semaines, un nœud par semaine (t = k/15).
Exemple complet : `configs/atlas_w21_tabulated.toml` (a_t = 1.85 t, H = médiane
hebdomadaire de la distance piale–blanche de l'atlas / 32.986 mm).

## Loi de croissance par nœud

`law = "nodal"` donne a_t et H **par nœud**, tabulés aux mêmes nœuds temporels que la
loi tabulée et interpolés linéairement dans le temps (constants hors des nœuds).
Chaque tétraèdre prend la moyenne de ses 4 nœuds.

```toml
[schedule]
law = "nodal"
t = [0.0, 0.066667, 0.133333]                # nœuds, strictement croissants (n ≥ 2)
a = "data/meshes/w21_left_610k.nodal_a.npy"  # NPY [n_nœuds_t, n_nœuds_maillage]
h = "data/meshes/w21_left_610k.nodal_h.npy"  # idem, H > 0 (unités normalisées)
a_scale = 1.0                                # facteurs globaux (optionnels, défaut 1)
h_scale = 1.0
```

Les fichiers sont lus par le CLI (`gyroxide run`) et vérifiés au chargement
(dimensions, valeurs finies, H > 0). La loi linéaire et la loi tabulée gardent le
chemin scalaire, inchangé au bit près.

`tools/atlas_nodal_tables.py` construit ces tables à partir de l'atlas dHCP (même
triangulation à toutes les semaines) : a = √(aire locale(SG) / aire locale(21)) − 1
(aire de sommet lissée), H = distance piale–blanche lissée (plancher 0.3 mm), nœuds
intérieurs = valeur du sommet de surface le plus proche. Exemple :
`configs/atlas_w21_nodal.toml`.

## Écarts connus avec les codes de référence

- **Signe du plan médian.** Dans Brains.cpp comme dans BrainGrowth
  (`calc_mid_plane`), un nœud parti sous le plan et passé au-dessus reçoit
  `F_y -= (y_m - y)/h_c a² K`, soit une force *positive* qui l'éloigne encore du
  plan, proportionnelle à la distance : c'est un emballement exponentiel. Sur un
  cerveau, la paroi médiale ne croît pas (gr = 0) et ne traverse sans doute jamais
  le plan, donc le défaut reste latent. Sur une boule coupée par le plan, il fait
  diverger la simulation vers t = 0.7. Par défaut gyroxide reproduit le code
  (`midplane_restoring = false`) ; `--midplane-restoring` utilise le signe
  rappelant.
- **Grille de contact non bornée** (table de hachage) au lieu de la boîte fixe
  `bw = 3.2` du code C (19.2 dans BrainGrowth) : mêmes paires tant que le
  maillage reste dans la boîte.
- **Ordre des sommes.** L'assemblage par gather (liste ELL) somme les
  contributions dans un ordre fixe : résultats déterministes, alors que le code C
  multi-thread a des courses de données sur `Ft[n] +=`. Comparer au code C lancé
  en mono-thread.
- **Précision.** wgpu calcule en f32, le CPU de référence (`--backend ndarray`)
  en f64.
- **Vérification de reconstruction des listes de contact** tous les
  `check_every` pas (10 par défaut) au lieu de chaque pas, pour éviter une lecture
  GPU → CPU par pas. Avec la marge h_s − h_c = 0.4a et le seuil 0.2a, il faudrait
  une vitesse de surface de l'ordre de 0.2a / (10 Δt) (≈ 36 en unités normalisées
  pour le préréglage Tallinen) pour manquer un contact ; `check_every = 1` redonne
  le comportement exact.

## Points d'extension prévus

Les lois de croissance passent par `gyroxide_model::Schedule` (énumération
sérialisée : linéaire, tabulée, par nœud) ; le masque gr par `GrowthMask` (uniforme, ellipsoïde, fichier par
nœud). À venir, selon la synthèse : croissance régionale (Gompertz par lobe),
α_TAN issu de l'IRM, croissance radiale, loi incrémentale avec relaxation de
Mandel, normales de la surface déformée, unités physiques (Pa, mm, semaines),
configuration de référence actualisée et encastrement des zones sans croissance.
