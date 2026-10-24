# Ajustement adjoint sur une région (sillon central)

Sous-maillage autour du sillon central (`tools/make_slab.py`, `data/slab_cs/` :
13 792 nœuds dont 2 497 fixés, 3 364 sommets de perte), atlas dHCP ramené à la
taille du modèle par une similitude. Évaluation : `tools/slab_eval.py` (écart en
mm, angle des normales, corrélation de la courbure moyenne H, profondeur des
plis = p95 − p5 le long des normales de la surface lissée). Figures :
`tools/slab_figure.py`, `tools/slab_fields.py`.

Perte composite commune : `--w-vertex 1 --w-normal 1 --w-lap 1 --w-area 0.5`,
multi-échelle `10:30,0:30`, croissance bornée à ±50 % de la mesure
(`--growth-band 0.5`).

## SG 27 (atlas : profondeur 0,64 mm)

| run | champs | mm | normales | corr H | profondeur |
|---|---|---|---|---|---|
| départ (croissance mesurée) | — | 1,258 | 19,6° | 0,43 | 0,36 |
| fit27_combo | a | 0,847 | 11,5° | 0,82 | 0,53 |
| fit27_band_c | a, c | 0,844 | 10,7° | 0,79 | 0,47 |
| fit27_band_k | a, k | 0,662 | 9,0° | 0,87 | 0,52 |
| fit27_band_ck | a, c, k (±0,3) | **0,485** | **5,3°** | **0,87** | 0,62 |
| fit27_band_ck_b3 | idem, β = 1e-3 | 0,497 | 5,5° | 0,86 | 0,63 |
| fit27_band_ck015 | a, c, k (±0,15) | 0,661 | 8,7° | 0,85 | 0,52 |

## SG 28, enchaîné depuis SG 27 (atlas : profondeur 0,91 mm)

Table de a reportée (`configs/mu_test/mu2_slab_cs_carry_*.toml`), état de départ
= x_fit27 + (complet28 − complet27), `--relax0 3000`, champs c, k initialisés par
`--fields-init` depuis SG 27.

| run | champs | mm | normales | corr H | profondeur |
|---|---|---|---|---|---|
| départ (croissance mesurée) | — | 1,419 | 20,4° | 0,58 | 0,63 |
| fit28_combo | a | 1,293 | 17,1° | 0,73 | 0,76 |
| fit28_combo_band | a (bande) | 1,463 | 20,8° | 0,69 | 0,56 |
| fit28_band_ck_r | a, c, k | **0,625** | **6,7°** | **0,79** | **0,87** |

## Lecture

- La croissance tangentielle seule (même à ±50 %) ne creuse pas le sillon :
  les plis de l'atlas ne sont pas des équilibres du modèle avec une croissance
  lisse. Il faut une variation locale de l'épaississement (c) et de la
  croissance du cœur (k). k fait l'essentiel (0,66 mm seul), c seul apporte
  peu, mais les deux ensemble descendent à 0,49 mm.
- Motif des champs, stable entre SG 27 et 28 et avec un lissage dix fois plus
  fort : k < 0 sous le fond du sillon et k > 0 dans une bande sous la paroi
  voisine ; c du même signe ; a plus élevé sur les crêtes. Les champs restent
  sous la borne (|k| = 0,3 sur 4 à 16 % des sommets).
- Resserrer la borne à ±0,15 dégrade (0,66 mm, profondeur 0,52) : l'amplitude
  nécessaire est de l'ordre de 0,2 à 0,3.
- Perte : la composante point-plan et les normales tolèrent un pli décalé, le
  laplacien le pénalise fortement (`tools/loss_probe.py`) ; l'erreur restante
  sur la région est surtout lisse, hors de portée avec des bords fixés.

## Deuxième région : sillon calcarin (SG 27, atlas : profondeur 1,44 mm)

Fond du sillon repéré par la profondeur locale de l'atlas à SG 28, région de
10 mm autour (`data/slab_calc/` : 11 915 nœuds dont 2 059 fixés, 2 263 sommets
de perte), `configs/mu_test/mu2_slab_calc.toml`, mêmes options qu'au sillon
central.

| run | champs | mm | normales | corr H | profondeur |
|---|---|---|---|---|---|
| départ (croissance mesurée) | — | 0,920 | 14,9° | 0,55 | 1,22 |
| fit27_combo_band | a (bande) | 0,760 | 12,3° | 0,65 | 1,29 |
| fit27_band_ck | a, c, k | **0,306** | **3,6°** | **0,92** | **1,41** |

Même motif qu'au sillon central : k < 0 sous les fonds (calcarin et branche
pariéto-occipitale, que la croissance mesurée ne forme pas), k > 0 en anneau sous
les crêtes ; c de même signe. Corrélation avec la profondeur locale de l'atlas
d (`(y − S³⁰y)·n`, lissée) : k r = +0,73, c r = +0,68 (sillon central : +0,62,
+0,65). Les pentes diffèrent (k ≈ 0,17 d contre 0,45 d par mm) mais pas
l'amplitude : pente × écart-type de d ≈ 0,07 dans les deux régions.

## Loi à un paramètre par champ (sillon central, SG 27)

c = 0,35 d, k = 0,45 d (régression sur fit27_band_ck_b3), champs gelés
(`--fields-init` + `--freeze-fields`), a ajusté dans la bande ±50 %,
`--relax0 3000` pour l'équilibre de départ.

| run | champs | mm | normales | corr H | profondeur |
|---|---|---|---|---|---|
| fit27_combo_band | a (bande) | 1,091 | 16,7° | 0,67 | 0,37 |
| fit27_law_d26 | a + loi, d de l'atlas SG 26 | 0,945 | 13,8° | 0,72 | 0,50 |
| fit27_law_s1 | a + loi c, k | 0,898 | 11,9° | 0,85 | 0,51 |
| fit27_law_s15 | a + loi × 1,5 | 0,861 | 10,7° | 0,88 | 0,58 |
| fit27_band_ck | a, c, k libres | 0,485 | 5,3° | 0,87 | 0,62 |

Transfert au calcarin de la loi du sillon central, normalisée par l'écart-type
de d (c = 0,060 d/σ_d, k = 0,077 d/σ_d) : 0,656 mm, 9,9°, corr H 0,78,
profondeur 1,41 (a seul 0,760 ; champs libres 0,306).

La loi récupère un tiers environ de l'écart entre a seul et les champs libres
(×1,5 : 40 %), avec une corrélation de courbure proche de celle des champs
libres : elle place les plis au bon endroit, l'écart restant est surtout
d'amplitude et de forme fine. Calculée sur la profondeur de la semaine
précédente (SG 26, donc sans voir la cible), elle garde l'essentiel du gain
(0,945 mm) : la position future des plis est déjà inscrite dans la géométrie
antérieure.
Doubler la loi (s = 2) imposée d'emblée fait diverger l'équilibre de départ
(NaN) : les champs libres atteignent ces amplitudes progressivement, avec a.

## Loi ajustée par l'adjoint (`--field-law`)

Les champs sont des combinaisons de cartes de base par sommet : c = Σ α_c,j f_j,
k = Σ α_k,j f_j ; theta contient a et les coefficients, le gradient des champs
est ramené aux coefficients (vérifié par différences finies : rapport 1,03).
Bases : d30 (profondeur locale `(y − S³⁰y)·n`, lissée, centrée réduite sur les
sommets de perte), d_fin (`(y − S³y)·n` orthogonalisée à d30, réduite), 1.
Les cartes sont divisées par 10 : sans cela le gradient des coefficients (somme
sur tous les sommets) prend tout le pas `--max-step` et a ne bouge pas.
`--field-band 5` borne alors les coefficients, pas les champs.

Coefficients ajustés (cartes non divisées) :

| région | c : d30 | c : d_fin | c : 1 | k : d30 | k : d_fin | k : 1 |
|---|---|---|---|---|---|---|
| sillon central | 0,233 | 0,053 | −0,044 | 0,087 | −0,043 | 0,084 |
| calcarin | 0,271 | 0,116 | −0,055 | 0,041 | −0,023 | 0,045 |
| loi commune (moyenne) | 0,252 | 0,084 | −0,050 | 0,064 | −0,033 | 0,064 |

Le terme dominant est un épaississement proportionnel à la profondeur : cortex
plus épais sur les crêtes, plus mince aux fonds (c jusqu'à ±0,9), au-delà de
l'épaisseur mesurée sur l'atlas, déjà dans le modèle.

La loi commune imposée d'emblée fait diverger l'équilibre de départ ; elle est
introduite par paliers (`tools/law_ramp.sh` : ×0,25, ×0,5, ×0,75). Le saut à
×1 diverge encore : la loi entière n'a d'équilibre qu'avec un a ajusté en même
temps. Loi commune ×0,75, gelée, a ajusté :

| SG 27 | a seul | loi commune ×0,75 | loi ajustée sur la région | champs libres |
|---|---|---|---|---|
| sillon central, mm | 1,091 | 0,821 | 0,758 | 0,485 |
| profondeur (atlas 0,64) | 0,37 | 0,61 | 0,72 | 0,62 |
| calcarin, mm | 0,760 | 0,550 | 0,522 | 0,306 |
| profondeur (atlas 1,44) | 1,29 | 1,46 | 1,50 | 1,41 |

Une seule loi à six coefficients, commune aux deux régions, donne la profondeur
des sillons de l'atlas et récupère 45 à 50 % de l'écart entre a seul et les
champs libres ; l'écart restant tient à la forme fine, pas à la profondeur.

## Contrôles et enchaînement SG 27 → 28 de la loi

Loi ajustée (6 coefficients + a) depuis des coefficients nuls, sur les cartes de
profondeur ou sur des cartes aléatoires lisses de même longueur de corrélation
(orthogonalisées à d30, même constante) :

| SG 27 | a seul | loi, cartes aléatoires | loi, profondeur (départ nul) | champs libres |
|---|---|---|---|---|
| sillon central, mm | 1,091 | 1,059 | 0,778 | 0,485 |
| calcarin, mm | 0,760 | 0,717 | 0,525 | 0,306 |

Le gain vient de la profondeur de l'atlas, pas des degrés de liberté ; les
coefficients retrouvés depuis zéro sont ceux de l'initialisation par régression
(c ≈ 0,19–0,26 d30, k ≈ 0,04–0,09 d30).

Enchaînement à SG 28 (`tools/chain_week.py` : a reporté, départ = x_fit27 +
(complet28 − complet27)), loi commune ×0,75 gelée sur les cartes de SG 27
(prédictive), a ajusté :

| SG 28 | a seul | loi commune prédictive |
|---|---|---|
| sillon central, mm | 1,463 (bande) / 1,293 | 1,222 |
| calcarin, mm | 0,920 | 0,704 |

Le gain se maintient au calcarin, pas au sillon central, où la profondeur de
SG 27 renseigne mal le sillon de SG 28 (atlas : 0,64 → 0,91 mm).

## Hémisphère entier (phase A, SG 27)

Modèle complet (115 778 nœuds, 610 828 tétraèdres), entrées `tools/make_full.py`
(`data/full/`, cibles ramenées par similitude, cartes de profondeur sur tout
l'atlas), `configs/mu_test/mu2_full.toml`. Une itération L-BFGS (équilibres de
la recherche linéaire + gradient) prend ~35 s sur la RTX 2080 Ti ; une semaine
de 60 itérations, ~40 min.

Les faces de `data/adjoint_prep_s` sont orientées vers l'intérieur : les cartes
de profondeur de l'hémisphère sont calculées avec des normales sortantes
(d < 0 aux fonds, comme sur les régions). Le `sulc` de l'atlas dHCP suit la
convention HCP (positif sur les crêtes).

Loi imposée seule (a mesuré), palier ×0,25 de la loi commune sans constantes :
0,81 mm contre 0,85 ; ×0,5 : 0,90 (trop forte sans ajustement de a). Les
constantes de la loi régionale (c ≈ −0,05) amincissent le cortex partout et
déplacent toute la surface de −0,23 mm le long des normales : sur une région,
les bords fixes et a les compensaient.

Loi ajustée par l'adjoint sur l'hémisphère (6 coefficients + a, cartes / 100 :
avec / 10 le gradient des coefficients, somme sur 32 000 sommets, prend le pas
et a ne bouge pas), 60 itérations, `tools/full_eval.py` :

| SG 27 | région | mm | normales | corr H | profondeur (atlas) |
|---|---|---|---|---|---|
| a seul | cortex | 0,749 | 10,3° | 0,79 | 0,70 (0,88) |
| loi | cortex | 0,744 | 9,3° | 0,84 | 0,74 |
| a seul | sillon central | 1,100 | 16,2° | 0,81 | 0,43 (0,71) |
| loi | sillon central | 0,995 | 14,2° | 0,87 | 0,49 |
| a seul | calcarin | 0,781 | 12,0° | 0,85 | 1,36 (1,49) |
| loi | calcarin | 0,896 | 10,5° | 0,89 | 1,44 |
| a seul | ailleurs | 0,621 | 8,5° | 0,77 | 0,67 (0,84) |
| loi | ailleurs | 0,634 | 7,9° | 0,81 | 0,71 |

Coefficients : c = 0,092 d30 + 0,021 d_fin, k = 0,015 d30 + 0,002 d_fin,
constantes ≈ 0 : mêmes signes que sur les régions, amplitude ~2,7 fois plus
faible (encore en croissance à la fin). La loi améliore la forme des plis
partout (normales, courbure, profondeur) ; l'écart en mm, dominé par la forme
lisse, ne bouge pas.

## Hémisphère entier, chaîne SG 27 → 30 (phase B)

`tools/full_chain.sh` : chaque semaine part de l'état ajusté précédent déplacé
comme la chaîne de référence, la correction de a est reportée ; avec `LAW=1` la
loi aussi, sur les cartes de profondeur de l'atlas de la semaine précédente
(prédictive), coefficients réajustés (« loi ») ou gelés à ceux de SG 27
(« loi figée », `--freeze-fields`). Écart après alignement rigide, cortex :

| | SG 27 | SG 28 | SG 29 | SG 30 |
|---|---|---|---|---|
| référence (croissance mesurée) | 0,84 | 1,18 | 1,52 | 1,87 |
| a seul | 0,75 | 1,03 | 1,29 | 1,58 |
| loi réajustée | 0,74 | 1,00 | 1,20 | 1,40 |
| loi figée | 0,74 | 1,00 | 1,22 | 1,46 |

`tools/full_eval.py`, cortex (profondeur de l'atlas : 1,03 / 1,19 / 1,41 mm) :

| | SG 28 normales / corr H / prof. | SG 29 | SG 30 |
|---|---|---|---|
| a seul | 12,6° / 0,79 / 0,78 | 14,6° / 0,79 / 0,90 | 17,1° / 0,79 / 1,04 |
| loi réajustée | 11,2° / 0,83 / 0,84 | 13,2° / 0,82 / 0,96 | 15,5° / 0,83 / 1,09 |
| loi figée | 10,9° / 0,84 / 0,85 | 12,8° / 0,83 / 0,97 | 15,0° / 0,84 / 1,11 |

Sillon central (profondeur de l'atlas 1,27 mm à SG 29, 1,58 à SG 30) : a seul
1,69 / 1,96 mm, loi 1,52 / 1,72, loi figée 1,51 / 1,73 ; profondeur 0,85 / 1,09
(a seul) contre 0,93 / 1,15 (loi figée). Aucun élément retourné.

Les coefficients réajustés croissent chaque semaine (c sur d30 : 0,108 → 0,135
→ 0,164 → 0,193 ; k : 0,016 → 0,014 → 0,016 → 0,019), avec des constantes qui
grandissent aussi. La loi figée, prédictive, garde l'essentiel du gain sur les
normales, la courbure et la profondeur ; la loi réajustée gagne en plus sur
l'écart en mm. L'écart à l'atlas croît quand même de semaine en semaine dans
toutes les chaînes : la profondeur reste en deçà de l'atlas (1,1 contre 1,4 mm
à SG 30).
