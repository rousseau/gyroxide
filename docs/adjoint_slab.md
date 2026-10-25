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

## Correction du cœur à l'échelle des lobes (étape 1)

Sur l'hémisphère, environ 54 % du carré de l'écart vient de l'échelle des lobes (S²⁰⁰), à SG 27
comme à SG 30. k reçoit, en plus des cartes de la loi, 40 cartes lisses (`--field-law-k`) : les
premières fonctions propres de Laplace–Beltrami de la surface de référence (SG 21), mode constant
exclu, réduites sur le cortex, / 100 (`tools/lobar_basis.py`, data/full/law/lb40). Les 40 premiers
modes portent 92 à 93 % de l'écart lobaire de SG 27 et de SG 30.

SG 27 : départ de l'ajustement de la loi en plateau (`fit27_lawfit100b`, 0,737 mm après 160
itérations), coefficients lobaires nuls, 60 itérations (`fit27_lb40`). SG 28 → 30 :
`LAW=1 tools/full_chain.sh lb out/full/fit27_lb40 27 30 --field-law-k <cartes>`, même protocole que
la « loi réajustée ».

| écart rigide, cortex (mm) | SG 27 | SG 28 | SG 29 | SG 30 |
|---|---|---|---|---|
| loi réajustée | 0,736 | 0,996 | 1,201 | 1,396 |
| loi + cœur lobaire | **0,671** | **0,741** | **0,883** | **1,116** |
| gain | −9 % | −26 % | −26 % | −20 % |
| lobes (S²⁰⁰), loi → + lobaire | 0,54 → 0,50 | 0,75 → 0,52 | 0,90 → 0,61 | 1,03 → 0,78 |
| profondeur des plis, + lobaire (atlas) | 0,77 (0,88) | 0,91 (1,03) | 1,04 (1,19) | 1,18 (1,41) |
| bande 4–8 mm, loi → + lobaire | 0,80 → 0,83 | 0,73 → 0,83 | 0,67 → 0,77 | 0,62 → 0,71 |
| k lobaire, écart-type | 0,019 | 0,051 | 0,063 | 0,069 |

Le gain porte sur toutes les échelles et toutes les régions (SG 30 : sillon central 1,72 → 1,41,
calcarin 1,54 → 1,16, ailleurs 1,27 → 1,02 mm) ; aucun élément retourné. La liberté lobaire débloque
aussi la loi, en plateau sans elle : ses coefficients repartent (c sur d30 : 10,8 → 11,7 à SG 27,
18,6 à SG 30 ; k sur d30 : 1,6 → 7,1). La constante de k atteint +4,8 % à SG 30 : déficit global de
croissance du cœur par rapport aux tenseurs mesurés. L'écart croît toujours (0,07, 0,14, 0,23 mm
par semaine) et les corrections aussi (k lobaire ±11 % au 95e centile à SG 30) : la loi n'est pas
stationnaire.

## Contrôles de la loi (étape 2)

Loi à 6 coefficients + a ajustée depuis des coefficients nuls, comme `fit27_lawzero`, en ne
changeant que les cartes de profondeur (`tools/law_controls.py`, `tools/law_controls.sh`) :
profondeur de l'atlas d'une semaine antérieure (calculée sur l'hémisphère, restreinte à la région),
carte de SG 27 tournée sur la sphère de l'atlas (*spin test*, deux rotations, |r| < 0,2 avec la
carte d'origine), profondeur de la surface du modèle lui-même (état de départ de SG 27, aucune
information de l'atlas). r : corrélation de la carte d30 avec celle de l'atlas à SG 27 ; part du
gain : (a seul − run) / (a seul − loi SG 27).

| SG 27 | a seul | aléatoire | rotation 1 | rotation 2 | modèle | atlas SG 24 | atlas SG 26 | atlas SG 27 |
|---|---|---|---|---|---|---|---|---|
| sillon central, mm | 1,091 | 1,059 | 1,068 | 1,029 | 1,038 | 1,043 | 0,890 | 0,778 |
| r | | | 0,00 | 0,03 | 0,61 | 0,56 | 0,91 | 1 |
| part du gain | | 10 % | 7 % | 20 % | 17 % | 15 % | 64 % | 100 % |
| calcarin, mm | 0,760 | 0,717 | 0,742 | 0,760 | 0,627 | 0,640 | 0,605 | 0,525 |
| r | | | −0,08 | 0,04 | 0,91 | 0,92 | 0,98 | 1 |
| part du gain | | 18 % | 8 % | 0 % | 57 % | 51 % | 66 % | 100 % |

- La position compte : une carte tournée, qui garde la distribution et la longueur de corrélation de
  la profondeur, ne fait pas mieux que des cartes aléatoires.
- Le gain suit la ressemblance de la carte avec la profondeur de SG 27 : la carte de SG 26
  (r = 0,91 et 0,98) en garde les deux tiers, celle de SG 24 seulement là où le sillon existe déjà
  (calcarin, r = 0,92). La « prédiction » d'une semaine tient donc à ce que l'atlas change peu en
  une semaine : c'est un a priori de position, pas encore une prédiction mécanique.
- La profondeur du modèle lui-même remplace celle de l'atlas là où le modèle a déjà le sillon au bon
  endroit (calcarin : 57 % du gain), pas là où il ne l'a pas (sillon central : 17 %). Une rétroaction
  géométrique peut approfondir un pli existant, pas créer celui qui manque.

Identifiabilité (`tools/law_identifiability.py`, sur `fit27_lawzero`) : jacobienne de l'équilibre
par rapport aux 6 coefficients par différences centrées (h = 0,2, a fixé), sur les sommets de perte.

| | valeurs singulières (mm par unité) | conditionnement | cos(c:d30, k:d30) | cos(c:1, k:1) |
|---|---|---|---|---|
| sillon central | 0,88 0,47 0,30 0,10 0,09 0,04 | 22 | +0,80 | +0,94 |
| calcarin | 1,01 0,59 0,41 0,08 0,04 0,04 | 29 | +0,74 | +0,86 |

Trois combinaisons sont bien déterminées, trois mal : à la même carte, épaississement et croissance
du cœur déplacent la surface presque de la même façon. Les données contraignent la combinaison
« épaississement + cœur » le long de la profondeur, pas chacun des deux séparément ; k déplace la
surface 3 à 6 fois plus que c par unité de coefficient.

Contrainte du cœur : à l'état de départ de SG 27, la dilatation élastique du cœur (ln det(F U⁻¹)) et
son étirement radial, ramenés aux sommets, sont peu corrélés au champ k ajusté librement (sillon
central +0,10 et +0,05, calcarin +0,37 et +0,22) : une croissance du cœur proportionnelle à la
contrainte locale n'est pas soutenue par ces cartes.

Remodelage partiel du cœur (`tools/remodel_chain.py --alpha 0.3`, out/remodel_a03) : de SG 22 à 26,
légèrement moins bon que le cœur élastique (0,544 contre 0,532 mm à SG 26) ; à SG 27, Newton cale
avec 7 éléments retournés et la relaxation amortie de secours diverge. Un relâchement uniforme du
cœur, même partiel, ne creuse pas les plis.

## Métriques standard et choix de la perte (étape 3)

`tools/fold_standard.py` : indice de gyrification A / A(enveloppe convexe), puissance de la courbure
moyenne par bandes d'échelle dyadiques (différence de deux diffusions de la chaleur, analogue des
bandes SpAnGy), similarité des distributions d'indice de forme (intersection d'histogrammes),
rapportés à l'atlas (out/fold_standard.json).

| SG 30, rapport à l'atlas | IG | 1–2 mm | 2–4 mm | 4–8 mm | 8–16 mm | 16–32 mm | indice de forme |
|---|---|---|---|---|---|---|---|
| croissance mesurée | 0,884 | 0,18 | 0,26 | 0,42 | 0,66 | 0,66 | 0,938 |
| a seul | 0,911 | 0,54 | 0,56 | 0,60 | 0,74 | 0,74 | 0,953 |
| loi réajustée | 0,921 | 0,65 | 0,60 | 0,62 | 0,75 | 0,76 | 0,942 |
| loi figée | 0,924 | 0,71 | 0,64 | 0,65 | 0,77 | 0,78 | 0,947 |

Le modèle manque de courbure à toutes les échelles, et de plus en plus avec les semaines (bande
4–8 mm de la loi : 0,80 à SG 27, 0,62 à SG 30). La similarité des indices de forme (0,94 à 0,96
partout) ne discrimine pas les modèles : le critère « > 0,8 » de la littérature est trop lâche.

`tools/loss_synthetic.py` : pli z = h sin(2πx/λ), λ = 5, 10, 15 mm, h = 1, 2, 3 mm ; une perte passe
si L(en phase) < L(décalé de λ/2) < L(plat) et L(en phase) < L(demi-amplitude) < L(plat).

| perte | passe | décalé de λ/2, rapporté au plat |
|---|---|---|
| sommets (L2), point-plan | 0/9 | 4,00 |
| varifold non orienté, σ = λ/4 | 1/9 | 1,9 à 3,5 |
| varifold non orienté, σ = λ/2 | 6/9 | 0,4 à 2,6 |
| varifold non orienté, σ = λ | 9/9 | 0,01 à 0,13 |
| puissance par bandes (log) | 3/9 | 0 (aveugle à la phase) |

Seul le varifold à σ ≈ λ (≈ 10 mm à ces âges) ordonne correctement tous les cas ; il est presque
aveugle à la position (décalé ≈ en phase) : c'est un terme d'amplitude, à combiner avec la
correspondance (positions) ou un σ plus fin une fois les plis amorcés.

Terme varifold dans l'adjoint (`--w-varifold w --varifold-sigma 10`, dans `Geo`) : tenseurs
d'orientation a n nᵀ des faces répartis par poids trilinéaires sur une grille de pas σ/2,
L = Σ_pq exp(−|B_p − B_q|²/σ²) ⟨D_p, D_q⟩, normalisé par sa valeur de départ ; gradient vérifié par
différences finies (`--diag`, rapport 1,0001).

| | sans varifold | avec varifold (w = 1, σ = 10 mm) |
|---|---|---|
| sillon central SG 27 (loi depuis zéro), mm / profondeur | 0,778 / 0,68 | 0,783 / 0,68 |
| calcarin SG 27, mm / profondeur | 0,525 / 1,49 | 0,525 / 1,49 |
| hémisphère SG 30 (chaîne lobaire), mm / profondeur (atlas 1,41) | 1,116 / 1,18 | 1,135 / 1,20 |
| hémisphère SG 30, IG / bande 4–8 mm | 0,941 / 0,71 | 0,945 / 0,73 |

Un terme qui récompense l'amplitude des plis indépendamment de leur phase ne fait presque pas
creuser le modèle : le manque de profondeur après SG 28 n'est pas un défaut de la perte. La perte
composite actuelle est retenue jusqu'à SG 30 ; le varifold reste disponible pour des plis mal placés
(sujets individuels, au-delà de SG 30).

## Rétroaction géométrique sur l'hémisphère (étape 4, premier test)

Chaîne lobaire SG 28 → 30 refaite avec la loi calculée chaque semaine sur la profondeur de la surface
du modèle lui-même (état de départ de la semaine, `MAPS=self tools/full_chain.sh`,
`tools/self_maps.py`) au lieu de l'atlas de la semaine précédente : aucune information de position
de l'atlas dans c et k.

| | SG 28 | SG 29 | SG 30 |
|---|---|---|---|
| r(profondeur du modèle, profondeur de l'atlas) | 0,89 | 0,85 | 0,84 |
| cartes de l'atlas : mm / normales / corr. H / profondeur | 0,741 / 9,3° / 0,86 / 0,91 | 0,883 / 11,0° / 0,85 / 1,04 | 1,116 / 13,4° / 0,85 / 1,18 |
| cartes du modèle | 0,750 / 10,6° / 0,81 / 0,91 | 0,931 / 12,9° / 0,80 / 1,02 | (1,242, non ajusté) |
| bande 1–2 mm, atlas → modèle | 0,64 → 0,58 | 0,75 → 0,61 | |

La profondeur propre du modèle rend l'amplitude (même profondeur des plis, même écart en mm à SG 28)
mais place moins bien les plis (normales, courbure), et l'écart se creuse avec les semaines. À
SG 30, l'ajustement est bloqué : tout pas, même 1/8 du pas initial, augmente la perte (3,3373 à
3,3414 contre 3,3367 au départ) alors que la référence est reproductible (dérive 7·10⁻⁸) :
l'équilibre saute de branche dès que θ bouge (claquage ou contact), aucun élément retourné. Les
rejets de la recherche linéaire sont maintenant imprimés avec leur cause (éléments retournés,
valeurs non finies, perte).

Bilan de l'étape 4 à ce stade : ni une croissance du cœur proportionnelle à la contrainte locale,
ni un relâchement uniforme du cœur (α = 0,3), ni une rétroaction sur la profondeur propre du modèle
ne remplacent la loi ajustée sur l'atlas. La rétroaction géométrique approfondit les plis existants
(calcarin, hémisphère à SG 28) ; la position des plis qui manquent (sillon central) vient de l'atlas.

## Remesure des tenseurs du cœur (étape 4a)

La correction lobaire ajustée (étape 1) et sa constante (+4,8 % à SG 30) compensent-elles un défaut
de mesure des tenseurs U ? `tools/atlas_core_tensors.py --smooth k --out` à k = 200 (reproduit la
table en service à 2·10⁻⁵ près, médiane), 100, 50, 10 ; variantes « v » mises à l'échelle chaque
semaine pour que det U atteigne le volume blanc de l'atlas (det U = 2,20 contre 2,40 à SG 30 pour
S²⁰⁰, 2,28 pour S¹⁰). Chaînes d'équilibres à croissance mesurée, sans ajustement
(`tools/remodel_chain.py --alpha 0 --core`), écart après alignement rigide sans changement d'échelle
sur l'atlas (lobes S²⁰⁰ / plis primaires S⁵⁰ / sommets, mm) :

| | SG 24 | SG 25 | SG 26 | SG 27 |
|---|---|---|---|---|
| référence (S²⁰⁰) | 0,131 / 0,169 / 0,213 | 0,208 / 0,276 / 0,350 | 0,325 / 0,427 / 0,532 | 0,521 / 0,682 / 0,841 |
| S²⁰⁰ + volume | 0,123 / 0,163 / 0,208 | 0,213 / 0,280 / 0,354 | 0,358 / 0,453 / 0,554 | 0,596 / 0,743 / 0,893 |
| S¹⁰⁰ + volume | 0,105 / 0,139 / 0,182 | 0,184 / 0,241 / 0,311 | 0,306 / 0,392 / 0,488 | 0,523 / 0,655 / 0,797 |
| S⁵⁰ + volume | sommets 0,164 | échec (2 éléments retournés, puis NaN) | | |

S¹⁰⁰ + volume, suite : SG 28 0,741 / 0,922 / 1,112, SG 29 0,967 / 1,189 / 1,428, SG 30 1,237 /
1,496 / 1,789 (référence 0,742 / 0,962 / 1,176, 0,978 / 1,248 / 1,517, 1,202 / 1,526 / 1,865) :
lobes inchangés, −4 à −6 % sur les plis primaires et au sommet.

Corriger le volume du cœur dégrade l'écart lobaire : le volume de l'hémisphère n'est pas
déficitaire, seul det U du cœur l'est, compensé par le cortex. Une mesure plus fine (S¹⁰⁰) améliore
les lobes tôt (−20 % à SG 24) mais plus à SG 27 (0,523 contre 0,521), et S⁵⁰ retourne des éléments.
La correction lobaire ajustée (−26 % à SG 28) n'est donc pas un artefact de la mesure de U : elle
compense un écart mécanique, le corps ne suivant pas la croissance prescrite au cœur.

## Prépatron et rétroaction (étape 4b)

Chaînes lobaires SG 28 → 30 (`tools/full_chain.sh`, même départ `fit27_lb40`) qui ne diffèrent que
par les cartes de la loi : atlas de la semaine précédente (`lb`, référence), atlas de SG 26 figé
(`MAPS=prep PREP=26`, prépatron de 2 à 4 semaines), profondeur du modèle (`MAPS=self`), les deux
(`MAPS=both`, cinq cartes par champ).

| écart rigide, cortex (mm) | SG 28 | SG 29 | SG 30 |
|---|---|---|---|
| atlas, semaine précédente | 0,741 | 0,883 | 1,116 |
| atlas SG 26 figé | 0,764 | 0,943 | 1,218 |
| profondeur du modèle | 0,750 | 0,931 | (1,242, bloqué) |
| atlas SG 26 + modèle | 0,773 | 0,947 | 1,252 |

SG 30, normales / corrélation de courbure / profondeur (atlas 1,41) : semaine précédente 13,4° /
0,85 / 1,18 ; SG 26 figé 16,2° / 0,80 / 1,14 ; SG 26 + modèle 16,9° / 0,79 / 1,12.

Le prépatron se dégrade avec l'horizon : les sillons qui émergent entre SG 26 et 30 n'y sont pas.
Avec les deux jeux de cartes, l'ajustement donne des coefficients négatifs à la profondeur du modèle
(c : −0,3 et −1,1 ; k : −0,9 et −2,6 à SG 28) : il s'en sert pour atténuer les plis déjà profonds,
pas pour les creuser. Ni un prépatron précoce, ni une rétroaction géométrique, ni leur combinaison ne
remplacent l'information de position récente de l'atlas ; la correction lobaire du cœur, elle, garde
son gain dans toutes les variantes (1,22 mm à SG 30 avec le prépatron de SG 26, contre 1,40 pour la
loi sans correction lobaire).

## Vers SG 36 (étape 5)

Chaîne de référence (croissance mesurée, sans ajustement, `tools/remodel_chain.py --alpha 0
--resume 31 --w1 36`) : 2,149 / 2,356 / 2,542 / 2,661 / 2,741 / 2,806 mm de SG 31 à 36, aucun
élément retourné (Newton cale presque chaque semaine, la relaxation amortie de secours suffit).
Entrées de l'hémisphère pour SG 31–36 : `tools/make_full.py --weeks 31 ... 36`.

Qualité des tétraèdres (rapport moyen 12 (3V)^{2/3} / Σ l², 1 = régulier) : 1er centile 0,43 à SG 21,
0,41 sur la chaîne de référence à SG 30, 0,37 sur l'ajustement lobaire à SG 30 ; une centaine de
tétraèdres sous 0,1 sur 610 828, aucun sous 0,02. Le maillage n'impose pas de remaillage à SG 30.

La chaîne lobaire échoue à SG 31 : l'équilibre de départ (état ajusté de SG 30 déplacé comme la
référence) diverge en NaN par relaxation amortie, même à pas de temps divisé par deux, et Newton seul
ne réduit pas la force (0,43, ou 0,09 pour un quart de semaine par continuation en temps,
`tools/time_ramp.sh`). Cause : la correction d'épaississement de la loi (jusqu'à −1,12) s'ajoute à
l'amincissement mesuré du cortex (−0,40 à SG 30, −0,48 à SG 31) et rend la croissance normale
1 + c négative sur 5 nœuds à SG 30, 15 à SG 31 (moins de 0,2 sur plus de 100) : tenseur de
croissance inversé. Les sauts de branche de la chaîne sur la profondeur propre à SG 30 en relèvent
probablement aussi.

Remède : correction multiplicative (`--thick-mult`), 1 + c = (1 + c_mesuré) exp(δ) > 0 par
construction, ≈ additive pour δ petit (facteur 1 + c_mesuré) ; dc/dδ = 1 + c dans le gradient
(vérifié par différences finies : rapport 1,000 à h = 0,5, 1, 2). `adjoint_week` accepte aussi un
temps de point de reprise entre deux nœuds (nœud suivant, a interpolé), pour la continuation en
temps.

Chaîne lobaire avec correction multiplicative (`LAW=1 tools/full_chain.sh lbm out/full/fit27_lb40 27
36 --field-law-k <cartes> --thick-mult`) : SG 28–30 identiques à la chaîne additive (0,751 / 0,894 /
1,121 contre 0,741 / 0,883 / 1,116 mm), puis la chaîne passe SG 31 et va jusqu'à SG 36 sans NaN ni
élément retourné. Mais les ajustements de SG 32, 33 et 34 se sont arrêtés après 10, 10 et 15
itérations, et SG 35 n'a fait aucun pas (état de départ non ajusté) ; SG 28–31 et 36 ont fait leurs
60 itérations. Les valeurs de SG 32–35 sont donc celles d'ajustements incomplets.

| | SG 30 | SG 31 | SG 32 | SG 33 | SG 34 | SG 35 | SG 36 |
|---|---|---|---|---|---|---|---|
| référence (croissance mesurée), mm | 1,865 | 2,146 | 2,349 | 2,532 | 2,648 | 2,724 | 2,785 |
| loi + cœur lobaire, mm | **1,121** | **1,377** | **1,616** | **1,837** | **1,994** | **2,091** | **2,159** |
| normales, référence → lobaire | 25,7° → 13,7° | 29,8° → 15,7° | 32,7° → 18,2° | 35,4° → 21,1° | 37,6° → 22,9° | 39,2° → 23,8° | 40,2° → 22,7° |
| corr. H, référence → lobaire | 0,46 → 0,86 | 0,41 → 0,86 | 0,37 → 0,84 | 0,33 → 0,81 | 0,30 → 0,79 | 0,28 → 0,79 | 0,28 → 0,82 |

Profondeur des plis (cortex ; atlas 1,41 / 2,02 / 2,36 mm à SG 30 / 33 / 36) : référence 0,83 / 0,99 /
1,08, lobaire 1,17 / 1,45 / 1,64. Rapport à l'atlas (`tools/fold_standard.py`) à SG 30 / 33 / 36 :
IG 0,884 / 0,810 / 0,774 (référence) contre 0,938 / 0,879 / 0,848 ; bande 4–8 mm 0,42 / 0,34 / 0,32
contre 0,71 / 0,60 / 0,56. Corrections : k lobaire, écart-type 0,070 / 0,073 / 0,074 ; coefficient
de c sur d30 18,9 / 21,9 / 24,8 ; δc entre −0,46 et +0,83 (1er et 99e centiles) à SG 36.

L'ajustement garde la position des plis jusqu'à SG 36 (corrélation de courbure 0,79 à 0,86, contre
0,28 pour la référence) et réduit l'écart d'un quart à un tiers, mais la profondeur et la puissance
de courbure restent en deçà de l'atlas (70 % de la profondeur, 56 % de la bande 4–8 mm à SG 36) :
les plis secondaires de SG 31–36 manquent au modèle.

Arrêts prématurés de SG 32–35 : à SG 35, tout pas de la recherche linéaire est refusé, la perte
augmentant d'autant plus que le pas est petit (3,2879 au pas 1, 3,2942 au pas 1/4, contre 3,2809) :
depuis l'équilibre de départ, Newton seul bascule dans un autre bassin au moindre changement de θ.
Avec 300 pas de relaxation amortie avant chaque Newton (`--relax 300`), les pas sont acceptés mais la
référence recalculée au même θ dérive de +2·10⁻³ à +8·10⁻³ d'une itération à l'autre et l'écart
reste à 2,09 mm : aux semaines tardives, l'état qui colle à l'atlas n'est pas un équilibre stable du
modèle, et l'ajustement bute sur un plancher de bruit de la perte de l'ordre de 10⁻².

## Plis secondaires (SG 33)

Régions où l'atlas creuse un sillon entre SG 30 et 33 que la chaîne lobaire ne forme pas : score
= déficit de profondeur du modèle × approfondissement de l'atlas entre SG 30 et 33, lissé ; pics hors
des régions déjà étudiées : frontale supérieure (sommet 5707) et inférieure centrale (26574, région
sylvienne). Sous-maillages de 10 mm de rayon (`tools/make_slab.py --full out/full/lbm_x/w{w}.npy`,
data/slab_front, data/slab_tinf), table de a reportée et champs c, k initialisés par sommet à l'état
ajusté de la chaîne lobaire à SG 33 ; puis a, c, k libres par sommet (`--field-band 1.5
--thick-mult`).

| SG 33 | départ (chaîne lobaire) | champs libres | atlas |
|---|---|---|---|
| frontale : mm / normales / corr. H / profondeur | 1,861 / 27,0° / 0,63 / 1,61 | 0,821 / 8,2° / 0,88 / 2,25 | 2,56 |
| inférieure : mm / normales / corr. H / profondeur | 2,207 / 42,7° / 0,33 / 1,45 | 1,760 / 35,1° / 0,51 / 1,43 | 2,34 |
| inférieure + varifold σ = 5 mm | | 1,842 / 34,6° / 0,38 / 1,54 | |

Frontale : plis au bon endroit mais trop peu profonds ; des champs libres les creusent (88 % de la
profondeur de l'atlas). Les corrections sont surtout fines (80 % de la variance sous S²⁰) et ne
suivent pas la profondeur de l'atlas (|r| ≤ 0,14 à toutes les échelles) : pas de loi de profondeur
pour ces plis. Inférieure : plis mal placés dès le départ ; l'ajustement bute sur un palier, avec ou
sans varifold.

Hémisphère entier, c et k libres par sommet depuis la chaîne lobaire (`out/full/fit33_free`, 52
itérations) : 1,837 → 1,553 mm (lobes 1,27 → 1,19) ; normales 21,1° → 11,9° ; corrélation de
courbure 0,81 → 0,93 ; profondeur 1,45 → 1,63 mm (atlas 2,02) ; puissance de courbure 1–2 / 2–4 /
4–8 / 8–16 mm : 0,70 / 0,65 / 0,60 / 0,71 → 0,97 / 0,83 / 0,70 / 0,75 ; corrections d'écart-type
0,07 à 0,09. Région sylvienne : corrélation 0,39 → 0,65, normales 39° → 29°.

Lecture : avec une croissance hétérogène quelconque, le modèle retrouve la forme fine des plis
secondaires là où ils sont amorcés ; leur amplitude s'arrête à 81 % de la profondeur (88 % sur la
région frontale) et 70 % de la bande 4–8 mm, l'ajustement s'arrêtant sur un palier (limite du
modèle ou de l'optimisation, non tranché) ; la région sylvienne, mal placée depuis les semaines antérieures, ne se rattrape pas
localement ; l'écart en mm reste dominé par la forme lobaire (1,19 sur 1,55 mm).

Profondeur des plis secondaires, région frontale (SG 33, a, c, k libres) :

| variante | itérations | mm | corr. H | profondeur (atlas 2,56) | arrêt |
|---|---|---|---|---|---|
| μ_g/μ_w = 2, β = 10⁻⁴ | 36 | 0,821 | 0,88 | 2,25 | éléments retournés |
| μ_g/μ_w = 3 | 35 | 0,795 | 0,90 | 2,27 | éléments retournés |
| μ_g/μ_w = 5 | 26 | 0,875 | 0,91 | 2,26 | perte plate |
| 20 éléments retournés tolérés (`--max-inverted 20`) | 37 | 0,821 | 0,88 | 2,25 | gradient faux (perte ∝ pas) |
| β = 10⁻³ | 55 | 0,816 | 0,89 | 2,25 | perte plate |
| β = 10⁻² | 45 | 0,851 | 0,88 | 2,19 | perte plate |

Ni la raideur relative, ni le lissage des champs, ni la tolérance aux éléments retournés ne changent
la profondeur ; avec un lissage plus fort l'ajustement converge (perte plate) au même état : ce
n'est pas un arrêt de l'optimiseur. Les bornes ne sont pas atteintes (1 % des sommets à la borne de
a, aucun à celle de c, k). À cet optimum, des tétraèdres du cœur, à 2–3,5 mm sous la surface, sous
les fonds qui se creusent, sont écrasés (qualité 0,006 à 0,02, contre 0,6–0,9 au départ) : avec les
trois modes de croissance disponibles (tangentielle, épaississement normal, cœur isotrope), la
meilleure forme atteint 88 % de la profondeur de l'atlas dans cette région. Piste suivante : élargir
ces modes (croissance du cœur anisotrope, radiale et tangentielle séparées).

Croissance du cœur anisotrope (`--fit-fields c,k,r`) : r ajoute r n nᵀ au tenseur du cœur (n :
normale de croissance du tétraèdre), soit une croissance tangentielle k et radiale k + r
(`Simulation::set_core_delta_radial`, gradient par autodifférentiation ; neutre à r = 0 : même
équilibre de départ ; contrôle par différences finies, rapport 1,000 à h = 0,5 et 1).

| SG 33, champs libres | mm | normales | corr. H | profondeur | arrêt |
|---|---|---|---|---|---|
| frontale, isotrope (β = 10⁻³) | 0,816 | 8,3° | 0,89 | 2,25 (atlas 2,56) | perte plate, it. 55 |
| frontale, anisotrope | 0,925 | 9,4° | 0,89 | 2,18 | élément retourné, it. 30 |
| frontale, anisotrope, β = 10⁻³ | 0,934 | 9,5° | 0,89 | 2,17 | élément retourné, it. 28 |
| sylvienne, isotrope | 1,760 | 35,1° | 0,51 | 1,43 (atlas 2,34) | perte plate |
| sylvienne, anisotrope | 1,687 | 32,4° | 0,59 | 1,56 | perte plate |

La composante radiale aide un peu dans la région sylvienne ; dans la région frontale, elle accélère
la déformation sous les fonds et l'ajustement bute plus tôt sur un tétraèdre du cœur retourné.
Le verrou commun est la dégénérescence d'éléments du cœur, 2 à 3,5 mm sous les fonds qui se
creusent, dans un maillage construit à SG 21 : prochaine étape, raffiner localement le cœur sous ces
fonds (scission d'arêtes intérieures seulement, surface et correspondance avec l'atlas inchangées).

Raffinement local et profondeur du découpage. Raffiner le cœur sous les fonds (`tools/refine_slab.py`,
bissection de la plus longue arête intérieure des tétraèdres dégradés et de leurs voisins, une
passe : 2 915 arêtes, 11 791 → 26 628 tétraèdres) n'aide pas : 2,21 mm de profondeur, arrêt à
l'itération 23 sur un gradient faux (la bissection crée des éléments effilés). Le vrai verrou était
le découpage : la région de 5 mm de profondeur a son fond fixé, et le cœur se trouve écrasé entre ce
fond et le sillon qui se creuse (tétraèdres écrasés à 2–3,5 mm sous la surface). Même région découpée
sur 10 mm (`make_slab.py --depth 10`, data/slab_front10, β = 10⁻³) :

| frontale, SG 33 | 5 mm de profondeur | 10 mm de profondeur | atlas |
|---|---|---|---|
| mm / rigide | 0,816 / 0,59 | **0,531 / 0,477** | |
| normales / corr. H | 8,3° / 0,89 | **6,1° / 0,89** | |
| profondeur des plis | 2,25 (88 %) | **2,45 (96 %)** | 2,56 |
| arrêt | perte plate, it. 55 | aucun (90 itérations, encore en baisse) | |
| tétraèdres de qualité < 0,05 | 14 | 4 | |

Avec une croissance libre, le modèle atteint 96 % de la profondeur de ce pli secondaire : le
« plafond » des essais précédents (raideur, lissage, cœur anisotrope, raffinement) venait du fond
fixé à 5 mm, pas du modèle. Les régions de 5 mm restent valables pour les sillons peu profonds
(SG 26–28) ; au-delà, découper sur 10 mm au moins.

## Une loi pour les plis secondaires (région frontale, 10 mm, SG 33)

Corrélations des corrections libres (écarts à l'état de la chaîne lobaire) avec des cartes candidates :
Δc ne suit rien (|r| ≤ 0,12) ; Δk est anticorrélé à la profondeur de l'atlas (r = −0,24 à −0,32
selon l'échelle) et à celle du modèle (−0,37) : le cœur pousse davantage sous les fonds secondaires,
à l'inverse des sillons primaires. Loi à 7 cartes par champ (c et k de la chaîne lobaire comme cartes,
profondeur de l'atlas à S³⁰, S¹⁰, S³, approfondissement entre deux semaines, constante), a libre,
coefficients partis de l'état de la chaîne lobaire (`out/slab_front10/law`) :

| SG 33, région frontale 10 mm | mm | normales | corr. H | profondeur (atlas 2,56) |
|---|---|---|---|---|
| a seul (c, k figés) | 1,563 | 20,2° | 0,80 | 1,95 |
| loi, cartes de SG 32 (prédictive) | 0,931 | 12,2° | 0,77 | 2,18 |
| loi, cartes de SG 33 | 0,778 | 11,7° | 0,80 | 2,28 |
| champs libres | 0,531 | 6,1° | 0,89 | 2,45 |

La loi récupère 76 % du gain des champs libres en mm (61 % en prédictif, cartes de SG 32 corrélées à
0,95 avec celles de SG 33). Coefficients de même signe dans les deux versions : épaississement
augmenté avec l'approfondissement de l'atlas (cortex aminci sous les sillons qui se creusent),
croissance du cœur plus forte sous les fonds (S³⁰ et S¹⁰) ; constantes c ≈ +0,2 et k ≈ −0,3,
probablement une correction régionale (bords fixés).

Hémisphère entier, SG 33. Ajuster la loi enrichie des cartes prédictives de SG 32 (profondeur S¹⁰,
S³, approfondissement de SG 29 à 32), coefficients partis de zéro depuis la chaîne lobaire : aucun
gain (1,837 → 1,841 mm, arrêt à l'itération 9), ni avec `--force-tol 5e-5` (1,829) ni avec
`--max-step 0.2` (1,835) : chaque pas sur un coefficient modifie tout l'hémisphère et l'équilibre
change de branche (pertes d'essai non monotones en fonction du pas), alors que l'ajustement libre,
aux pas locaux, progresse (1,553 mm). Régression des corrections libres de l'hémisphère sur ces
cartes : R² = 0,22 pour Δc (approfondissement r = +0,45), 0,13 pour Δk (signe de la relation à la
profondeur opposé à celui de la région frontale) ; la loi régressée, imposée, donne 1,822 mm
(équilibre relaxé). Les plis secondaires sont atteignables avec une croissance libre, mais aucune loi
simple sur la profondeur de l'atlas ne décrit cette croissance à l'échelle de l'hémisphère ; sur la
région frontale seule, une loi en capture 61 à 76 %.

## Ajustabilité aux semaines tardives

Diagnostic à SG 33 (hémisphère, `adjoint_week --probe s1,s2,...` : perte le long de la première
direction de descente, résidu de Newton, paires de contact) :

1. Les pas de Newton résolvaient K δ = F par un gradient conjugué plafonné à 100 itérations : à
   SG 33, des équilibres d'essai légèrement faux (la perte s'écarte de la prédiction linéaire dès le
   pas 1/4). Avec `--newton-cg 600`, elle la suit exactement jusqu'au pas 1/4.
2. Le contact n'y est pour rien : sans contact, mêmes pertes aux pas 1/2 et 1 (3,526082, 3,529201).
3. Défaut de code : la boucle de Newton testait la convergence après le pas ; tout équilibre faisait
   au moins un pas, même depuis un état convergé, et ce pas sur un résidu d'arrondi déplaçait les
   modes mous (0,004 mm, +1·10⁻³ sur la perte à chaque référence). Corrigé : convergence testée
   avant chaque pas (dérive des références 10⁻⁸ au lieu de 10⁻³).
4. Reste un plancher : au-delà d'une dizaine d'itérations, l'état ajusté devient métastable et tout
   essai résolu par Newton seul bascule vers un équilibre voisin plus haut d'environ 0,01. Une
   relaxation amortie avant Newton à chaque essai (`--relax 300`), une tolérance de force au-dessus
   du bruit d'arrondi (`--force-tol 5e-5`) et une recherche linéaire tolérante (`--ls-tol 3e-3`,
   nouvelle option) laissent l'ajustement aller au bout.

SG 35 de la chaîne lobaire (0 itération auparavant) : 60 itérations, 2,091 → 2,061 mm (2,053 au mieux).
Réglages pour les semaines tardives : `--newton-cg 600 --relax 300 --force-tol 5e-5 --ls-tol 3e-3`.
