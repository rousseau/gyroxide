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
| fit27_law_s1 | a + loi c, k | 0,898 | 11,9° | 0,85 | 0,51 |
| fit27_band_ck | a, c, k libres | 0,485 | 5,3° | 0,87 | 0,62 |

La loi récupère environ un tiers de l'écart entre a seul et les champs libres.
Doubler la loi (s = 2) imposée d'emblée fait diverger l'équilibre de départ
(NaN) : les champs libres atteignent ces amplitudes progressivement, avec a.
