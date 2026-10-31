# Plan : points de comparaison avec la littérature

Objectif : situer le modèle de Tallinen enrichi (étapes 0 à 5, hypothèses H1 à H12 de la
présentation) par rapport à d'autres modèles du plissement et à des données indépendantes de
l'atlas dHCP. Toutes les comparaisons suivent un **protocole commun** : on ne compare que ce qui
est mesuré de la même façon.

## Protocole commun

- Données : atlas dHCP, hémisphère gauche, SG 21 → 36 ; maillage `w21_left_610k_s`.
- Métriques : écart rigide au cortex (mm, `tools/deviation_mm.py`), aux échelles S²⁰⁰ / S⁵⁰ /
  sommets ; normales, corrélation de courbure, profondeur des plis (`tools/full_eval.py`) ; IG et
  puissance de courbure par bandes (`tools/fold_standard.py`) ; plus la loi d'échelle de Mota
  (volet A ci-dessous), ajoutée à `fold_standard.py`.
- Une variante de mécanisme se mesure par la **part du gain** qu'elle récupère entre deux
  références fixes : « a seul » (étape 3) et « loi de l'atlas » (étape 4), comme dans les
  contrôles de la loi (H6). 0 % : rien de plus qu'un champ quelconque ; 100 % : aussi bien que
  l'atlas.
- Chaque volet produit une ligne du tableau de comparaison (volet F) et, au besoin, une
  hypothèse nouvelle ou révisée dans la présentation.

## A. Loi d'échelle universelle (Mota & Herculano-Houzel 2015 ; Wang et al. 2016)

**Hypothèse.** Les surfaces simulées suivent la loi T^½ · A_t = k · A_e^1,25 (T épaisseur
corticale, A_t aire totale, A_e aire exposée) comme les cerveaux réels, et chaque étape s'en
rapproche.

**Test.**
1. Atlas SG 21 → 36 : T = moyenne de l'épaisseur pial-blanc (`nodal_h`), A_t = aire de la surface
   piale, A_e = aire de l'enveloppe convexe. Vérifier d'abord que l'atlas fœtal suit la loi
   (pente de log(T^½ A_t) contre log A_e) : la loi est établie sur des cerveaux adultes et
   d'espèces différentes, son domaine fœtal n'est pas acquis.
2. Mêmes grandeurs pour chaque étape : Tallinen tel quel (`out/fwd_tallinen_s`), carte mesurée,
   référence à croissance mesurée (`out/remodel_a0`, `out/real/ref`), chaîne ajustée (`out/full`,
   `out/material/fit*_1_50`). T du modèle : épaisseur prescrite H(t) × (1 + c) aux sommets.
3. Trajectoires dans le plan (log A_e, log T^½ A_t) : pente et décalage k de chaque modèle,
   rapportés à ceux de l'atlas.

**Critère.** Écart de pente et de décalage à l'atlas, par étape. Un modèle à la bonne pente et au
mauvais décalage plisse au bon rythme avec une épaisseur fausse.

**Coût.** Une demi-journée, aucune simulation nouvelle. **Risque** : la loi est globale (cerveau
entier) ; sur un hémisphère avec mur médian plan, A_e doit être calculée de la même façon pour
l'atlas et le modèle.

## B. Prépatron par réaction-diffusion (Lefèvre & Mangin 2010 ; Striegel & Hurdal 2009)

**Hypothèse.** Un prépatron de Turing calculé sur la surface de SG 21, sans aucune information de
l'atlas plissé, donne une carte de croissance radiale qui récupère une part du gain de la loi de
profondeur (H6). Réfutée si la part reste au niveau des cartes aléatoires (10 à 18 %).

**Test.**
1. Système de Turing (Schnakenberg, comme Lefèvre & Mangin) sur la surface de SG 21, domaine en
   croissance selon a(t) mesuré, laplacien cotangent (`tools/lobar_basis.py`) ; nombre d'onde
   réglé pour donner la longueur d'onde des sillons primaires de l'atlas (~10 mm), quelques
   tirages des conditions initiales.
2. Cartes de base du motif à SG 26–27, centrées réduites comme d30, d_fin
   (`tools/law_controls.py`), puis ajustement de la loi à 6 coefficients sur les deux régions
   (sillon central, calcarin), départ à coefficients nuls, comme les contrôles de H6.
3. Si la part dépasse ~30 % : chaîne de l'hémisphère SG 28 → 30 avec ces cartes à la place de
   l'atlas (`MAPS=` dans `tools/full_chain.sh`, nouveau mode).

**Critère.** Part du gain, corrélation de la carte de Turing avec la profondeur de l'atlas à
SG 27, et sa variabilité entre tirages (un prépatron dont la position dépend du tirage ne peut pas
placer les sillons d'un atlas moyen).

**Coût.** 1 à 2 jours (solveur de réaction-diffusion en Python, ajustements ~1 h par région).
**Risque** : sans contrainte de position (conditions aux limites, gradients d'expression), Turing
donne la longueur d'onde mais pas la position ; un résultat négatif serait attendu et devrait
être présenté comme tel, pas comme une réfutation de tout prépatron.

## C. Croissance du cœur guidée par la tension des fibres (Holland et al. 2015 ; Garcia, Wang, Kroenke 2021)

**Hypothèse.** Un cœur dont la croissance le long des fibres suit leur tension produit la
correction lobaire (H8) et une part de la croissance radiale hétérogène (H5), que l'on a dû
ajuster sur l'atlas. C'est la rétroaction mécanique que H7 n'a pas testée (H7 : croissance
isotrope proportionnelle à la contrainte, remodelage isotrope).

**Test.**
1. Direction des fibres par tétraèdre : direction radiale (normale de croissance n, déjà utilisée
   par `set_core_delta_radial`), ou gradient de la distance à la surface piale.
2. Règle de croissance, dans le cadre de `tools/remodel_chain.py` (tenseurs du cœur par
   tétraèdre via `--core-override`) : après l'équilibre de la semaine,
   U_eff ← U_eff + α (λ_f − 1) e eᵀ, avec λ_f l'étirement élastique le long de la fibre e
   (allongement si tendue, rien si comprimée) ; α = 0 redonne la référence.
3. Chaîne SG 22 → 30 à croissance mesurée pour quelques α (0,1 ; 0,3 ; 1), ancien matériau puis
   matériau réaliste. Comparer à la référence (étape 2) et à la correction lobaire ajustée
   (étape 5) : écart rigide à l'échelle S²⁰⁰, et corrélation de la croissance supplémentaire du
   cœur avec le champ k lobaire ajusté.

**Critère.** Réduction de l'écart lobaire par rapport à la référence (la correction ajustée fait
−26 % à SG 28) et corrélation avec k ajusté. Un α qui retourne des éléments ou fait diverger les
équilibres est un résultat de méthode, à continuer en α comme on l'a fait en K.

**Coût.** 1 à 2 jours de code (Python, sans toucher au solveur), ~4 h de calcul par chaîne.
**Risque** : la direction radiale est une approximation des fibres réelles ; un test plus fidèle
demanderait des orientations mesurées (IRM de diffusion fœtale), hors de portée ici.

## D. Cartes d'expansion longitudinales (Garcia et al. 2018, PNAS)

**Hypothèse.** L'expansion corticale mesurée sur des individus suivis longitudinalement
(prématurés, 28 à 38 semaines, correspondance anatomique contrainte) ressemble davantage à notre
croissance tangentielle ajustée (a + corrections) qu'à celle mesurée sur l'atlas. Elle teste
aussi une hypothèse de travail : la correspondance des sommets de l'atlas est-elle matérielle ?

**Test.**
1. Données publiques sur BALSA (étude K65Z, compte requis) : surfaces et cartes d'expansion
   moyennes.
2. Ramener leurs cartes sur la sphère de l'atlas dHCP (recalage sphérique MSM entre leur gabarit
   et celui de l'atlas ; transformations dHCP ↔ fs_LR publiées).
3. Corrélations par intervalle d'âge (28–30, 30–32, … semaines) entre leur expansion et :
   (a) a(t) mesuré sur l'atlas, (b) a ajusté, (c) l'expansion totale de la surface du modèle
   ajusté. Aux échelles lobaire et des plis primaires (S²⁰⁰, S⁵⁰).

**Critère.** Si (b) ou (c) est plus proche que (a) de l'expansion longitudinale, l'ajustement
corrige un biais de l'atlas moyen ; sinon, il compense autre chose (mécanique absente).

**Coût.** 1 jour pour l'accès et la lecture des données, 2 à 3 jours pour le recalage.
**Risque** : prématurés et fœtus diffèrent (environnement extra-utérin) ; le recouvrement
d'âge (28–36) est bon mais le recalage entre gabarits ajoute son propre écart, à mesurer sur un
test aller-retour.

## E. Comparaison de code à code (Wang et al. 2021 ; Alenyà et al. 2022 ; arXiv 2607.08534)

**Hypothèse.** À géométrie initiale et entrées égales, les autres modèles biomécaniques issus
d'IRM fœtale se placent sur notre échelle des ajouts au niveau de l'étape 1 ou 2.

**Test.** Récupérer le code (équipe ou dépôt public), le lancer sur la surface de SG 21 de
l'atlas avec ses propres paramètres, puis avec nos entrées mesurées (a(t), H(t), cœur) ;
mesurer avec le protocole commun. Lire d'abord la prépublication 2607.08534 (modèle FEniCS ouvert).

**Coût.** Variable : quelques jours si le code est ouvert et documenté. **Risque** : maillages et
conventions différents ; ne comparer que des surfaces ramenées sur l'atlas.

## F. Restitution

- Un tableau « modèles de la littérature » : ligne par modèle ou mécanisme, colonnes = écart en
  mm à SG 26 / 28 / 30, part du gain, loi de Mota (pente, décalage), corrélation avec Garcia 2018.
- Dans la présentation : une diapositive de positionnement (fin de la partie Cadre) et, dans les
  hypothèses, H7 complétée par le volet C et une hypothèse nouvelle par volet concluant (B, D).
- `docs/adjoint_slab.md` : une section par volet, chiffres et commandes.
- Références à revérifier avant citation (seules Garcia 2018, Budday 2017 et Wang 2021 l'ont été).

## Ordre proposé

| volet | coût | dépend de | ce qu'il tranche |
|---|---|---|---|
| A. loi de Mota | ½ jour | — | réalisme global de chaque étape, critère indépendant de l'atlas |
| B. prépatron de Turing | 1–2 jours | — | origine de la position des sillons (H6) |
| C. tension des fibres | 2–3 jours | — | origine de la correction lobaire (H7, H8) |
| D. Garcia 2018 | 3–4 jours | accès BALSA | biais de l'atlas moyen, correspondance |
| E. code à code | variable | code disponible | position des autres modèles sur l'échelle |

A d'abord (rapide, sans calcul) ; B et C ensuite, car ils testent directement les deux questions
ouvertes de la synthèse (position des sillons, origine de la carte) ; D en parallèle dès que
l'accès aux données est obtenu ; E selon la disponibilité des codes.

## Résultats

### A. Loi de Mota (29 octobre)

`tools/mota_law.py out/mota_law.json --plot out/mota_law.png`. Épaisseur du modèle : volume de la
couche corticale (poids du modèle, g = 1 / (1 + exp(10 (d/H − 1)))) / aire, calibrée sur SG 21
(1,32 mm contre 1,43 mm pour l'atlas, × 1,08). Atlas : A_t 5 335 → 16 882 mm², A_e 5 666 → 13 709 mm²,
T 1,43 → 1,10 mm de SG 21 à 36.

| | pente (SG) | écart à la droite de l'atlas, log10 (SG 26 / 30 / 36) |
|---|---|---|
| atlas | 1,15 (21–36) | 0 |
| 0 · Tallinen 2016 | 1,78 (22–26) | +0,097 / — / — |
| 2 · croissance mesurée | 1,00 (22–36) | +0,004 / −0,022 / −0,044 |
| 2 · idem, matériau réaliste | 1,06 | +0,006 / −0,012 / −0,024 |
| 3 · tangentielle ajustée | 1,22 (23–26) | +0,009 / — / — |
| 5 · loi + cœur lobaire | 1,18 (27–36) | — / +0,017 / +0,015 |
| 5 · idem, matériau réaliste | 1,23 | — / +0,029 / +0,033 |

Rapports à l'atlas, A_t / A_e / T :

| | SG 26 | SG 30 | SG 36 |
|---|---|---|---|
| 0 · Tallinen 2016 | 1,19 / 1,01 / 1,11 | | |
| 2 · croissance mesurée | 0,95 / 0,97 / 1,04 | 0,83 / 0,94 / 1,17 | 0,72 / 0,93 / 1,30 |
| 2 · idem, matériau réaliste | 0,95 / 0,97 / 1,05 | 0,82 / 0,93 / 1,24 | 0,70 / 0,92 / 1,45 |
| 5 · loi + cœur lobaire | | 0,93 / 0,99 / 1,26 | 0,85 / 0,99 / 1,41 |
| 5 · idem, matériau réaliste | | 0,92 / 0,99 / 1,33 | 0,84 / 0,98 / 1,54 |

- L'atlas fœtal suit une loi de puissance, d'exposant 1,15 sur l'hémisphère (mur médian exclu de
  A_t), pas 1,25 : la loi adulte n'est pas reprise telle quelle, on compare à la droite de l'atlas.
- Tallinen tel quel s'en écarte vite (+25 % à SG 26) : trop d'aire plissée, trop tôt.
- Les modèles enrichis restent à quelques pour cent de la droite (ajusté : +4 % à SG 36), mais par
  compensation : aire corticale déficitaire (−15 % à SG 36) et cortex trop épais (+41 %).
  La loi d'échelle seule est donc un critère faible ; sa décomposition est informative.
- Le cortex du modèle ne réalise pas l'aire prescrite par a(t) (mesurée sur l'atlas) : −28 % à
  SG 36 sans ajustement, −15 % ajusté ; il s'épaissit d'autant plus que le tissu est incompressible
  (réaliste : +45 à +54 %). Cause à trancher : compression élastique tangentielle du cortex, ou
  croissance pondérée par g < 1 dans la profondeur de la couche (modèle de Tallinen).

### Origine du déficit d'aire (tools/area_budget.py)

Par sommet du cortex, l'aire de SG 21 multipliée par la croissance tangentielle du modèle,
(1 + g_m a)^2 avec g_m et a ceux du tétraèdre qui porte le triangle (aire prescrite), comparée à
l'aire de l'atlas et à l'aire réalisée (référence à croissance mesurée, out/remodel_a0 et out/real/ref).

| SG | prescrite / atlas | réalisée / prescrite | idem, réaliste | p10 / p50 / p90 par sommet |
|---|---|---|---|---|
| 24 | 0,999 | 0,985 | 0,984 | 0,97 / 0,99 / 1,00 |
| 26 | 0,996 | 0,957 | 0,954 | 0,92 / 0,96 / 0,98 |
| 28 | 0,993 | 0,901 | 0,895 | 0,84 / 0,91 / 0,95 |
| 30 | 0,989 | 0,838 | 0,828 | 0,75 / 0,85 / 0,91 |
| 33 | 0,985 | 0,764 | 0,750 | 0,66 / 0,77 / 0,87 |
| 36 | 0,983 | 0,731 | 0,716 | 0,62 / 0,73 / 0,84 |

- La loi de croissance prescrit bien l'aire de l'atlas (−1,7 % à SG 36 : pondération g_m et moyenne
  sur le tétraèdre) ; la seconde hypothèse est écartée.
- Le déficit est élastique : la surface est comprimée tangentiellement, de plus en plus avec l'âge
  (aire à 73 % de l'aire prescrite à SG 36, soit −15 % en longueur), un peu plus avec le matériau
  réaliste. Le cortex ne plisse pas assez pour loger la surface qu'il fabrique.
- L'épaisseur en excès en est le pendant : le produit aire × épaisseur reste proche de l'atlas
  (0,94 ; réaliste 1,02 à SG 36), le cortex comprimé s'épaissit à volume à peu près conservé.
- La compression est la plus forte sur les crêtes des gyri et là où la croissance est la plus rapide :
  à SG 36, aire réalisée / atlas 0,66 à moins de 1 mm de l'enveloppe convexe, 0,73 à 0,77 plus
  profond ; corrélation de Spearman avec a, −0,28.
- Conséquence : la profondeur plafonnée et le déficit d'aire sont un même défaut, un plissement
  insuffisant pour la croissance imposée. Pistes : contrainte du cœur trop forte (rigidité, croissance
  du cœur), rapport des modules, ou relaxation (remodelage) des contraintes dans le cortex.

Bilan d'aire des modèles ajustés (a ajusté, theta_full.npy) : ils prescrivent 5,5 % d'aire de plus que
l'atlas à SG 36 (1,055) et sont moins comprimés (réalisée / prescrite 0,80 contre 0,73) ; le gain
d'aire (0,72 → 0,85 de l'atlas) vient pour moitié de chacun. La carte inférée surestime donc un
peu la croissance tangentielle.

### Leviers du plissement (tools/perturb_eq.sh, tools/lever_eq.sh, tools/fold_quick.py)

Référence à croissance mesurée (μ_g/μ_w = 2, K = 5), un paramètre changé par continuation depuis son
équilibre, à SG 30 et 36. Rapports à l'atlas : aire corticale, aire de l'enveloppe, profondeur
moyenne à l'enveloppe ; écart rigide en mm.

| SG 36 | aire | enveloppe | profondeur | écart |
|---|---|---|---|---|
| référence | 0,718 | 0,930 | 0,588 | 2,81 |
| perturbée 0,3 mm, rééquilibrée | 0,718 | 0,930 | 0,588 | 2,81 |
| μ_w = 0,25 (rapport 4) | 0,755 | 0,956 | 0,669 | 2,97 |
| μ_w = 0,125 (rapport 8) | 0,796 | 0,985 | 0,750 | 3,23 |
| cœur × 1,25 | 0,781 | 1,038 | 0,548 | 3,68 |
| cœur × 0,75 / × 0,5 | 0,714 / 0,706 | 0,922 / 0,898 | 0,596 / 0,626 | non convergé |

À SG 30 : rapport 8, aire 0,829 → 0,871, profondeur 0,69 → 0,80, écart 1,87 → 1,99 mm ; cœur × 1,25,
aire 0,890 mais enveloppe 1,02 et profondeur 0,66.

- Stabilité : l'équilibre de SG 36 perturbé (champ lisse aléatoire, 0,3 mm rms, longueur d'onde 6 mm)
  revient exactement au même état. Le déficit n'est pas un artefact de suivi de branche du solveur.
  À SG 30 le rééquilibrage n'a pas convergé (165 éléments inversés), et 1 mm casse le maillage :
  non concluant.
- Le rapport des raideurs est le levier de la quantité de plissement : un cœur 8 fois plus mou que le
  cortex rend plus d'un quart de l'aire manquante et porte la profondeur moyenne de 59 à 75 % de l'atlas.
  Mais l'écart à l'atlas augmente (2,81 → 3,23 mm) : à croissance mesurée inchangée, les plis
  supplémentaires ne sont pas aux bons endroits ou déforment les lobes. À tester avec un réajustement.
- La croissance du cœur n'est pas le bon levier : plus forte, elle gonfle l'enveloppe (aire gagnée
  par dilatation, plis moins profonds, écart 3,68 mm) ; plus faible, l'aire baisse encore
  (équilibres non convergés, tendance seulement).
- Les rééquilibrages loin de l'équilibre de départ sont lents (30 min à 2 h contre 2 min), d'où un
  balayage limité à deux semaines.
