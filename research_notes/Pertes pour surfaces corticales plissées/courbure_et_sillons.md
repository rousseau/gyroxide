# Pertes fondées sur la courbure, la profondeur sulcale, le contenu spectral et la géométrie des sillons (et fonctions de coût du recalage cortical)

Contexte d'usage : ajustement adjoint d'un modèle mécanique de plissement (Tallinen 2016) à l'atlas de surfaces fœtales dHCP ; maillages à triangulation commune (correspondance issue d'un recalage sphérique, pas matérielle). Question centrale : pondérer la perte L2 par sommet par la courbure locale de la cible est-il sain, et quelles alternatives ?

Note méthodologique : 16 appels d'outils ; sources primaires lues en texte intégral pour Spherical Demons (Yeo 2010), MSM (Robinson 2014, 2018), SPANGY (Germanaud 2012), Vox2Cortex (Bongratz 2022) et Kerachni et al. 2026 (groupe Lefèvre/Rousseau). Fischl 1999 : texte intégral non accessible (PDF bloqué) ; seuls des énoncés secondaires (via Yeo 2010) sont cités.

---

## 1. L2 pondérée par la courbure (courbure moyenne, curvedness, shape index de la cible) : précédents et risques

### Takeaway
Le précédent le plus direct n'est pas en recalage sphérique mais en reconstruction de surface corticale par apprentissage : Vox2Cortex (CVPR 2022) pondère une distance de Chamfer par la courbure **de la vérité terrain uniquement**, plafonnée (w = min(1+H̄, 5)), précisément pour contrer l'effet lissant des régulariseurs dans les zones très plissées. C'est un rééquilibrage de la perte, pas une perte de forme : il accroît l'attention sur les crêtes/fonds de la cible mais reste aveugle aux plis parasites que le modèle crée là où la cible est lisse — ce qui exige un terme complémentaire symétrique (ou fondé sur les caractéristiques du modèle).

### Cited Findings
- Vox2Cortex définit une « curvature-weighted Chamfer loss » : L = (1/|P_gt|) Σ_{u∈P_gt} κ(u) min_v ‖u−v‖² + (1/|P_pred|) Σ_{v∈P_pred} κ(ũ) min_u ‖v−u‖², où ũ est le point vérité terrain le plus proche de v ; donc **les deux directions utilisent la courbure de la cible** — [Bongratz et al., CVPR 2022, arXiv:2203.09446](https://arxiv.org/abs/2203.09446)
- Poids : κ(p) = min{1 + H̄(p), κ_max} avec H̄ la courbure moyenne discrète (valeur absolue implicite, κ ≥ 0), κ_max = 5 — [Bongratz et al. 2022](https://arxiv.org/pdf/2203.09446v1)
- Justification des auteurs : « In regions with high curvature […] the smoothing effect of the regularization terms can lead to lower geometric accuracy » ; et choix délibéré : « we only use curvature weights from ground-truth points as the potentially wrong curvature in a prediction could be misleading » — [Bongratz et al. 2022](https://arxiv.org/pdf/2203.09446v1)
- Le supplément montre que, à erreur égale, le point de plus forte courbure reçoit un gradient plus grand (gradient d'un terme ∝ 4κ(a)(u−a)) ; l'ablation « κ ≡ 1 » sert de référence — [Vox2Cortex supplementary](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Bongratz_Vox2Cortex_Fast_Explicit_CVPR_2022_supplemental.pdf); [Bongratz et al. 2022](https://arxiv.org/pdf/2203.09446v1)
- Shape index SI = (2/π) arctan((κ1+κ2)/(κ2−κ1)), κ1 ≥ κ2, ∈ [−1,1] (sulcal pit [−1,−0.5], sulcal saddle, gyral saddle, gyral node [0.5,1]) ; curvedness CVD = sqrt((κ1²+κ2²)/2) = « puissance » / acuité du pli ; SI indépendant de la taille du maillage (Koenderink & van Doorn 1992, réf. [48]) — [Kerachni et al., arXiv:2607.08534, 2026](https://arxiv.org/abs/2607.08534)
- Dans les méthodes classiques de recalage cortical (FreeSurfer, Spherical Demons, MSM), la courbure n'est **pas** utilisée comme poids d'une distance de position, mais comme **caractéristique à apparier** (voir §2) — [Yeo et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2862393); [Robinson et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4190319)
- Une recherche ciblée n'a pas trouvé de méthode de recalage cortical sphérique pondérant explicitement une distance sommet-à-sommet par la courbure de la cible ; les usages proches sont : courbes sulcales géodésiques pondérées par la courbure (pour attirer les courbes au fond des vallées), coûts à deux termes lignes sulcales + courbure moyenne — [résultats de recherche agrégés, ex. Shi/Joshi fundi, IPMI 2013](https://iacl.ece.jhu.edu/proceedings/ipmi2013/papers/7917/79170364.pdf)

### Inferences
- **Sain comme repondération, insuffisant comme seul critère.** Avec correspondance fixe (même triangulation), w_i‖x_i − y_i‖² avec w_i = f(courbure cible) est un L2 à poids constants (les poids ne dépendent pas des variables) : différentiabilité triviale, aucun bruit de second ordre dans le gradient — c'est son principal avantage par rapport à une perte *sur* la courbure du modèle.
- **Asymétrie :** les poids ne sont élevés que là où la cible plie. Un pli parasite du modèle dans une région lisse de la cible n'est pénalisé qu'avec le poids de base (1 dans Vox2Cortex). Vox2Cortex accepte ce biais (sa seconde direction de Chamfer utilise aussi κ de la cible). Pour l'ajustement d'un modèle mécanique, où des plis spurious sont un mode de défaillance attendu, il faut un plancher de poids non négligeable (w ≥ 1, plafonné comme κ_max) et/ou un terme symétrique (ex. sur la courbure du modèle, ou un terme global type spectre de puissance §3).
- **Correspondance non matérielle :** la correspondance vient d'un recalage sphérique qui aligne lui-même sulc/curv ; si un pli simulé est décalé de quelques mm, le L2 pondéré pénalise fortement crête et fond (là où les poids sont maximaux) — il *amplifie* la sensibilité au décalage, l'inverse de ce qu'on veut si on vise la « présence » d'un pli plutôt que sa position exacte. Lisser les poids (diffusion sur la surface cible, échelle ~ demi-longueur d'onde des plis) atténue ce problème.
- Préférer curvedness (≥ 0, invariant au signe crête/fond) ou |H| plafonné ; SI est sans dimension mais **indéfini/instable près des points ombilicaux** (κ1 ≈ κ2, dénominateur nul) et sur les zones plates (bruit amplifié) : mauvais candidat pour des poids non lissés.

### Gaps
- Pas trouvé de précédent publié de L2 *à correspondance fixe* pondérée par la courbure pour l'appariement de modèles de croissance corticale ; pas d'étude de sensibilité au décalage des plis pour ce type de pondération.
- Texte original de Koenderink & van Doorn 1992 (« Surface shape and curvature scales », Image Vis. Comput.) non consulté directement.

---

## 2. Pertes directes sur cartes de courbure / profondeur sulcale ; coûts de FreeSurfer, Spherical Demons, MSM

### Takeaway
Tous les recalages corticaux de référence apparient des **cartes scalaires de pliage lissées** (sulc / average convexity, courbure moyenne de surfaces lissées ou gonflées), avec une stratégie grossier→fin et des régulariseurs de distorsion (métrique/aire chez FreeSurfer, spline/difféomorphe chez Spherical Demons, énergie de déformation hyperélastique chez MSM 2018). La leçon transférable : comparer des caractéristiques multi-échelles normalisées (par la variance de l'atlas) plutôt que la courbure brute du maillage fin.

### Cited Findings
- **FreeSurfer (Fischl et al. 1999)** : construit un motif de plissement moyen comme fonction sur la sphère unité et aligne non rigidement chaque sujet à cette moyenne — [Fischl et al., Hum Brain Mapp 1999](https://pubmed.ncbi.nlm.nih.gov/10619420/). Selon Yeo et al. : « The FreeSurfer registration algorithm uses the same similarity measure as Demons, but explicitly penalizes for metric and areal distortion », distorsion mesurée sur la surface corticale originale — [Yeo et al., IEEE TMI 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2862393)
- **Spherical Demons (Yeo et al. 2010)** : objectif = terme d'intensité pondéré par l'inverse de la variance de l'atlas (SSD normalisée, atlas = moyenne et écart-type par sommet) + terme de couplage transformation cachée/warp + régulariseur en norme hilbertienne approximé par lissage itératif des champs tangents (γ = 1, 10 itérations) ; optimisation Gauss-Newton/Levenberg-Marquardt ; ne tient pas compte des propriétés métriques de la surface originale (contrairement à FreeSurfer) — [Yeo et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2862393)
- Caractéristiques Spherical Demons : « mean curvature of the cortical surfaces, mean curvature of the inflated cortical surfaces and average convexity », average convexity ≈ profondeur sulcale ; multi-échelle ic4 (2 562 sommets) → ic7 (163 842), 15 itérations par niveau + recherche rotationnelle ; Dice 89.6 vs 88.9 (FreeSurfer) hémisphère gauche ; < 5 min vs > 1 h — [Yeo et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2862393)
- Avec l'atlas comme image mobile, le terme log σ entre dans l'objectif et sa contribution au Hessien est supprimée car négative et déstabilisante pour Gauss-Newton — [Yeo et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2862393)
- **MSM (Robinson et al. 2014)** : MRF discret, coût = Σ coûts de données par point (similarité de patchs : CC, NSSD, NMI, α-MI) + potentiels pairs de régularisation (norme de Frobenius du log de la rotation relative entre points de contrôle voisins) ; optimisé par Fast-PD (potentiels non métriques) ; grilles icosaédriques d'ordre 2→5 ; déplacement max 0.4 × espacement ; caractéristiques : sulc, courbure moyenne (« finer-grained indicator »), myéline, rs-fMRI — [Robinson et al., NeuroImage 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4190319)
- MSM 2014 reconnaît : « cortical folding is not consistent across subjects in many brain regions » et les plis ne coïncident pas toujours avec la microarchitecture ; la variabilité des plis fins rend les moyennes de courbure plus floues — [Robinson et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4190319)
- **MSM higher-order (Robinson et al. 2018)** : régularisation par énergie de déformation hyperélastique sur triangles (gradient de déformation F projeté dans le plan tangent, élongations principales λ1, λ2, J = λ1λ2, R = λ1/λ2) : W = (μ/2)(R^k + R^−k − 2) + (κ/2)(J^k + J^−k − 2), pénalisant symétriquement en log l'expansion et la contraction ; k=1 ≈ néo-hookéen compressible modifié ; données par triplets (CC entre patchs) ; réglage κ=1.6, μ=0.4, k=2 — [Robinson et al., NeuroImage 2018](https://kclpure.kcl.ac.uk/ws/files/81229914/multimodal_surface_matching_minorrevision.pdf) (DOI 10.1016/j.neuroimage.2017.10.037)
- Motivation 2018 : la régularisation originale ne pénalisait pas uniformément toutes les distorsions, laissant des pics de distorsion non plausibles — [Robinson et al. 2018 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5991912)
- **Profondeur sulcale** : la Depth Potential Function (Boucher, Whitesides & Evans 2009) est une méthode de type diffusion (équation de Poisson écrantée), pas une distance à l'enveloppe ; elle interpole entre courbure moyenne (α→∞) et average convexity (α→0) ; non robuste au changement d'échelle isométrique (corrigé par DPF*, Dieudonné, Auzias & Lefèvre 2025) — [BrainVISA DPF](https://brainvisa.info/axon/en/processes/DepthPotentialFunction.html); [Dieudonné et al., arXiv:2501.05436](https://arxiv.org/html/2501.05436v2); [Dieudonné, slides AMU 2023](https://irmf.int.univ-amu.fr/MyUploads/Dieudonne_Sulcal_Depth_RMN_09_03_2023.pdf)
- Profondeurs par distance : euclidienne sous-estime les plis très convolués, géodésique surestime les bassins peu profonds ; l'enveloppe convexe exacte surestime la profondeur dans les régions concaves (orbitofrontal, temporal interne), d'où des proxys par fermeture morphologique, définis de façon incohérente entre études — [Dieudonné et al. 2025](https://arxiv.org/html/2501.05436v2); [slides AMU 2023](https://irmf.int.univ-amu.fr/MyUploads/Dieudonne_Sulcal_Depth_RMN_09_03_2023.pdf)

### Inferences
- **Perte sur cartes de caractéristiques à correspondance fixe :** L = Σ_s Σ_i a_i (f_s(X)_i − f_s(Y)_i)² / σ_{s,i}², f_s = courbure moyenne (cotangent Laplacien) lissée à l'échelle s, ou DPF/sulc. Avec correspondance figée, c'est l'analogue du terme de données Spherical Demons/FreeSurfer mais sans déformation de la paramétrisation : un décalage de pli devient une erreur pleine (pas de « tolérance »). Le multi-échelle (lissage fort d'abord, comme sulc → curv dans FreeSurfer) élargit le bassin d'attraction.
- **Différentiabilité :** H = ‖L_cot X‖/(2A) (ou la projection sur la normale pour le signe) est différentiable presque partout mais dépend des cotangentes (singulières pour des triangles dégénérés) et de l'aire de Voronoï mixte ; le gradient est d'ordre 2 en X et donc **bruité et amplifie les hautes fréquences**. Recommandation : calculer H, puis le lisser par diffusion (heat kernel / résolution implicite (M + tL)⁻¹) sur le maillage *cible* fixe (opérateur constant) pour garder un gradient linéaire propre ; ou utiliser la DPF qui est elle-même une diffusion (déjà régularisante). sulc FreeSurfer (déplacement cumulé lors de l'inflation) n'est pas commodément différentiable ; la DPF l'est (système linéaire).
- Pondérer par 1/σ² de l'atlas (comme Yeo) est naturel si l'atlas dHCP fournit une variance par sommet ; sinon, normaliser par l'énergie globale de la caractéristique.
- Les régulariseurs hyperélastiques de MSM 2018 sont de la même famille que l'énergie élastique de Tallinen : pas pertinents comme perte, mais confirment que dans notre cas la « déformation » est portée par le modèle physique, pas par un champ de recalage.

### Gaps
- Équations exactes de Fischl 1999 (termes J_d, J_A, terme de données et calendrier sulc→curv) non vérifiées en texte intégral (PDF inaccessible). Le calendrier grossier-fin sulc puis curv est de notoriété FreeSurfer (mris_register) mais non sourcé ici.
- Pas trouvé de travail évaluant une perte *sur la courbure du modèle* dans un ajustement adjoint de modèle de croissance.

---

## 3. Pertes spectrales : fonctions propres de Laplace–Beltrami, SPANGY, puissance par bande

### Takeaway
SPANGY décompose la courbure moyenne sur les fonctions propres de Laplace–Beltrami (LB) du maillage du sujet et regroupe la puissance en 7 bandes à doublement de fréquence (B0–B6) ; B4–B6 portent ~93 % de la puissance et correspondent aux plis primaires/secondaires/tertiaires. La puissance par bande est **invariante aux décalages de position des plis** (et à la rotation des bases propres dégénérées), donc complémentaire d'une perte par sommet — mais elle ne contraint pas la localisation. Le groupe Lefèvre/Rousseau l'utilise déjà (avec le shape index comme proxy) pour optimiser l'épaisseur corticale d'un modèle de plissement sur l'atlas dHCP.

### Cited Findings
- SPANGY : C_i = ∫_M C φ_i, spectre brut C_i² ; bandes définies par des fréquences spatiales F(i) = sqrt(λ_i)/(2π) = 1/WL(i), limites 2^k F(1) (doublement de fréquence entre bandes) ; puissance de bande BSC(k) = Σ_{i∈bande k} C_i² ; « Total Folding Power » via Parseval ; ~5000 fonctions propres calculées — [Germanaud et al., NeuroImage 2012](https://www.unicog.org/publications/germanaudLefevre_spangyFoldingAnalysis_nimg12.pdf)
- B4–B6 = 93 % de la puissance analysée, liées aux variations de courbure dues aux plis ; bandes basses = forme globale ; segmentation B4/B5/B6 ↔ plis 1er/2e/3e ordre ; puissance totale de plissement en allométrie positive (B4 constant, B5 comme le total, B6 bien plus vite) — [Germanaud et al. 2012](https://www.unicog.org/publications/germanaudLefevre_spangyFoldingAnalysis_nimg12.pdf)
- Limite clé : « since the eigenfunctions are defined on a per-individual basis, there is no mathematically exact matching of eigenfunctions of the same i-order (Knossow et al., 2009; Lombaert et al., 2011) » → regroupement en bandes comme réduction de dimension et lissage — [Germanaud et al. 2012](https://www.unicog.org/publications/germanaudLefevre_spangyFoldingAnalysis_nimg12.pdf)
- La synthèse bande par bande est un filtrage passe-bande ; la segmentation de la bande la plus haute est bruitée (spectre tronqué) — [Germanaud et al. 2012](https://www.unicog.org/publications/germanaudLefevre_spangyFoldingAnalysis_nimg12.pdf)
- Kerachni et al. (2026, FetalFoldSim, FEniCS, maillage initial dHCP 21 SA, simulation 22–29 SA) : métriques locales SI, curvedness et SpAnGy ; bandes étendues B−6…B6 (signe de la courbure) ; ils remplacent la courbure moyenne par le **shape index** comme proxy car indépendant de la taille du maillage ; optimisation de l'épaisseur H0 ∈ {0.8, 1.5, 2.25, 6} mm par similarité spectrale B4–B6 avec l'atlas à 28 SA ; à 28 SA, B5/B6 à interpréter prudemment, interprétation centrée sur B4 — [Kerachni et al., arXiv:2607.08534](https://arxiv.org/abs/2607.08534)
- Kerachni et al. justifient les métriques locales : les métriques globales (GI) ne rendent pas compte du gradient spatio-temporel hétérogène du plissement — [Kerachni et al. 2026](https://arxiv.org/abs/2607.08534)

### Inferences
- **Perte spectrale candidate :** L_spec = Σ_k (log P_k(X) − log P_k(Y))², P_k = puissance de bande de H (ou SI) sur la base LB. Invariances : rotation/réflexion, décalage des plis, ambiguïtés de signe et de sous-espaces propres dégénérés (puisque la somme sur une bande est invariante si la bande contient des sous-espaces propres entiers). Sensibilités : nombre et amplitude des plis à chaque échelle — exactement ce que la L2 pondérée ne capture pas (pénalise les plis parasites par excès de puissance).
- **Différentiabilité :** deux options. (a) Base LB **figée sur la cible** (maillage partagé → base fixe, projection linéaire de H(X)) : gradient simple, coût = un produit matrice dense N×K par itération ; c'est une analyse de Fourier « dans le repère de la cible », non invariante à la position mais lisible par échelle. (b) Base recalculée sur le modèle : gradient des valeurs/vecteurs propres (perturbation, instable aux valeurs propres quasi dégénérées) — coûteux et fragile ; à éviter dans un adjoint. Le choix (a) donne en plus une décomposition multi-échelle de l'erreur par sommet (filtres passe-bande).
- La perte spectrale seule est sous-déterminée (beaucoup de surfaces ont le même spectre de bandes) : à utiliser en complément d'un terme local.
- SI comme proxy : sans dimension (bon pour comparer des tailles différentes) mais instable sur zones plates/ombilicales ; H plus lisse mais dépend de l'échelle — à normaliser par la taille (ex. H × sqrt(A)).

### Gaps
- Le détail exact de la fonction de similarité spectrale utilisée par Kerachni et al. pour choisir H0 n'a pas été extrait (seules les sections de méthode lues).
- Aucune étude trouvée sur la différentiation de pertes SPANGY dans une boucle d'optimisation par gradient.

---

## 4. Repères sulcaux : fonds, lignes, sulcal pits, graphes et bassins

### Takeaway
Les sulcal pits (maxima locaux de profondeur, via DPF + watershed) sont des repères stables, y compris chez le fœtus (Yun et al. 2020), et appariables par graphes ; mais leur extraction est combinatoire (watershed, seuils, fusion de bassins) donc **non différentiable** et sensible aux paramètres. Ils servent mieux de **métrique d'évaluation** ou de terme doux (ex. profondeur/DPF échantillonnée aux positions des pits de la cible) que de perte d'optimisation.

### Cited Findings
- Auzias et al. 2015 : sulcal pits = points de profondeur maximale dans les plis, définis algorithmiquement ; influence des paramètres de la méthode de référence (Im et al. 2010) peu évaluée → cadre d'optimisation des paramètres — [Auzias et al., NeuroImage 2015, PMID 25676916](https://busqueda.bvsalud.org/portal/resource/es/mdl-25676916)
- Implémentation BrainVISA : profondeur par DPF, pits par watershed sur la carte de profondeur ; « parameters have a direct influence on the results » — [BrainVISA SulcalPitsExtraction](https://brainvisa.info/axon/en/processes/SulcalPitsExtraction.html)
- Im et al. 2010 : positions des pits relativement invariantes entre individus, fortes fréquence/densité dans des zones focales — [Im et al., Cereb Cortex 2010](https://labs.childrenshospital.org/neuroim/publications/spatial-distribution-deep-sulcal-landmarks-and-hemispherical-asymmetry-cortical)
- Yun et al. 2020 (48 fœtus) : position et variance spatiale des pits fœtaux similaires à l'adulte, temporellement stables malgré la croissance ; ordre d'émergence régionalement varié — [Yun et al., Cereb Cortex 2020](https://profiles.wustl.edu/en/publications/temporal-patterns-of-emergence-and-spatial-distribution-of-sulcal/)
- Graphes sulcaux : pits = nœuds ; similarité par appariement spectral de graphes avec attributs (position 3D, profondeur, aire) et relations — [Im et al., NeuroImage 2011](https://labs.childrenshospital.org/neuroim/publications/quantitative-comparison-and-analysis-sulcal-patterns-using-sulcal-graph) ; bassins sulcaux comme nœuds, arêtes entre bassins adjacents (Takerkart et al. 2017), multi-graph matching — [Yadav et al., arXiv:2301.13532](https://ar5iv.labs.arxiv.org/html/2301.13532)
- Lignes sulcales : extraction par distance géodésique anisotrope / densité de chemins géodésiques (Le Troter, Auzias & Coulon 2012) — [Children's Hospital neuroim](https://labs.childrenshospital.org/neuroim/publications/automatic-extraction-sulcal-lines-cortical-surfaces-based-anisotropic-geodesic); [BrainVISA/Auzias](https://brainvisa.info/axon/en/processes/SulcalPitsExtraction.html)

### Inferences
- Version différentiable : pour chaque pit p de la cible (calculé une fois), pénaliser −DPF(X)_p ou (DPF(X)_p − DPF(Y)_p)², et/ou exiger un minimum local de H/SI au voisinage (softmin sur un disque géodésique). Les fonds sulcaux de la cible peuvent servir de support de poids (équivalent à §1 mais sur une ligne).
- Distance entre ensembles de pits (modèle vs cible) ou entre lignes de fond : Chamfer/Hausdorff — non différentiable via l'extraction, mais excellent **critère d'évaluation** robuste aux petits décalages (tolérance explicite en mm).
- Une limitation spécifique au fœtus : avant ~28 SA, de nombreux sillons ne sont pas encore formés ; l'appariement de graphes avec nœuds manquants pénaliserait des différences de timing plutôt que de forme.

### Gaps
- Pas de travail trouvé utilisant des pits ou lignes sulcales comme perte différentiable pour un modèle de croissance.
- Coûts exacts d'appariement de lignes sulcales en recalage (ex. Joshi, Shi) non extraits.

---

## 5. Métriques spécifiques à la gyrification : LGI, folding power, Toro, Lefèvre & Mangin, Germanaud

### Takeaway
Les métriques de gyrification (LGI de Schaer, folding power de SPANGY, curvedness intégrée) quantifient la *quantité* de plissement locale ou par échelle, avec tolérance aux décalages par construction (intégration sur des voisinages) ; elles sont naturelles comme termes globaux/régionaux mais ne déterminent pas la topographie des plis.

### Cited Findings
- LGI (Schaer et al., IEEE TMI 2008) : mesure surface-based de gyrification calculée localement en des milliers de points via des ROI circulaires 3D (rapport surface piale / surface d'une enveloppe externe dans la ROI) ; implémentée dans FreeSurfer (`-localGI`, mris_compute_lgi) ; le GI 2D classique est biaisé par l'orientation des coupes — [Schaer et al. 2008, EPFL](https://infoscience.epfl.ch/record/91007/files/04359040.pdf?version=2); [FreeSurfer LGI wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/LGI?action=raw)
- Folding power SPANGY et son allométrie (voir §3) — [Germanaud et al. 2012](https://www.unicog.org/publications/germanaudLefevre_spangyFoldingAnalysis_nimg12.pdf)
- Curvedness = « power of folding », sharpness du pli ; utilisée avec SI et SpAnGy pour comparer simulations et atlas dHCP — [Kerachni et al. 2026](https://arxiv.org/abs/2607.08534)
- Kerachni et al. citent GI comme métrique de référence fréquente mais insuffisante localement — [Kerachni et al. 2026](https://arxiv.org/abs/2607.08534)

### Inferences
- LGI repose sur une enveloppe externe (fermeture morphologique) et des ROI de 25 mm (adulte) : non différentiable simplement et échelle inadaptée au fœtus ; un substitut différentiable est le rapport d'aire local surface/surface lissée (diffusion forte), ou l'intégrale locale de curvedness.
- Un objectif composite raisonnable pour l'adjoint : (i) L2 position (pondérée doucement par curvedness lissée de la cible, plancher ≥ 1) ; (ii) erreur multi-échelle sur H ou DPF lissées, projetées sur une base LB figée de la cible ; (iii) terme de puissance par bande (B4 surtout à 28 SA) pénalisant excès/défaut de plissement indépendamment de la position ; évaluation par pits/SI/LGI.

### Gaps
- Travaux de Toro (prédiction du plissement, ex. Toro & Burnod 2005) et Lefèvre & Mangin (2010, modèle de réaction-diffusion, PLoS Comput Biol) non vérifiés en source primaire dans cette recherche — à citer avec prudence.
- Pas de validation trouvée de LGI sur des surfaces fœtales dHCP pour le choix d'échelle des ROI.
