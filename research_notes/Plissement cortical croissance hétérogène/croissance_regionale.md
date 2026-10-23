# Regional growth heterogeneity in mechanical models of cortical folding

Scope: who has put spatially varying growth (tangential cortical, radial/thickness, subcortical/subplate/white-matter) into brain-folding mechanics models, how, with what results, and whether the heterogeneity is treated as cause or consequence of folding. Comparison target: our Tallinen-2016-type neo-Hookean model fitted by an adjoint to the dHCP fetal atlas (SG 26→28), central and calcarine sulci. In our fits tangential growth alone cannot deepen the sulci. We also need two more fields: (i) differential cortical thickening, thicker at crests and thinner at fundi; (ii) a core/subplate growth correction, negative under fundi and positive under crests. A 6-coefficient depth law for these fields is shared by both sulci.

Verification status: unless marked otherwise, each reference below was resolved in this session through a PubMed record (DOI listed), the PMC full text, or the publisher page. Items marked **[not re-verified]** come from background knowledge or secondary citation only.

---

## Q1. Which models use regionally varying tangential growth?

### Takeaway
Regionally varying tangential growth runs from the conceptual (Ronan 2014, Toro & Burnod 2005) to data-driven whole-brain models (Hou et al. 2025 Soft Matter; Hou et al. 2026 Nat Commun), with growth laws fitted parcel by parcel to dHCP surface-area trajectories. The canonical Tallinen 2014/2016 models use spatially uniform tangential growth. Recent work agrees on three points: tangential growth is the dominant driver; heterogeneity mostly controls fold placement, onset timing and regional realism, and much less the wavelength; and no forward model with parcel-level growth reproduces sulcal depth everywhere.

### Cited Findings
**Conceptual and early models**
- Toro & Burnod (2005) modelled the cortex as a closed 2D surface pulled radially by glial/axonal fibres, with elasticity, plasticity and surface growth. In this model convolutions are "a natural consequence of cortical growth". They introduced "geometric, mechanic and growth asymmetries" to mimic early regionalization, and reproduced gradients in degree of convolution and the primary/secondary/tertiary sequence. — [Toro & Burnod 2005, Cereb Cortex 15:1900, doi:10.1093/cercor/bhi068](https://doi.org/10.1093/cercor/bhi068)
- Ronan et al. (2014; online 2013) proposed differential tangential expansion as a sufficient mechanism for gyrification. They used intrinsic (Gaussian) curvature as an MRI proxy of differential expansion. Its pattern and local GI were consistent across independent datasets, and intrinsic curvature predicted folding globally and regionally. The authors argue this explains the co-localisation of cytoarchitecture and folds. This is an imaging and geometric argument, not an FE simulation. — [Ronan et al. 2014, Cereb Cortex 24:2219, doi:10.1093/cercor/bht082](https://doi.org/10.1093/cercor/bht082); review: [Ronan & Fletcher 2015, Brain Struct Funct, doi:10.1007/s00429-014-0961-z](https://doi.org/10.1007/s00429-014-0961-z)
- Bayly et al. (2013) used tangential cortical expansion with stress-dependent growth of the deeper layers. The wavelength is set by the ratio of cortical growth rate to stress-induced growth rate: faster cortical growth gives shorter wavelengths, slower growth gives a near-lissencephalic surface. No skull or inner constraint is required. Initial shape and mechanical heterogeneity modulate the pattern. — [Bayly et al. 2013, Phys Biol 10:016005 (PMC3616769)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3616769)

**Tallinen models (uniform tangential growth)**
- Tallinen et al. (2014) showed that gyrification arises as a nonlinear consequence of a mechanical instability: grey matter expands tangentially while white matter constrains it. Patterns depend on relative expansion and relative thickness, and the simulations give cusped sulci and smooth gyri. — [Tallinen et al. 2014, PNAS (PMC4156754)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4156754)
- Tallinen et al. (2016) used a 3D-printed gel and simulations from a smooth fetal MRI brain. They conclude that "size, shape, placement and orientation of the folds arise through iterations and variations of an elementary mechanical instability modulated by early fetal brain geometry", i.e. without regional growth control. — [Tallinen et al. 2016, Nat Phys 12:588 (HAL record)](https://hal.archives-ouvertes.fr/hal-01342578), doi:10.1038/nphys3632
  - **[not re-verified]** The 2016 tangential growth is a function of depth only, g = 1 + α(t)/(1+exp(10(y/H(t)−1))), and is spatially uniform along the surface. The core grows more slowly. Regional growth variation is discussed as a limitation. The HAL page was bot-blocked, so this formula could not be checked in this session.
- Hou et al. (2026) list Tallinen 2016, Alenyà et al. 2022 (Brain Multiphysics), Solhtalab et al. 2025 (HBM) and Wang et al. 2021 (Sci Rep) as realistic-geometry models with "globally uniform growth". — [Hou et al. 2026, Nat Commun (PMC13490475)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13490475/)
  - Alenyà 2022 was not opened in this session **[not re-verified]**.

**Data-driven regional growth (2025–2026)**
- **Hou et al. 2025, Soft Matter, "Role of data-driven regional growth model in shaping brain folding patterns"** — [doi:10.1039/d4sm01194e, PMC11718650](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718650/)
  - *Data.* Surface area from dHCP (549 scans, 29–45 PMW) and BCP (488 scans, 41–144 PMW). The cortex is split into 18 NMF parcels with shared area trajectories; 5 parcels were studied.
  - *Growth laws.* Fitted by symbolic regression (PySR).
    - Tangential: (a·t+b)·tanh(c·t)+1, R² > 0.85.
    - Radial (thickness): polynomial, inverted-U, noisier fit. Thickness data come from a separate postnatal dataset linearly extrapolated back to the perinatal period, which the authors flag as a limitation.
    - Growth is orthotropic (in-plane vs out-of-plane), implemented in ABAQUS through a thermal analogy.
  - *Geometry and material.* ~80×80 mm slab, 2 mm cortex over 50 mm core, real regional surface smoothed into a plane. Neo-Hookean, μ_cortex = 0.31 kPa, μ_WM = 0.45 kPa.
  - *Results.*
    - Regional growth gives lower MAPE than isotropic or constant-tangential models.
    - Same final growth with different time courses (linear vs Gompertz) gives nearly identical folds: magnitude matters more than trajectory.
    - Tangential growth affects folding more than radial growth.
    - Mean curvature was poorly predicted (R² ≈ 0).
    - A multi-region model (parcels 1, 4, 11) matched better but still missed the central sulcus.
    - The authors say intrinsic factors (initial curvature, stiffness, thickness ratio) decide where bulking starts, while growth mostly sets timing and extent.
- **Hou et al. 2026, Nat Commun 17, "Biophysical modeling of anatomically realistic prenatal cortical folding development"** (Hou, Wu, …, Razavi, Liu, Kuhl, Li, X. Wang) — [doi:10.1038/s41467-026-75582-9; PMID 42463653; PMC13490475](https://pmc.ncbi.nlm.nih.gov/articles/PMC13490475/)
  - *Data.* 120 dHCP fetal scans, 22–40 GW (iBEAT V2). 36 NMF parcels (18/hemisphere; the Methods mention 19, an internal inconsistency). Tangential and radial growth are normalised to 22 GW and fitted by symbolic regression.
    - Tangential growth is near-linear and much faster than radial growth, highest in temporal, parietal and peri-Sylvian regions.
    - Radial growth varies by region: some regions rise steadily, some plateau, some peak around 31 GW and then decline.
  - *Model.* Realistic 22 GW geometry, about 1.6 M hexahedra. Near-incompressible neo-Hookean cortex and subcortex (0.65 vs 0.68 kPa). The cortex grows orthotropically; the subcortex grows isotropically and homogeneously. A uniform pressure stands in for the meninges and skull.
  - *Four scenarios compared.*
    - Isotropic uniform: bulky gyri, few sulci.
    - Tangential uniform: under-developed folds near the central sulcus.
    - Tangential regional and orthotropic regional: best regional GI, thickness and sulcal count. Orthotropic regional was retained because thickness matched better.
  - *Validation.*
    - Global GI rises near-linearly after ~27 GW.
    - Six landmarks are reproduced in location and relative size: central, postcentral, intraparietal, superior temporal, lateral and inferior frontal sulci.
    - Primary folds stabilise around 32 GW; sulcal depth is biphasic, with a transition near 29 GW.
    - Vertex-wise shape-index similarity exceeds 0.8.
    - Sulcal depth deviates most in peri-Sylvian, interhemispheric and isthmus-cingulate regions.
  - *Sensitivity.* Thicker cortex gives fewer, deeper folds. Changing the trajectory mostly shifts onset timing. Halving magnitude gives less developed folds. Subcortical growth mainly changes volume and sulcal depth.
  - *Malformations.* Lissencephaly, pachygyria and polymicrogyria are reproduced qualitatively.

**Other regional, local or perturbation approaches**
- Choi, …, Walsh, Mahadevan (2025) built gel and FE models of ferret gyrification. Using genetically manipulated ferrets, they identify cortical thickness and cortical expansion rate as the primary drivers of dysmorphogenesis, and extend this to human malformations. Their framing: genetics is the proximal cause, acting mechanically through geometry and differential growth. — [Choi et al. 2025, eLife, doi:10.7554/eLife.107141](https://doi.org/10.7554/eLife.107141)
- Zarzor, Blümcke & Budday (2023) coupled cell-density advection–diffusion (VZ and OSVZ sources, migration, cortical diffusion) to finite growth, in 2D quarter-circles and 3D half-spheres. Growth is driven by cell density, tangential ≫ radial in the cortex and isotropic below. With a gradient or random OSVZ proliferation, the first folds appear where proliferation is highest, but final wavelength and complexity are essentially unchanged. — [Zarzor et al. 2023, eLife 12:e82925, doi:10.7554/eLife.82925](https://elifesciences.org/articles/82925)
  - Extension to four cell types (RGC, IPC, oRG, neurons) with comparison to fetal MRI: [Zarzor et al. 2024, Sci Rep, doi:10.1038/s41598-024-75952-7](https://doi.org/10.1038/s41598-024-75952-7)
- Chavoshnejad et al. (2023) used bilayer spheres with initial undulations and uniform or heterogeneous axon-fibre density in the white matter. Differential tangential growth is the "inducer" of folding. Large initial undulations and axon fibres then regulate where gyri and sulci form: dense-fibre locations end up in gyri, and outward/inward undulations become gyri/sulci. Without fibre concentrations, differential growth gives only random patterns. — [Chavoshnejad et al. 2023, Sci Rep, doi:10.1038/s41598-023-40086-9](https://doi.org/10.1038/s41598-023-40086-9)
- Wang, Lefèvre, Bohi, Harrach, Dinomais & Rousseau (2021) compared tangential vs isotropic growth modes on ellipsoids. Growth mode barely changes complexity. Initial geometry (slenderness) changes fold orientation and depth: slender shapes give more folds along the long axis and deeper sulci. Thinner initial cortex gives higher spatial frequency and shallower sulci. — [Wang et al. 2021, Sci Rep 11:7686, doi:10.1038/s41598-021-87124-y](https://doi.org/10.1038/s41598-021-87124-y)
- Solhtalab et al. (2025) built an image-based, true-scale model from the first time point of longitudinal fetal MRI. It predicts later sulcal-pit distributions, and the pits stay stable. The growth is not regionalised (Hou 2026 classifies it as uniform). — [Solhtalab et al. 2025, Hum Brain Mapp, doi:10.1002/hbm.70332](https://doi.org/10.1002/hbm.70332)

**Measurements that motivate regional tangential growth**
- Garcia et al. (2018) tracked 30 preterm infants scanned 2–4 times between 28 and 38 wk PMA, with surfaces registered by aMSM.
  - From 30 to 38 wk, expansion was highest in lateral parietal, occipital and temporal cortex, and lower in medial and insular cortex.
  - Expansion hotspots and folding hotspots correlate only moderately (r = 0.46 left, 0.42 right).
  - The authors argue that uniform-growth models on realistic geometry do not reproduce conserved primary folds, but they ran no simulation. Gyral-vs-sulcal growth could not be resolved.
  - — [Garcia et al. 2018, PNAS 115:3156, doi:10.1073/pnas.1715451115 (PMC5866555)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5866555/)
- Knutsen et al. (2012) measured ferret growth between P14 and P28. Relative cortical growth increases with geodesic distance from the origin of the transverse neurogenetic gradient, and is related to changes in diffusion FA. — [Knutsen et al. 2012, Cereb Cortex, doi:10.1093/cercor/bhs042](https://doi.org/10.1093/cercor/bhs042)
  - Method paper: [Knutsen et al. 2010, J Biomech Eng, doi:10.1115/1.4002430](https://doi.org/10.1115/1.4002430)

### Inferences
- No model found uses an inverse or adjoint method to *infer* spatially varying growth fields from consecutive fetal surfaces. Existing data-driven work (Hou 2025/2026) prescribes parcel-wise growth from area and thickness trajectories in a forward way. Parcels are coarse (18/hemisphere) and have no gyral/sulcal resolution, so they cannot capture crest-vs-fundus differences like ours. Our approach looks methodologically distinct.
- Hou 2026 reports that uniform tangential growth under-develops the central-sulcus region, and that sulcal depth is the least well-matched metric. Both are consistent with our finding that tangential growth alone cannot deepen the central/calcarine sulci.

### Gaps
- The full Tallinen 2016 growth law and its statements on regional variation could not be re-checked (HAL bot-blocked).
- Holland et al. "gyrification by constrained cortical expansion": no paper by Holland with this title was found. The title "Gyrification from constrained cortical expansion" belongs to Tallinen, Chung, Biggins & Mahadevan, PNAS 2014. The premise in the question likely conflates the two.
- The user-provided description of PMC11718650 as "Wang et al." is inaccurate: the first author is Jixin Hou; Xianqiao Wang is the corresponding author.

---

## Q2. Which models vary cortical thickness / radial growth, or subcortical (subplate, white-matter) growth? Effects on location, depth, wavelength

### Takeaway
Cortical thickness heterogeneity is mostly modelled in two ways. The first is a prescribed perturbation, which pins fold location: a thin spot becomes a sulcus, and deep primary sulci followed by secondary folds emerge. The second is an *emergent* outcome of folding, giving thicker gyri and thinner sulci (Holland/Budday/Kuhl), or growth coupled to curvature (Wang, Demirci & Holland 2020). Subcortical heterogeneity has been modelled as stress-dependent subcortical growth (Bayly 2013; Balouchzadeh 2023), as axonal tension or fibre density (Wang 2024; Chavoshnejad 2023), and as regional OSVZ proliferation (Zarzor 2023). None of these combines a thickness field and a subcortical growth field as joint unknowns fitted to data.

Two findings bear directly on our crest/fundus fields:
- Ferret data show that thickening precedes folding in regions destined to fold, and gyral thickness later outpaces sulcal thickness (Garcia, Basinski & Kroenke 2025).
- The subplate is reported to be thicker under future gyri and to dissolve first under sulci (Rana et al. 2019 review).

### Cited Findings
**Cortical thickness: a cause (prescribed) or a consequence (emergent)?**
- Costa Campos et al. (2021) ran neo-Hookean FE slabs with tangential growth of the top layer only. Homogeneous thickness gives a single wavelength. Inhomogeneous thickness gives early structures "reminiscent of deep sulci", followed by shallower, more variable secondary undulations. — [Costa Campos et al. 2021, NeuroImage 231:117779, doi:10.1016/j.neuroimage.2021.117779](https://doi.org/10.1016/j.neuroimage.2021.117779)
- Holland, Budday, Goriely & Kuhl (2018) combined analytics, FE and polymer experiments. After wrinkling, ridges become thicker than valleys as a generic symmetry breaking ("gyri are universally thicker than sulci"). They also analysed hundreds of human brains showing systematic region-dependent thickness variation. Thickness heterogeneity here is a *consequence* of the instability. — [Holland et al. 2018, Phys Rev Lett 121:228002, doi:10.1103/PhysRevLett.121.228002](https://doi.org/10.1103/PhysRevLett.121.228002)
- Wang, Demirci & Holland (2020) made cortical growth depend on local mean curvature, so that growth is preferentially gyral or sulcal. Implemented in Abaqus/Explicit through a VUMAT.
  - The authors state explicitly: "As opposed to the popular pre-patterning hypothesis, our model treats inhomogeneous cortical growth as the result of folding rather than the cause".
  - Gyral–sulcal thickness differences appear only at low stiffness ratios.
  - Compared against human thickness data, the results suggest that cortical growth is higher in gyri than in sulci.
  - — [Wang, Demirci & Holland 2020, Biomech Model Mechanobiol, doi:10.1007/s10237-020-01400-w](https://doi.org/10.1007/s10237-020-01400-w)
- Demirci & Holland (2022) analysed 501 ABIDE adults. Thickness increases monotonically from concave to saddle to convex shapes, and is correlated with curvature and sulcal depth. — [Demirci & Holland 2022, Hum Brain Mapp, doi:10.1002/hbm.25776](https://doi.org/10.1002/hbm.25776)
  - Across 12 primate species the same gyral-sulcal thickness spectrum holds, and the gyral–sulcal difference grows with brain size: [Demirci, Hoffman & Holland 2023, NeuroImage, doi:10.1016/j.neuroimage.2023.120283](https://doi.org/10.1016/j.neuroimage.2023.120283)
- Garcia, Basinski & Kroenke (2025) studied a ferret template (P8–P38, aMSM).
  - Early on, cortex is relatively thick in regions destined for high curvature, whether they become gyri or sulci.
  - At maturity, gyral cortex is thicker than sulcal cortex.
  - Curvature change correlates positively with *subsequent* area expansion and negatively with *previous* expansion.
  - Proposed sequence: relative thickening, then curvature increase, then area increase, with gyral thickness rate exceeding sulcal rate in the final phase.
  - — [Garcia, Basinski & Kroenke 2025, Dev Neurosci, doi:10.1159/000544824](https://doi.org/10.1159/000544824)
- Budday, Steinmann & Kuhl (2014) used Föppl–von Kármán analytics plus finite-growth FE. Cortical thickness, stiffness and growth set the critical time, pressure and wavelength. Thinner cortex gives more and smaller folds (polymicrogyria-like); thicker cortex gives fewer folds (lissencephaly-like). Secondary folds and asymmetric patterns appear post-critically. — [Budday et al. 2014, J Mech Phys Solids, doi:10.1016/j.jmps.2014.07.010](https://doi.org/10.1016/j.jmps.2014.07.010)
  - Review of radial vs tangential expansion events: [Budday, Steinmann & Kuhl 2015, Front Cell Neurosci, doi:10.3389/fncel.2015.00257](https://doi.org/10.3389/fncel.2015.00257)
- Wang, Lefèvre, …, Rousseau (2021): thinner initial cortex gives higher fold frequency but shallower sulci. — [doi:10.1038/s41598-021-87124-y](https://doi.org/10.1038/s41598-021-87124-y)
- Hou et al. (2026): thicker cortex gives fewer, deeper folds. Regional radial growth (orthotropic regional) improves the thickness match over tangential-only regional growth, while GI and sulcal counts are similar. — [PMC13490475](https://pmc.ncbi.nlm.nih.gov/articles/PMC13490475/)
- Hou et al. (2025): radial growth had less effect on folding than tangential growth. — [PMC11718650](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718650/)

**Subcortical, subplate and white-matter heterogeneity**
- Bayly et al. (2013): subcortical growth driven by stress (it grows in response to the cortex-induced stress). The growth-rate ratio sets the wavelength. — [PMC3616769](https://pmc.ncbi.nlm.nih.gov/articles/PMC3616769)
- Balouchzadeh, Bayly & Garcia (2023) — [Brain Multiphysics 4:100065, doi:10.1016/j.brain.2023.100065 (PMC11213281)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213281/)
  - *Model.* Two layers: 1.5 mm cortex and 15 mm subcortex. Uniform isotropic cortical growth doubling the area. The subcortex is either hyperelastic or viscoelastic with stress-dependent growth ġ = a(σ−σ₀)g.
  - *Data.* Sphere, spheroid and triaxial geometries compared with four preterm infants, each scanned at four time points (27–38 wk).
  - *Effect of stress-dependent growth.*
    - Folds become more sinusoidal, with symmetric gyral/sulcal curvature distributions closer to human data. The hyperelastic version gives sharp, self-contacting sulci.
    - Mean sulcal curvature rises steadily, as in humans.
    - Fold alignment with the long axis is stronger.
  - *Caveats.* Absolute sulcal lengths stay well below human values: on axis 3, 322 mm (viscoelastic) and 500 mm (hyperelastic) vs 628 ± 80 mm in humans. Sulcal depth was not measured. Regional subcortical growth was not modelled.
- Wang, Wang & Holland (2024) combined a bilayer FE model with axonal tension in the subcortex and differential cortical growth.
  - Axonal tension acts as a perturbation that fixes fold *location*, while the system's natural stability sets the *wavelength*.
  - An axon bundle of reasonable size overpowers a ~5% thickness perturbation, which the authors call "typical in the literature".
  - Small stiffness differences between multiple tracts change the pattern.
  - — [Wang, Wang & Holland 2024, Soft Matter, doi:10.1039/d4sm00129j](https://doi.org/10.1039/d4sm00129j)
- Garcia, Wang & Kroenke (2021) proposed tension-induced fibre growth in white matter during folding. Seen only as a search result, so content is **[not re-verified]**. — [Nat Commun 2021, doi:10.1038/s41467-021-26971-9](https://www.nature.com/articles/s41467-021-26971-9)
- Solhtalab et al. (2025) modelled stress-guided "axon reorientation" with stress-induced growth. Axon bundles end up denser in gyri than in sulci, supported by DTI. — [Solhtalab et al. 2025, Nat Commun, doi:10.1038/s41467-025-56362-3](https://doi.org/10.1038/s41467-025-56362-3)
- Zarzor et al. (2023): the OSVZ starts with uniform thickness and becomes thicker under gyri and thinner under sulci *after* folding. This is presented as a mechanical consequence matching histology. — [eLife 2023](https://elifesciences.org/articles/82925)
- Rana et al. (2019) reviewed the subplate as a potential driver of folding — [Cereb Cortex 29:4697, doi:10.1093/cercor/bhz003](https://doi.org/10.1093/cercor/bhz003)
  - Ferret (Smart & McSherry 1986, as cited): the subplate deepens under future gyri and is thinner under future sulci.
  - Human MRI (Wang et al. 2010, as cited): the subplate is patchy at 27 GW; it dissolves first under sulci and last under gyral apices.
  - Sheep DTI (Geng 2009, as cited): FA minima at future sulci.
  - Duque et al. 2016 (as cited) attribute subplate thickening to cortico-cortical and ascending axons.
  - The review calls the causal link "currently circumstantial".
- Rash, Arellano, Duque & Rakic (2022) used immunohistochemistry and transcriptomics in primates.
  - Neurogenesis switches to gliogenesis before most gyri form.
  - They argue that post-neurogenic intracortical neuropil growth, glial addition and enlargement of subcortical white matter are the primary forces of gyrification ("classical model").
  - They reject the idea of neurogenic OSVZ hotspots that create convexities.
  - — [Rash et al. 2022/2023, PNAS 120:e2210967120, doi:10.1073/pnas.2210967120](https://doi.org/10.1073/pnas.2210967120)
  - Earlier, gliogenesis in the OSVZ was shown to promote enlargement and gyrification: [Rash et al. 2019, PNAS, doi:10.1073/pnas.1822169116](https://doi.org/10.1073/pnas.1822169116)

### Inferences
- **Our thickness field (thicker at crests, thinner at fundi, as early as SG 26–28).** Holland 2018 and Wang/Demirci/Holland 2020 explain this pattern as *emergent* in the late or post-critical regime. Our adjoint instead needs it *imposed as growth* to deepen sulci. Garcia 2025 provides ferret evidence that the gyral-over-sulcal thickness rate develops during fold formation. Wang/Demirci/Holland's conclusion that "growth is higher in gyri than in sulci" matches the sign of our field.
- **Our core correction (negative under fundi, positive under crests).** This pattern is closest to the following:
  - histological descriptions of subplate thinning or earlier dissolution under sulci and thickening under gyri (Rana 2019);
  - stress-dependent subcortical growth (Bayly 2013; Balouchzadeh 2023), which tends to grow where tension is and shrink where compression is;
  - Rash 2022's emphasis on white-matter enlargement.

  Hou 2026's finding that subcortical growth mainly controls sulcal depth supports the idea that a subcortical term is needed for depth.
- **Our shared depth law.** A 6-coefficient law tying both fields to sulcal depth, and shared by two sulci, is functionally similar to the curvature-coupled growth of Wang/Demirci/Holland 2020 (feedback from geometry to growth). It could be read as evidence for a mechanically or geometrically driven (consequence) rather than region-specific (prepattern) mechanism. But it is a fitted correlation, not a demonstrated feedback.

### Gaps
- No FE study was found that prescribes or infers an explicit, spatially varying subplate growth field (as opposed to stress-dependent growth or homogeneous subcortex) and reports its effect on sulcal depth. Hou 2026 treats the subcortex as homogeneous.
- No quantitative study was found that compares the size of a thickness perturbation needed to deepen a given sulcus with fetal MRI thickness maps at 26–28 GW.
- Garcia, Wang & Kroenke 2021 (tension-induced fibre growth) was not read in full.

---

## Q3. Is heterogeneity a cause (prepattern/protomap) or a consequence of mechanical folding?

### Takeaway
The field is split, and several recent papers take a hierarchical middle position. Mechanics, via differential tangential growth, causes folding and largely sets wavelength and depth. Pre-existing heterogeneities act as small perturbations that fix *where* folds form: gene-expression domains, OSVZ proliferation, axon bundles, thickness, initial geometry. Thickness and growth heterogeneity measured at maturity is at least partly a mechanical *consequence*.

### Cited Findings
**Cause / prepattern side**
- de Juan Romero et al. (2015) ran transcriptomics of ferret germinal layers. Thousands of genes are differentially expressed between prospective fold and fissure regions, including ~80% of genes mutated in human malformations. They form discrete expression domains, present in ferret and human but not mouse, that map future folds and fissures. The authors write: "Protomaps of gene expression within germinal layers may contribute to define cortical folds". — [de Juan Romero et al. 2015, EMBO J, doi:10.15252/embj.201591176](https://doi.org/10.15252/embj.201591176)
- Alexander-Bloch et al. (2020) found that local (<1 cm) genetic correlations of cortical thickness are heterogeneous and stronger at gyral crests than at sulcal fundi. They are more tied to folding in primary sensorimotor than in association cortex. The authors interpret patterned genetic influence on thickness as "a causal factor" in folding. — [Alexander-Bloch et al. 2020, PNAS, doi:10.1073/pnas.1912064117](https://doi.org/10.1073/pnas.1912064117)
- Ronan et al. (2014) argue that differential expansion links cytoarchitecture and folds. — [doi:10.1093/cercor/bht082](https://doi.org/10.1093/cercor/bht082)
- Garcia et al. (2018) argue that nonuniform growth is required, since uniform-growth models on realistic geometry do not reproduce conserved primary folds. The evidence is correlational. — [PMC5866555](https://pmc.ncbi.nlm.nih.gov/articles/PMC5866555/)
- Hou et al. (2026) treat regional growth laws as inputs, essential for realistic morphology. They link these laws to regionalised gene expression and spatial transcriptomics, without testing causality. — [PMC13490475](https://pmc.ncbi.nlm.nih.gov/articles/PMC13490475/)
- Rana et al. (2019) say subplate features may predict sulcal and gyral locations, but the evidence is "circumstantial". — [doi:10.1093/cercor/bhz003](https://doi.org/10.1093/cercor/bhz003)
- Wang, Wang & Holland (2024): axonal tension, a connectivity-based prepattern, determines fold placement. — [doi:10.1039/d4sm00129j](https://doi.org/10.1039/d4sm00129j)
- Chavoshnejad et al. (2023): a hierarchy in which growth induces folding and heterogeneous axon density locates the gyri. — [doi:10.1038/s41598-023-40086-9](https://doi.org/10.1038/s41598-023-40086-9)

**Consequence / mechanics-first side**
- Tallinen et al. (2016): fold size, shape, placement and orientation result from a mechanical instability modulated by fetal geometry. — [HAL](https://hal.archives-ouvertes.fr/hal-01342578)
- Wang, Demirci & Holland (2020): inhomogeneous growth is "the result of folding rather than the cause", explicitly opposed to pre-patterning. — [doi:10.1007/s10237-020-01400-w](https://doi.org/10.1007/s10237-020-01400-w)
- Holland et al. (2018): gyral/sulcal thickness asymmetry is a generic consequence of wrinkling. — [doi:10.1103/PhysRevLett.121.228002](https://doi.org/10.1103/PhysRevLett.121.228002)
- Zarzor et al. (2023): OSVZ thickness variations and fan-like radial fibres are more likely results than causes of folding. Inhomogeneous proliferation only sets where the first folds appear. — [eLife](https://elifesciences.org/articles/82925)
- Hou et al. (2025): initial curvature, stiffness and thickness ratio decide the bulking sites; growth mostly sets timing and extent. — [PMC11718650](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718650/)
- Rash et al. (2022) argue against neurogenic prepattern hotspots, on timing grounds (neurogenesis ends before gyrification). They favour post-neurogenic neuropil and white-matter growth. This is non-mechanical in method but compatible with growth heterogeneity arising late. — [doi:10.1073/pnas.2210967120](https://doi.org/10.1073/pnas.2210967120)

**Middle positions and reviews**
- Garcia, Basinski & Kroenke (2025): growth, thickness and folding are temporally interleaved. Thickening precedes curvature, and area expansion follows it, which is neither pure cause nor pure consequence. — [doi:10.1159/000544824](https://doi.org/10.1159/000544824)
- Choi et al. (2025): genetics is the proximal cause, expressed mechanically through geometry and differential growth. — [doi:10.7554/eLife.107141](https://doi.org/10.7554/eLife.107141)
- Reviews:
  - Llinares-Benadero & Borrell 2019 (genetic, cellular and mechanical determinants): [Nat Rev Neurosci, doi:10.1038/s41583-018-0112-2](https://doi.org/10.1038/s41583-018-0112-2)
  - Barresi et al. 2024 (OSVZ/bRG indispensable; relative weight of mechanics unresolved): [Cereb Cortex, doi:10.1093/cercor/bhae055](https://doi.org/10.1093/cercor/bhae055)
  - Budday et al. 2015: [Front Cell Neurosci, doi:10.3389/fncel.2015.00257](https://doi.org/10.3389/fncel.2015.00257)

### Inferences
- A fitted heterogeneous growth field, like ours, cannot by itself distinguish cause from consequence. The same field could be a prepattern or a mechanical feedback. Our observation that one *depth-dependent* law fits both sulci leans toward a generic, geometry-coupled mechanism of the Wang/Demirci/Holland type rather than sulcus-specific prepatterns.
- A discriminating test would compare two scenarios. In the first, the field is frozen at its SG 26 value (prepattern). In the second, it is recomputed from current depth (feedback). The question is which one better predicts SG 28 and beyond.

### Gaps
- No modelling study was found that formally tests the prepattern and feedback hypotheses against the same longitudinal fetal data.

---

## Q4. Quantitative comparisons of simulated vs real (MRI) folds, with vs without heterogeneity

### Takeaway
Only Hou et al. 2025 (Soft Matter) and Hou et al. 2026 (Nat Commun) compare uniform and regional growth quantitatively against MRI. Regional growth improves regional GI, thickness, sulcal counts and landmark placement. Sulcal depth remains the weakest metric in places, and mean curvature is poorly predicted in the slab version. Other quantitative comparisons (Balouchzadeh 2023; Solhtalab 2025 HBM) use uniform cortical growth.

### Cited Findings
- **Hou 2026.**
  - Of the four scenarios, the two regional ones (tangential regional and orthotropic regional) best match real regional GI and thickness, with similar sulcal counts.
  - Shape-index similarity exceeds 0.8.
  - Six landmarks are reproduced.
  - Sulcal depth deviates in peri-Sylvian, interhemispheric and isthmus-cingulate regions.
  - Uniform tangential growth under-develops the central-sulcus region.
  - — [PMC13490475](https://pmc.ncbi.nlm.nih.gov/articles/PMC13490475/)
- **Hou 2025.**
  - Regional growth gives lower MAPE than isotropic or constant-tangential growth.
  - Sulcal depth and GI correlate well with real data in most regions, but mean curvature has R² ≈ 0.
  - Single-region models exaggerate folding in fast-growing parcels.
  - The multi-region model misses the central sulcus.
  - — [PMC11718650](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718650/)
- **Balouchzadeh 2023.** Stress-dependent subcortical growth improves the curvature-distribution and sulcal-curvature trajectories against 4 preterm infants. Absolute sulcal lengths remain too short. — [PMC11213281](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213281/)
- **Solhtalab et al. 2025 (HBM).** An individual-specific forward model reproduces the distribution of sulcal pits at a later scan. — [doi:10.1002/hbm.70332](https://doi.org/10.1002/hbm.70332)
- **Chavoshnejad 2023.** Statistical correlation between the locations of initial undulations or dense fibres and the final gyri and sulci, validated qualitatively against imaging. — [doi:10.1038/s41598-023-40086-9](https://doi.org/10.1038/s41598-023-40086-9)

### Inferences
- No published comparison was found that reports sulcal *depth* error at specific sulci, with vs without a thickness or subcortical heterogeneity field, at fetal ages around 26–28 GW. Our central/calcarine adjoint results (depth unattainable with tangential growth alone; recovered with thickness and core fields) would fill that gap.
- The dHCP fetal data used by Hou 2026 (22–40 GW) overlaps our atlas window, which allows a direct comparison of their parcel growth ratios with our fitted fields.

### Gaps
- Hou 2026's per-sulcus numbers for depth error, and any calcarine-specific results, were not extracted. The last ~34k characters of the PMC page were not read.
- Alenyà et al. 2022 (Brain Multiphysics) and Wang et al. 2019 (EMBC, [doi:10.1109/EMBC.2019.8856670](https://doi.org/10.1109/EMBC.2019.8856670)) were not read in detail.
