# Cortical thickness contrast between gyral crests and sulcal fundi: data and explanatory models

Scope: (a) measured crest/fundus thickness contrast from fetus to adult; (b) models that explain it, and whether it is a cause or a consequence of folding; (c) prepattern models; (d) pitfalls when measuring thickness on fetal MRI. Context: an adjoint fit of Tallinen 2016 to the dHCP fetal atlas (SG 26–28) needs a depth-dependent thickening that is stronger than the pial–white thickness contrast measured on the atlas.

Note on the reference given in the brief: PMC10237175 is **Holland, Budday, Li, Shen, Goriely & Kuhl (2020)**, "Folding drives cortical thickness variations", Eur. Phys. J. Spec. Top. 229:2757–2778, doi:10.1140/epjst/e2020-000001-6. It is not by "Wang, Zhang et al.". Most PMC/PNAS/J Neurosci full texts could not be fetched (403 or a bot check), so some numbers come from secondary summaries. These are flagged.

---

## Q1. Empirical crest vs fundus thickness: magnitude, ratio, age of appearance

### Takeaway
Gyri are thicker than sulci at every postnatal age measured. In adults the ratio is about 1.12–1.16 on MRI, and the classical histology value is about 1.6–2.0. In dHCP neonates the ratio grows from about 1.04 at 29 PMW, where the difference is not significant, to about 1.24 at 43 PMW. I found **no direct MRI measurement of crest/fundus thickness at SG 26–28 in utero**. The available fetal evidence (25–31 GW) is indirect: thickness grows faster under future or current gyri, and the microstructure of crests and fundi differs before the sulcal pits form.

### Cited Findings
**Classical / adult**
- Bok (1929) set out the equi-volume principle: cortical segments keep their volume, while layer and cortical thickness change to compensate for folding. Crests are thick with thin deep layers, and fundi are thin with thick deep layers. Citation: Z. Gesamte Neurol. Psychiatr. 121:682–750, doi:10.1007/BF02864437. The DOI was taken from a reference list and not resolved directly, so it is *partially verified*. — [BigBrainWarp references](https://bigbrainwarp.readthedocs.io/en/latest/pages/references.html); an English translation with commentary is reported at [par.nsf.gov/servlets/purl/10402033](https://par.nsf.gov/servlets/purl/10402033) (not opened).
- Waehnert et al. (2014) turned Bok's principle into an equi-volume laminar model. It fits myelinated bands in high-resolution MRI better than equidistant or Laplace laminae. NeuroImage 93:210–220, doi:10.1016/j.neuroimage.2013.03.078 (PubMed record dated 2013 epub). — [PubMed 23603284](https://pubmed.ncbi.nlm.nih.gov/23603284/)
- Fischl & Dale (2000) introduced automated surface-based thickness measurement, with inter-subject SD < 0.5 mm. PNAS 97:11050–11055, doi:10.1073/pnas.200033797. Its abstract does not give crest/fundus numbers, and I could not open the full text (403). The often-cited claim that "crowns are thicker than fundi" in this paper is **unverified here**. — [PubMed 10984517](https://pubmed.ncbi.nlm.nih.gov/10984517/)
- Tallinen et al. (2014) report that in their 2D simulation the deformed gray-matter thickness "at the gyrus … is nearly twice that at the base of the sulcus; the same pattern as seen in all real brains". Holland 2020 cites Tallinen 2014 for histological gyral/sulcal ratios of 1.6–2.0. PNAS 111:12667, doi:10.1073/pnas.1406015111. — [arXiv 1503.03853 (text extracted locally)](https://arxiv.org/pdf/1503.03853); [Holland 2020 PMC10237175](https://pmc.ncbi.nlm.nih.gov/articles/PMC10237175/)
- Adult MRI values from Holland et al. (2020):
  - 9 adults: gyral 2.74 mm, sulcal 2.37 mm, ratio 1.157.
  - 564 brains aged 7–64 y: 2.87 mm vs 2.47 mm, ratio 1.162.
  - Gyri were thicker in all datasets (p ≪ 1e-10).
  — [PMC10237175](https://pmc.ncbi.nlm.nih.gov/articles/PMC10237175/)
- Zhang et al. (2016), HCP adults (n = 10, Destrieux labels): gyral 2.91 ± 0.89 mm, sulcal 2.59 ± 0.81 mm, ratio about 1.12. The text says "roughly 1.2×" and misprints the sulcal value as 2.91, which is inconsistent with its Table 1. Sci Rep 6:37272, doi:10.1038/srep37272. — [PMC5112531](https://pmc.ncbi.nlm.nih.gov/articles/PMC5112531/)
- Holland, Budday, Goriely & Kuhl (2018) analysed "hundreds of human brains". They report a region-dependent thickness pattern and conclude that gyri are universally thicker than sulci. PRL 121:228002, doi:10.1103/PhysRevLett.121.228002. Only the abstract was read. — [ORA Oxford record](https://ora.ox.ac.uk/objects/uuid:45bf9744-4653-4d05-b730-1c312fa11d41)

**Infant / child**
- Li, Lin, Gilmore & Shen (2015), 73 infants with 219 longitudinal scans at 0, 1 and 2 y: "gyri were thicker than sulci". The anterior bank of the central sulcus was thicker than the posterior bank, and the spatial pattern of thick and thin regions was largely present at birth. J Neurosci 35:9150–9162, doi:10.1523/JNEUROSCI.4107-14.2015. — [PubMed 26085637](https://pubmed.ncbi.nlm.nih.gov/26085637/)
- Holland et al. (2020) reanalysed the same infant cohort:
  - Within one month of birth: gyral 2.03 mm, sulcal 1.78 mm (ratio about 1.14).
  - Year 1: 2.92 / 2.52 mm. Year 2: 2.94 / 2.50 mm.
  - Regional ratios ranged from about 1.04 (parietal, year 0) to about 1.26 (temporal, years 1–2).
  — [PMC10237175](https://pmc.ncbi.nlm.nih.gov/articles/PMC10237175/)
- Li et al. (2015, HBM): cortical fibre density is higher under gyri than sulci at 0, 1 and 2 y, and correlates positively with thickness at gyral "3-hinge" sites. doi:10.1002/hbm.23003. — [PubMed 26417847](https://pubmed.ncbi.nlm.nih.gov/26417847/)

**Neonatal / preterm (dHCP), closest to SG 26–30**
- Demirci & Holland (2024), dHCP, 29–43 PMW, convex vs concave vertices:
  - No significant difference at 29 PMW.
  - Convex points significantly thicker from 30 PMW onward. Cohen's d rises from 0.15 at 29 PMW to 0.91 at 43 PMW.
  - The thickness ratio rises non-linearly, from **1.04 at 29 PMW to 1.24 at 43 PMW**. Convex thickness goes from about 1.18 to 1.29 mm, concave from about 1.00 to 1.06 mm.
  - Mean thickness is about flat (1.15 → 1.12 → 1.19 mm), so the contrast is a redistribution.
  - Sulcal (concave) points thin until about 35 PMW, then both classes thicken, gyri more so. The summary I obtained is slightly inconsistent here: the concave range 1.00 → 1.06 versus "thinning until 35 PMW" should be checked in the paper.
  - The authors note that buckling-only simulations do not reproduce the non-linear rise of the ratio.
  - Cereb Cortex 34(2):bhad462, doi:10.1093/cercor/bhad462.
  — [OUP](https://academic.oup.com/cercor/article/34/2/bhad462/7588692)
- Rajagopalan et al. (2017), 27 preterm neonates without brain injury, 30–46 PMW. In primary gyri, surface-area and thickness growth rates were below average along sulcal pits and above average on gyral banks and crests, in both GM and WM. Hum Brain Mapp 38:4322, doi:10.1002/hbm.23618. — [PubMed 28608653](https://pubmed.ncbi.nlm.nih.gov/28608653/)

**Fetal (in utero / atlas / animal)**
- Zhang et al. (2016) used the longitudinal fetal MRI atlas from brain-development.org, 25–31 GW. Per-vertex thickness growth rate (slope of thickness vs age) averaged 0.35 ± 0.26 on gyral vertices and 0.10 ± 0.26 on sulcal vertices (p ≈ 4e-98).
  - Units are not stated in the summary I read, so they are *to check* (plausibly mm/week).
  - Only 51% of vertices met R² ≥ 0.6.
  - This is the only fetal evidence I found that thickness *grows faster* at crests during 25–31 GW. It implies a crest/fundus contrast that develops over that window.
  — [PMC5112531](https://pmc.ncbi.nlm.nih.gov/articles/PMC5112531/)
- Wilson et al. (2024 preprint, dHCP in utero diffusion, 2 mm acquisition). Anisotropic tissue fraction is lower in future fundi and higher at crests. It "decreases immediately prior to the formation of sulcal pits". Within-patch correlations appear at 25 GW (Sylvian fissure), 27 GW (central sulcus) and about 35 GW (most primary sulci). CP thickness is not measured, and fetuses ≤ 24 GW were excluded. bioRxiv doi:10.1101/2023.10.16.562524, not peer reviewed (eLife reviewed preprint). — [PMC11230247](https://pmc.ncbi.nlm.nih.gov/articles/PMC11230247/); [eLife RP 100895](https://elifesciences.org/reviewed-preprints/100895)
- Wang, Studholme, …, Kroenke (2017), rhesus macaque in utero, G85–G135. Loss of cortical diffusion anisotropy, which tracks dendritic maturation in histology, is coupled regionally and temporally to folding but not to surface expansion. The authors read this as structural differentiation within the cortex being coupled to gyral/sulcal formation. J Neurosci 37:1971–1983, doi:10.1523/JNEUROSCI.3157-16.2017. — [PubMed 28069920](https://pubmed.ncbi.nlm.nih.gov/28069920/)
- Xu et al. (2022), Chinese fetal atlas, 23–38 GW (n = 90). Mean cortical thickness is biphasic: it rises, then falls, with an inflection around GA 31 w. No crest/fundus split is reported in the abstract. J Neurosci, doi:10.1523/JNEUROSCI.1285-22.2022. — [PubMed 36323525](https://pubmed.ncbi.nlm.nih.gov/36323525/)
- dHCP fetal surface atlas (21–36 GW, 242 in-utero scans) provides pial, white and mid-thickness surfaces plus thickness, curvature and sulcal-depth maps. DOI 10.12751/g-node.qj5hs7; formal citation "to follow"; contact S. Karolis. — [GIN README](https://gin.g-node.org/doi/dhcp_fetal_brain_surface_atlas/src/master/README.md)

### Inferences
- Taken together, Demirci & Holland (non-significant, ratio about 1.04 at 29 PMW) and Zhang 2016 (faster gyral thickening at 25–31 GW) suggest that at SG 26–28 the *measurable* MRI contrast is small, roughly 1.0–1.1. Most of the contrast builds after 30 weeks, alongside deepening and secondary folding.
- Adult MRI ratios (about 1.12–1.16) are clearly below the histological ratio of about 1.6–2.0. Surface-based MRI therefore seems to underestimate the contrast systematically, even in adults. See Q4.
- Diffusion and macaque microstructure data show that crests and fundi differ before or at the time of pit formation. This leaves room for a *prepattern*, in microstructure or growth, that is not yet visible as a thickness difference.

### Gaps
- No in-utero or histological measurement of the CP thickness ratio between crest and fundus at exactly SG 26–28 was found. Fetal histology of gyral vs sulcal CP thickness (Kostović-type material) was not located in this search.
- Fischl & Dale 2000 crest/fundus numbers: not verified.
- The Zhang 2016 growth-rate units and the precise PRL 2018 numbers were not read from the papers.
- The values printed in the dHCP fetal atlas thickness maps were not inspected.

---

## Q2. Models: is the contrast a cause or a consequence of folding? What ratio do they predict?

### Takeaway
Bilayer growth models with a *uniform* initial cortex (Tallinen 2014; Holland 2018/2020) produce thick crests and thin fundi as a nonlinear, post-buckling consequence. The ratio is about 1 at the first (wrinkling) bifurcation and becomes large only once folds are deep: about 1.4 at the second bifurcation, about 1.8 at self-contact, and about 2 in Tallinen's 2D sulci. The authors themselves do not claim that mechanics is the sole cause. Genetic and cellular factors are acknowledged, and the observed non-linear rise of the ratio is not reproduced by buckling alone.

### Cited Findings
- Tallinen et al. (2014) model a thin stiff growing gray matter on a soft core, with modulus ratio close to 1 for brain-like cusped sulci. The deformed thickness at the gyrus is "nearly twice" that at the sulcus base (2D, g up to 2.25). Their Fig. 2 defines Ts and Tg and compares sulcus width and depth with porcupine, cat and human. — [arXiv 1503.03853](https://arxiv.org/pdf/1503.03853), doi:10.1073/pnas.1406015111
- Holland et al. (2020), neo-Hookean morphoelastic bilayer studied analytically, by FE and with pre-stretched silicone experiments (stiffness contrast 4.7):
  - FE gyral/sulcal ratio: **1.000 at the first bifurcation, 1.430 at the second, 1.790 at self-contact**, with final values of about 2.84 vs 1.58 mm.
  - Sulcal thickness falls below 65% of its initial value after the second bifurcation.
  - Polymer experiments give ratios of 1.274–1.622, increasing with layer thickness.
  - Thickness asymmetry emerges "even in the complete absence of regional information".
  - The authors conclude that genetic, geometric and physical factors are coupled.
  — [PMC10237175](https://pmc.ncbi.nlm.nih.gov/articles/PMC10237175/)
- Holland et al. (2018, PRL): after buckling, the two sides of the film experience different forces. This symmetry breaking makes ridges thicker than valleys and is presented as a generic feature of wrinkled film/substrate systems. — [ORA](https://ora.ox.ac.uk/objects/uuid:45bf9744-4653-4d05-b730-1c312fa11d41), doi:10.1103/PhysRevLett.121.228002
- Demirci & Holland (2024) state that buckling-based simulations do not reproduce the nonlinear increase in the ratio seen in dHCP data. They suggest that axonal growth and subplate changes contribute. — [Cereb Cortex bhad462](https://academic.oup.com/cercor/article/34/2/bhad462/7588692)
- Bayly et al. (2013) use a stress-dependent growth model that explains gyral wavelength. It treats the cortex as a plate of *uniform* thickness h and reports no crest/fundus thickness difference; wavelength scales with h. Phys Biol 10:016005, doi:10.1088/1478-3975/10/1/016005. — [PMC3616769](https://pmc.ncbi.nlm.nih.gov/articles/PMC3616769/)
- Budday, Raybaud & Kuhl (2014) use a growing cortex on a stretch-driven growing core, calibrated on very-preterm MRI. Changes in cortical growth and thickness reproduce lissencephaly and polymicrogyria. This is a *global* thickness effect, not a crest/fundus prepattern. Sci Rep, PMC4090617. — [PMC4090617](https://pmc.ncbi.nlm.nih.gov/articles/PMC4090617)
- Wang, Lefèvre, Bohi, Al Harrach, Dinomais & Rousseau (2021) vary initial (uniform) cortical thickness in a Tallinen-type model. Thinner cortex gives higher fold frequency but **shallower sulci**. Sci Rep 11:7686, doi:10.1038/s41598-021-87124-y. — [PubMed 33833302](https://pubmed.ncbi.nlm.nih.gov/33833302/)

### Inferences
- In Tallinen/Holland-type mechanics the deformation-induced contrast scales with fold amplitude: about 1 at onset and about 1.4 once folds are deep. At SG 26–28, when primary sulci are still shallow, a uniform-growth model would generate little contrast by itself.
- If the model's mechanics contributes little at this stage, the depth-dependent thickening fitted by the adjoint is mostly an *imposed* growth contrast. This is consistent with Zhang 2016's growth-rate data and with the "cellular plus mechanics" view of Demirci & Holland 2024.
- Measured pial–white thickness is the *combined* result of differential growth and mechanical thinning or thickening. A fitted growth contrast that *exceeds* the measured contrast therefore points to one or more of three explanations:
  1. The measurement underestimates the true contrast (Q4).
  2. The model's mechanics thin crests or thicken fundi at this stage, partly cancelling imposed growth. For example, Tallinen's growth law is tangential, and volumetric thickening is redistributed.
  3. The growth contrast also stands in for missing physics, such as subplate or axonal effects, or a stiffness heterogeneity that limits depth.
  These are hypotheses, not sourced findings.

### Gaps
- No paper found that quantifies the crest/fundus ratio inside Tallinen 2016's 3D brain simulation (Nat Phys, doi:10.1038/nphys3632) at fetal stages.
- PRL 2018 numerical predictions (ratio vs stiffness ratio) were not extracted.

---

## Q3. Work imposing a thickness or growth prepattern that controls fold location or depth

### Takeaway
Several models show that a regional prepattern of faster growth or thicker cortex fixes *where* gyri form. The prepattern sets a gyrus, not a sulcus, at the fast or thick site. A local stiffness increase instead seeds a sulcus. Effects on *depth* are less quantified. Uniform-thickness studies show that thicker cortex gives deeper, wider folds.

### Cited Findings
- Zhang et al. (2016) model a 3D bilayer plate (2 mm cortex) in which a central region grows 1.2× faster; the factor was chosen from the adult gyral/sulcal thickness ratio.
  - Over 50 runs with homogeneous growth, the centre became a gyrus 14 times, a sulcus 19 times and a bank 17 times, i.e. random.
  - Over 50 runs with heterogeneous growth, it became a **gyrus 47/50 times (94%), a sulcus 0 times** and a bank 3 times.
  - Increasing the stiffness of that region (2×, 4×, 8×) instead produced sulci.
  - Fold size and depth were explicitly not studied.
  - A corrigendum exists (doi:10.1038/srep42269).
  — [PMC5112531](https://pmc.ncbi.nlm.nih.gov/articles/PMC5112531/); [corrigendum record](https://pure.nwpu.edu.cn/zh/publications/corrigendum-mechanism-of-consistent-gyrus-formation-an-experiment)
- Toro & Burnod (2005) built a morphogenetic model in which geometric, mechanical and growth asymmetries reproduce gradients in the degree of convolution and the dependence of convolution on cortical thickness. Cereb Cortex 15:1900–1913, doi:10.1093/cercor/bhi068. — [PubMed 15758198](https://pubmed.ncbi.nlm.nih.gov/15758198/)
- Hou et al. (2025) fitted regional growth laws by ML symbolic regression on surface expansion *and cortical thickness*, using 735 subjects (> 1000 scans, 29 PMW–2 y), and fed them into folding simulations. They argue that regional growth variation shapes folding patterns. Soft Matter, doi:10.1039/d4sm01194e. Only the abstract was read. — [PubMed 39791229](https://pubmed.ncbi.nlm.nih.gov/39791229/)
- Wang et al. (2021, Rousseau group) varied uniform initial thickness: thinner cortex gives more frequent and shallower folds. — [PubMed 33833302](https://pubmed.ncbi.nlm.nih.gov/33833302/)
- A web summary attributes to Wang et al. (2021) the finding that "simulations with preferential growth in gyri best reproduce the brain's thickness patterns". **I could not identify which paper this is, so it is unverified.** — [search result snippet, no primary source]

### Inferences
- A "thickening increasing with depth" law (thicker at crests) is mechanically the same kind of prepattern as Zhang 2016's faster-growing region. Such laws pin gyri, and by complement sulci, in place. Imposing them is consistent with the literature as long as it is interpreted as a growth prepattern (cause) rather than only a mechanical consequence.
- Because a stiffer region nucleates a sulcus (Zhang 2016), stiffness and growth fields can trade off in an inverse problem. An identifiability check (c, k vs growth) seems warranted.

### Gaps
- No paper found that quantifies *sulcal depth* as a function of the amplitude of an imposed crest/fundus thickness prepattern. This is the exact quantity relevant to the adjoint fit.

---

## Q4. Measurement pitfalls: does fetal MRI, atlas averaging or partial volume reduce the measured contrast?

### Takeaway
Fetal cortical plate thickness (about 1–1.5 mm in neonates, per Demirci & Holland) is close to the reconstructed voxel size of about 0.5–0.8 mm. Partial volume, contrast inversion and narrow sulci bias surface placement. I found no study that directly quantifies the bias in the *crest/fundus contrast*. Several mechanisms plausibly compress it:
- partial volume in tight sulci;
- smoothing of thickness maps;
- averaging across subjects with imperfectly aligned folds, as in atlas construction.
Even adult MRI ratios (about 1.15) fall below histology (about 1.6–2.0).

### Cited Findings
- The fetal CP thickness approaches the best achievable fetal MRI resolution, so partial volume is a major obstacle. Immature WM appears brighter than CP, and partial voluming at the CSF boundary can mimic WM, which causes holes and topological errors in thin or highly folded cortex. — [Dou et al., arXiv 2004.12847](https://arxiv.org/pdf/2004.12847)
- Smaller fetal brains have narrower cortical ribbons and sulci, which makes surfaces more sensitive to partial volume and ambiguous boundaries. Self-intersections are used as an error metric. — [MICCAI 2024 paper 3476](https://papers.miccai.org/miccai-2024/paper/3476_paper.pdf)
- dHCP in-utero diffusion is acquired at 2 mm and T2 is reconstructed at 0.5 mm for surface extraction. Partial volume increases in smaller fetal brains. — [Wilson et al. PMC11230247](https://pmc.ncbi.nlm.nih.gov/articles/PMC11230247/)
- The dHCP fetal surface atlas averages co-registered surfaces with adaptive Gaussian kernels in age, followed by five MSM iterations driven by sulcal depth and then curvature. Kernel widths are not given in the README. — [GIN README](https://gin.g-node.org/doi/dhcp_fetal_brain_surface_atlas/src/master/README.md); method adapted from Bozek et al. (2018), NeuroImage 179:11–29 (cited in the README, DOI not checked).
- The dHCP neonatal pipeline paper (Makropoulos et al. 2018) covers 465 subjects aged 28–45 PMW. Surfaces were topologically correct and matched manual boundaries in 85% of cases, with significant errors in about 2%. NeuroImage 173:88–112, doi:10.1016/j.neuroimage.2018.01.054. Its partial-volume handling was not read. — [Nottingham repository](https://nottingham-repository.worktribe.com/output/1161804)
- Multisite fetal MRI: field strength, in-plane resolution and slice thickness measurably affect cortical thickness, curvature and sulcal depth. Xu et al. (2024), Eur Radiol, doi:10.1007/s00330-024-11084-w. Abstract only. — [PubMed 39299951](https://pubmed.ncbi.nlm.nih.gov/39299951/)
- Some preterm studies regress curvature out of thickness, which removes the crest/fundus contrast by construction. — [ScienceDirect S1053811921007618](https://www.sciencedirect.com/science/article/pii/S1053811921007618) (snippet only)

### Inferences
- An atlas built as an average over 242 subjects has smoother folds than individuals. Its "thickness" is either an average of per-subject thickness maps (contrast blurred by misregistration) or the pial–white distance on averaged surfaces (contrast flattened by surface averaging). Either way, the atlas contrast should be read as a **lower bound** on the individual contrast. This makes a fitted contrast above the atlas value plausible rather than suspect.
- In tight early sulci, CSF partial volume tends to push the pial surface inward or merge banks. This shortens measured fundus thickness, which raises the contrast. In shallow, wide early folds the opposite smoothing effect probably dominates. The sign of the net bias at SG 26–28 is not established in the literature found.
- The nearest-point or symmetric distance used in FreeSurfer/dHCP-style thickness differs from Bok-type, volume-preserving radial thickness, especially on curved crests and fundi. The comparison metric between the model and the atlas should be the same on both sides.

### Gaps
- No study quantifies crest/fundus thickness contrast versus resolution, super-resolution method or atlas averaging in fetal MRI. A local synthetic test, such as degrading the model output to the atlas pipeline, would answer this for this project.
- Bozek 2018 and Karolis (fetal atlas) papers were not opened. The kernel sigma used in the atlas is unknown.
