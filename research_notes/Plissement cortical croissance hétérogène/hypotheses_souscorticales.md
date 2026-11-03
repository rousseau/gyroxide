# Subcortical hypotheses for cortical folding: subplate, oSVZ / basal radial glia, axonal tension, fiber-growth mechanics

Context for the reader: our adjoint fit of a Tallinen-2016-type model to the dHCP fetal atlas gives an isotropic **core** growth correction δ < 0 under sulcal fundi and δ > 0 under gyral crests (|δ| up to ~0.3), in the same places as a cortical thickness contrast. The question is which subcortical mechanism this δ field could stand for, and whether the literature supports it.

Verification: every reference below was checked against PubMed metadata (title, authors, journal, DOI) in this session, except where marked **[unverified]**. The claims come from the abstracts that PubMed returned. I did not read full texts, so finer quantitative details (for example exact proliferation ratios) are listed under Gaps.

---

## 1. Axonal tension: Van Essen 1997 → Xu 2010 → Bayly/Garcia/Kroenke reviews

### Takeaway
Axons are under tension in developing and adult white matter. In ferret, however, that tension does not run across developing gyri, so it is not considered the primary folding driver. The consensus in mechanics is that differential tangential growth of the cortex drives folding, and that subcortical factors (tension, spatio-temporal growth patterns) mainly bias **where** folds form. Van Essen kept a revised tension model ("DES+") in 2020, and Holland's group showed in 2024 that axon tension is an effective fold-placement perturbation.

### Cited Findings
- Original TBM hypothesis: tension along axons in white matter "can explain how and why the cortex folds in a characteristic species-specific pattern"; tension also keeps wiring compact. — [Van Essen 1997, Nature 385:313](https://doi.org/10.1038/385313a0)
- Adult mouse: radial cuts through the white matter open, but cuts confined to cortex do not. White matter is in circumferential tension of ~100–1000 Pa and cortex is in compression. FE models show this can come from **more growth in gray than white matter**. — [Xu, Bayly, Taber 2009, Biomech Model Mechanobiol 8:253](https://doi.org/10.1007/s10237-008-0131-4)
- Ferret (key paper): there is "considerable tension" along axons circumferentially in subcortical tracts and **radially inside developing gyri**, but "axonal tension is not directed across developing gyri, suggesting that axon tension does not drive folding". FE models show that "differential cortical growth accompanied by **remodeling of the subplate** leads to outward folds and stress fields consistent with the microdissection". Local perturbations, such as temporal differences in when cortical growth starts, can fix fold location. — [Xu et al. 2010, J Biomech Eng 132:071013](https://doi.org/10.1115/1.4001683)
- Review: the leading hypotheses are circumferential cortical expansion, radial growth, and axonal tension. Mathematical models are used to test which ones obey physical law with plausible parameters. — [Bayly, Taber, Kroenke 2013, J Mech Behav Biomed Mater 29:568](https://doi.org/10.1016/j.jmbbm.2013.02.018)
- Phil Trans B review: the authors "highlight evidence of a combined mechanism, in which **spatio-temporal patterns bias the locations of primary folds**, but tangential growth of the cortical plate induces mechanical instability to propagate primary and higher-order folds". The three candidate mechanisms named are (i) tangential cortical growth, (ii) spatio-temporal patterns of neuron birth/migration, and (iii) axon tension. — [Garcia, Kroenke, Bayly 2018, Phil Trans R Soc B 373:20170321](https://doi.org/10.1098/rstb.2017.0321)
- Consensus framework: an expanding outer zone is coupled to an inner zone (future white matter) that grows more slowly, "**perhaps in response to stress induced by expansion from the outer layer**". The review notes that the cellular origins of these forces are not known. — [Kroenke & Bayly 2018, J Neurosci 38:767](https://doi.org/10.1523/JNEUROSCI.1105-17.2017)
- Van Essen's 2020 "DES+" revision of TBM adds tangential tension and "sulcal zipping" forces along the outer cortical margin, plus tension in the white-matter core, which compete against radially biased tension in cortical gray matter. He proposes experiments to test it. — [Van Essen 2020, PNAS 117:32868](https://doi.org/10.1073/pnas.2016830117)
- Bilayer FE model with axon bundles in the subcortex plus differential cortical growth: axon tension acts as a perturbation that **sets fold location**, while the bilayer stability sets the wavelength. "An axon bundle of reasonable size will overpower up to a 5% thickness perturbation". A small difference in axon stiffness between tracts changes the pattern. — [Wang X, Wang S, Holland 2024, Soft Matter 20:3053](https://doi.org/10.1039/d4sm00129j)
- Residual stress measured in adult mouse and minipig: cortex is predominantly compressive, while white matter has "strongly anisotropic tensile stresses" aligned with axon orientation (dMRI). This fits "constrained cortical expansion and tension-induced axonal growth". — [Balouchzadeh, Kroenke, Garcia, Bayly 2025, J Biomech Eng 147(9)](https://doi.org/10.1115/1.4068879)
- 2025 opinion piece: constrained cortical expansion generates folds through instability, and the stress that folding induces in turn mediates axonal growth. Mechanics is proposed as the link between morphology and fascicle organization. — [Garcia, Kroenke, Bayly 2025, Trends Neurosci 48:395](https://doi.org/10.1016/j.tins.2025.04.001)

### Inferences
- Xu 2010 measured radial tension *inside* gyri. Mechanically, that is the stress state that would make a gyral core grow more if growth is stress-driven (tension-induced fiber elongation). So a δ>0 under crests is *consistent* with axon/tissue tension, but as a **response** to folding rather than a cause.
- In the Xu 2010 FE model, "remodeling of the subplate" plays the role of a softer or adaptive subcortical layer. That is the closest classical mechanical counterpart to a spatially varying core growth or stiffness under folds.

### Gaps
- I found no quantitative in-vivo measurement of gyrus-vs-sulcus differences in subcortical *stiffness* during fetal folding. Residual-stress data are adult (mouse, minipig) or neonatal ferret.
- I did not read the Xu 2010 full text, so the exact form of "subplate remodeling" in their model (growth or softening) is not confirmed.

---

## 2. Fiber-growth mechanical models (Zhang, Razavi, Chavoshnejad, Garcia, Holland)

### Takeaway
Two families of models link subcortical fibers to fold location. In the **causal** family (Zhang/Razavi/Chavoshnejad 2016–2023), denser or stiffer axonal fibers or a faster-growing local region pre-pattern gyri. In the **consequence** family (Garcia 2021 Nat Commun; Solhtalab/Razavi 2025 Nat Commun), folding stress drives fiber elongation or reorientation, which produces denser fibers under gyri. Both reproduce the empirical finding that fiber density and terminations are higher under gyri. This is the same sign as our δ (more core growth under crests).

### Cited Findings
- Empirical basis: dMRI in human, chimpanzee and macaque shows "axonal fibers connected to gyri are significantly denser than those connected to sulci". The authors suggest "axonal pushing" as a mechanism. — [Nie et al. 2012, Cereb Cortex 22:2831](https://doi.org/10.1093/cercor/bhr361)
- Human adult and fetal data: "thicker cortices are consistently found in gyral regions and gyral cortices have higher growth rates". In FE models, gyri form where the cortical plate grows faster, and global differential growth alone gives only a random pattern. — [Zhang et al. 2016, Sci Rep 6:37272](https://doi.org/10.1038/srep37272)
- 3-hinge gyri have much higher DTI fiber density than 2-hinge gyri. In FE models, regions with "higher concentrations of growing axonal fibers tend to form 3-hinge gyri". — [Ge et al. 2018, Cereb Cortex 28:3550 (DOI verified; pages unverified)](https://doi.org/10.1093/cercor/bhx227)
- Theory: radial structures (RGC scaffold controlled by Trnp1, plus maturing axons) scaffold convolution. Regionally heterogeneous RGC development and distribution regulate gyri versus sulci, and axons regulate convex (gyral) patterns. — [Razavi et al. 2017, Front Comput Neurosci 11:76](https://doi.org/10.3389/fncom.2017.00076)
- Interplay of heterogeneous cortical growth and forces along axons regulates circumferential convolutions on gyral banks. — [Zhang et al. 2017, J Comput Neurosci 42:197 (pages unverified)](https://doi.org/10.1007/s10827-017-0637-9)
- "Mechanical hierarchy" (bilayer spheres with initial undulations and uniform or heterogeneous axon fibers): **differential tangential growth is the inducer** of folding. Initial undulations and axonal fibers "regulate the folding patterns and determine the location of gyri and sulci", and "locations with dense axonal fibers after folding settle in gyri rather than sulci". Fiber density also correlates with local gyrification index. — [Chavoshnejad et al. 2023, Sci Rep 13:13177 (DOI verified; article number unverified)](https://doi.org/10.1038/s41598-023-40086-9)
- Tension-induced fiber growth (consequence model): "Simulations incorporating stress-induced fiber elongation indicate that subcortical stresses resulting from folding are sufficient to induce stereotyped fiber organization beneath gyri and sulci", supported by high-resolution ex vivo DTI of fetal rhesus macaque. The model explicitly targets "subcortical remodeling that occurs during the period of folding". — [Garcia, Wang, Kroenke 2021, Nat Commun 12:6681](https://doi.org/10.1038/s41467-021-26971-9)
- Stress landscape and axon pathfinding: an "axon reorientation" model with stress-induced growth gives "higher axon bundle density in gyri compared to sulci", depending on the growth rate and stiffness of the navigating bundles. Supported by in-vivo human DTI. — [Solhtalab, Foroughi, Pierotich, Razavi 2025, Nat Commun 16:1187](https://doi.org/10.1038/s41467-025-56362-3)
- Follow-up preprint (not peer-reviewed): stress-dependent axon reorientation explains U-fibers, and why gyrus-to-gyrus connections outnumber sulcus-to-sulcus ones. — [Solhtalab et al. 2026, bioRxiv 10.64898/2026.08.12.744475 (DOI prefix as returned by PubMed; unusual, verify)](https://doi.org/10.64898/2026.08.12.744475)
- Axonal-elongation continuum model (Holland, Miller, Kuhl): white-matter volume change results from microscopic axonal growth calibrated on chick neurons. Anisotropic white-matter growth "intrinsically induces symmetry breaking" and gives regionally varying wavelengths and sulcal depths. — [Holland, Miller, Kuhl 2015, Ann Biomed Eng 43:1640](https://doi.org/10.1007/s10439-015-1312-9)
- Theoretical white-matter stiffness map from fiber architecture (FE + CNN, tractography-based anisotropic constants). This is a tool, not a fetal measurement. — [Chavoshnejad et al. 2024, Phys Biol 21](https://doi.org/10.1088/1478-3975/ad88e4)

### Inferences
- Our isotropic δ field is a scalar proxy. In Garcia 2021 and Solhtalab 2025 the subcortical growth is **anisotropic** (along fibers) and driven by stress. A single isotropic growth multiplier with δ>0 under crests and δ<0 under fundi is the natural isotropic projection of "fibers elongate radially under gyri (tension) and are compressed or reoriented tangentially under sulci". This interpretation is consistent with the literature, but our fit cannot distinguish it from a pre-patterned (causal) heterogeneity.
- One test is to compare the *timing* of δ with the timing of folding. If δ contrast appears only after a fold starts, it fits the consequence models. If δ is needed before any visible indentation, it fits pre-patterning.

### Gaps
- None of these models fit to dHCP data or invert for a growth field. I found no prior adjoint or inverse estimate of a subcortical growth map, so our result appears to be novel in that respect. This is not exhaustively confirmed.
- Typical magnitudes of fiber-driven growth heterogeneity in these models (for comparison with |δ|≈0.3) were not accessible from the abstracts.

---

## 3. Subplate and oSVZ / basal radial glia differences under future gyri vs sulci (ferret, primate, human)

### Takeaway
There is direct histological evidence in ferret and marmoset, and supporting gene-expression evidence in ferret, that germinal and subplate compartments differ under prospective gyri versus sulci, **before** the folds appear. The oSVZ is thicker and more proliferative, and the subplate grows locally, under future gyri. In primates, Rakic's group argues that oSVZ neurogenesis ends before most gyri form, and that the relevant subcortical driver during folding is oSVZ **gliogenesis**, white-matter enlargement and neuropil growth. Ferret experiments (Shinmyo 2022) show that localized astrogenesis in deep regions is required for gyri. All of this is the same sign as our δ (more subcortical growth under crests).

### Cited Findings
- Ferret classic histology: "a gyrus was initiated by an area of **local growth appearing in the subplate** and then in the suprajacent segment of cortical plate". "At the site of the future sulci the cortical plate remained immature and growth was retarded." Later, tangential growth concentrates in the parasulcal area. — [Smart & McSherry 1986, J Anat 147:27 (PubMed 3693076; no DOI)](https://pubmed.ncbi.nlm.nih.gov/3693076/)
- Intermediate (basal) radial glia (IRGCs; Pax6+/Tbr2−, pial fiber only) are exclusive to gyrencephalic species and massively generated in the OSVZ. Their fanning fibers spread migrating neurons tangentially. "The tangential expansion of particular cortical regions was **preceded by high proliferation in the underlying OSVZ**", and reducing IRGCs reduced cortical surface. — [Reillo et al. 2011, Cereb Cortex 21:1674 (epub 2010)](https://doi.org/10.1093/cercor/bhq238)
- Ferret germinal zones: ISVZ and OSVZ emerge at different times, differ in IRGC abundance, and have different cell-cycle kinetics. — [Reillo & Borrell 2012, Cereb Cortex 22:2039](https://doi.org/10.1093/cercor/bhr284)
- Transcriptomics of individual germinal layers in ferret, comparing "regions prospective of fold and fissure": thousands of genes are differentially expressed, including ~80% of the genes mutated in human cortical malformations. Discrete expression domains "map the eventual location of folds or fissures" in ferret and human, but not mouse. — [de Juan Romero et al. 2015, EMBO J 34:1859](https://doi.org/10.15252/embj.201591176)
- **Marmoset, directly relevant to our calcarine fits:** at GW13, "the outer subventricular zone (oSVZ) in non-sulcal regions was thicker than that in the presumptive **calcarine** sulcal region", *preceding* calcarine infolding. "Differential expansion of the oSVZ may be involved in gyral convolution and sulcal infolding." — [Sawada et al. 2014, Neuroscience 257:158](https://doi.org/10.1016/j.neuroscience.2013.10.067)
- Macaque occipital lobe: going from presumptive area 17 to area 18 there is "progressive thinning of the radially structured subventricular zone" (OSVZ). The subplate is augmented late. — [Smart et al. 2002, Cereb Cortex 12:37](https://doi.org/10.1093/cercor/12.1.37)
- Human OSVZ has large numbers of non-epithelial, self-renewing radial-glia-like cells (oRG). — [Hansen et al. 2010, Nature 464:554](https://doi.org/10.1038/nature08845)
- bRG are abundant even in the near-lissencephalic marmoset, so a high bRG abundance "may be necessary, but is not sufficient, for gyrencephaly". — [Kelava et al. 2012, Cereb Cortex 22:469](https://doi.org/10.1093/cercor/bhr301)
- **Primate counterpoint (Rakic group).** In macaque parietal cortex, the oSVZ first adds superficial neurons, then switches to producing astrocytes and oligodendrocytes. Neurogenesis is complete around E90 while the cerebrum is still lissencephalic, and "oSVZ **gliogenesis**, rather than neurogenesis, correlates with rapid enlargement of the cerebrum and development of convolutions". — [Rash et al. 2019, PNAS 116:7089](https://doi.org/10.1073/pnas.1822169116)
- Same group: neurogenic "hotspots" in the oSVZ are contradicted by the timing. Intracortical neuropil growth, glial addition and **enlargement of subcortical white matter** are "the primary forces" of post-neurogenic surface expansion and gyrus formation. — [Rash et al. 2023, PNAS 120:e2210967120](https://doi.org/10.1073/pnas.2210967120)
- Ferret gain- and loss-of-function: "reducing astrocyte number prevents gyri formation in the ferret cortex, while increasing astrocyte number in mice … can induce gyrus-like protrusions". "The vertical expansion of **deep pallial regions** achieved by localized astrogenesis is crucial for gyri formation" (FGF positive feedback). — [Shinmyo et al. 2022, Sci Adv 8:eabi5209](https://doi.org/10.1126/sciadv.abi5209)
- FE follow-up (preprint, not peer-reviewed): astrocytes are "especially prominent under the gyri". Two hypotheses are compared, morphogenetic (astrocytes push the cortex out) versus stress-driven (astrocytes proliferate in response to tension). The stress-driven version reproduces the subcortical tension seen experimentally and is judged "more likely". — [Taneja, Saito, Kawasaki, Holland 2025, bioRxiv](https://doi.org/10.1101/2025.10.17.683172)
- Cortical plate (not core): FLRT1/3 deletion makes mouse cortex form sulci through reduced adhesion and faster migration, without progenitor amplification. "FLRT1/3 expression is low in the human cortex and in **future sulcus areas of ferrets**." — [Del Toro et al. 2017, Cell 169:621 (pages unverified)](https://doi.org/10.1016/j.cell.2017.04.012). Follow-up: combining with progenitor expansion enhances folding, and expanding intermediate progenitors favors sulci while expanding apical progenitors favors gyri. — [Chun et al. 2025, Nat Commun](https://doi.org/10.1038/s41467-025-62858-9)
- Subplate review: the subplate is proposed as a "potential driver" of region-dependent sulcal/gyral development through axon guidance cues before folds appear. This is a hypothesis-level review. — [Rana et al. 2019, Cereb Cortex 29:4697](https://doi.org/10.1093/cercor/bhz003)
- Human in-vivo MRI (68 fetuses, 22–32 GW): subplate thickness is regionally heterogeneous after correcting for geometry, highest in parietal and perisylvian regions. Mean SP thickness grows +39.2% (±11.0 SD across regions) while surface area grows more uniformly (+64.3% ±0.7). There are early (pericentral, medial occipital) and late (prefrontal, temporal, parietal) trajectories. — [Gondová et al. 2026, PNAS 123:e2528620123](https://doi.org/10.1073/pnas.2528620123)
- In fetuses with congenital heart disease (n=260), reduced SP volume and thickness was associated with disturbed sulcal development. This is correlational. — [Wu et al. 2022, Cereb Cortex 32:2858](https://doi.org/10.1093/cercor/bhab386)
- Reviews framing the cellular determinants: [Llinares-Benadero & Borrell 2019, Nat Rev Neurosci](https://doi.org/10.1038/s41583-018-0112-2); [Borrell 2018, J Neurosci 38:776 (pages unverified)](https://doi.org/10.1523/JNEUROSCI.1106-17.2017); [Del-Valle-Anton & Borrell 2022, Physiol Rev](https://doi.org/10.1152/physrev.00016.2021); [Fernández & Borrell 2023, Curr Opin Neurobiol](https://doi.org/10.1016/j.conb.2023.102711).

### Inferences
- The δ<0 under the calcarine fundus that we fit is qualitatively matched by Sawada 2014 (thinner oSVZ at the presumptive calcarine sulcus in marmoset) and Smart 2002 (thinning of the radial SVZ toward the area 17/18 border in macaque). This is the most specific literature match to our calcarine result, though in other primates.
- In the dHCP fetal age range (roughly 21–36 GW), human neurogenesis is largely complete for most regions (per Rash 2019/2023 in macaque, by analogy). The candidate biological substrates for a "core" growth excess under crests are therefore (a) gliogenesis and astrocyte expansion in deep regions (Shinmyo 2022, Rash 2019), (b) subplate expansion and axonal ingrowth (Smart & McSherry 1986, Gondová 2026), and (c) fiber elongation in white matter (Garcia 2021). Neuron production by oSVZ hotspots is less plausible at these ages in primates (contested; see Rash 2023 versus the Borrell school).
- A Tallinen "core" lumps the subplate, IZ/white matter and germinal zones together, so δ can only represent their sum.

### Gaps
- I found no human fetal histology quantifying oSVZ or subplate thickness *specifically* under prospective gyri versus sulci (the ferret and marmoset data are the direct ones). Gondová 2026 reports regional (lobar/parcel) differences, not gyral-vs-sulcal.
- Kriegstein et al. 2006 (Nat Rev Neurosci 7:883, "Patterns of neural stem and progenitor cell division may underlie evolutionary cortical expansion") is often cited for regional SVZ amplification predicting gyri. **[unverified]** I did not retrieve it in this session.
- Exact OSVZ proliferation ratios under gyrus versus sulcus in Reillo 2011 are not in the abstract.

---

## 4. Models where subcortical (or regional) growth heterogeneity sets fold location

### Takeaway
Several model families can place folds through a spatial bias. The bias can be in **cortical** growth or timing (Toro & Burnod 2005; Xu 2010; Zhang 2016), in **subcortical** fibers or tension (Chavoshnejad 2023; Wang & Holland 2024), in **cell-density-driven** growth from germinal zones (Zarzor & Budday 2023/2024), or in **migration and adhesion** (Del Toro 2017; Chun 2025). Explicit models where subcortical isotropic growth is heterogeneous and fitted to data are rare. The closest are the Zarzor/Budday multifield models (OSVZ → cell density → growth) and the stress-driven subcortical growth models (Garcia 2021; Taneja 2025).

### Cited Findings
- Toro & Burnod: closed cortical surface with glial and axonal fibres pulling radially, plus elasticity, plasticity and surface growth. Introducing "geometric, mechanic and growth asymmetries" (early regionalization) reproduces gradients in convolution and the primary, secondary and tertiary sequence. — [Toro & Burnod 2005, Cereb Cortex 15:1900](https://doi.org/10.1093/cercor/bhi068)
- Zarzor, Blümcke, Budday: a multifield model couples proliferation in VZ/ISVZ/OSVZ and migration (advection-diffusion) to finite growth. It is validated on human fetal histology and used to ask how the OSVZ affects folding. — [Zarzor et al. 2023, eLife 12:e82925](https://doi.org/10.7554/eLife.82925)
- Extension: cell-density fields for RGC, IPC, oRG and neurons, with cortex expansion driven by cell density, compared with fetal MRI. — [Zarzor et al. 2024, Sci Rep 14:26103](https://doi.org/10.1038/s41598-024-75952-7)
- Curvature-coupled growth (Holland group): cortical growth depends on mean curvature, so "inhomogeneous cortical growth [is] the result of folding rather than the cause". With similar cortical and subcortical stiffness, the fit to thickness data suggests "cortical growth is higher in gyri than in sulci". Gyral-sulcal thickness variation appears at low stiffness ratios. — [Wang S, Demirci, Holland 2021, Biomech Model Mechanobiol 20:555](https://doi.org/10.1007/s10237-020-01400-w)
- Thickness asymmetry is a **generic physical outcome** of wrinkling in a growing layer: "Gyri are universally thicker than sulci", without needing pre-patterned growth. — [Holland, Budday, Goriely, Kuhl 2018, Phys Rev Lett 121:228002](https://doi.org/10.1103/PhysRevLett.121.228002)
- Thickness inhomogeneity in the cortex yields deep primary sulci, then secondary undulations. — [da Costa Campos et al. 2021, NeuroImage 231:117779](https://doi.org/10.1016/j.neuroimage.2021.117779)
- Ferret cortical growth is spatially non-uniform. Relative growth increases with distance from the origin of the transverse neurogenetic gradient and relates to FA change, which is a timing-gradient bias. — [Knutsen et al. 2013, Cereb Cortex 23:488](https://doi.org/10.1093/cercor/bhs042)
- Ferret and human gel and FE models: cortical thickness and expansion rate are primary drivers of dysmorphogenesis, and differential growth plus initial geometry suffices across species. — [Choi et al. 2025, eLife 14:e107141](https://doi.org/10.7554/eLife.107141); [Yin et al. 2025, eLife 14:e107138](https://doi.org/10.7554/eLife.107138)
- Tallinen et al. 2016, "On the growth and form of cortical convolutions", Nat Phys 12:588, is our base model. Isotropic tangential cortical growth over a slower-growing core reproduces human folding from fetal MRI geometry. — **[unverified in this session; not indexed in PubMed]** [doi:10.1038/nphys3632](https://doi.org/10.1038/nphys3632). A related verified paper on the bilayer instability regimes as a function of modulus ratio: [Tallinen & Biggins 2015, Phys Rev E 92:022720](https://doi.org/10.1103/PhysRevE.92.022720)
- "Wang two-layer model with subplate": I could not identify a specific paper matching this description (Xianqiao Wang's group = Razavi/Zhang papers above; Xincheng/Shuolun Wang = Holland group). The closest verified "subplate as distinct layer" mechanical treatment is Xu et al. 2010 (subplate remodeling). Striedter, Srinivasan & Monuki 2015 (Annu Rev Neurosci, "Cortical folding: when, where, how, and why?") is **[unverified]** in this session.

### Inferences
- Our adjoint result is a data-driven version of the Toro & Burnod / Xu 2010 idea of a spatial perturbation of growth. Unlike those models, the perturbation sits in the core.
- **Main confound:** Holland 2018 and Wang/Holland 2021 show that the thickness contrast (gyri thicker) arises from mechanics alone, and that curvature-coupled *cortical* growth fits the data. Because δ is co-located with the thickness contrast, the adjoint may be using core growth to compensate for missing gyral-sulcal cortical growth asymmetry, or for missing anisotropy. This is an identifiability problem between cortical and subcortical growth heterogeneity. It is worth testing with a parameterization that lets cortical growth vary too (or curvature-coupled growth), to see whether δ survives.

### Gaps
- I found no published inverse or adjoint estimation of a spatial subcortical growth field from fetal MRI, so I cannot compare magnitudes with |δ|≈0.3.
- The Zarzor models predict specific OSVZ-driven radial growth profiles under gyri, but their quantitative gyrus/sulcus subcortical growth ratios are not in the abstracts.

---

## 5. Cause vs consequence; strength of evidence

### Takeaway
The evidence that subcortical compartments **differ** under future gyri and sulci is moderate to strong in ferret, moderate in non-human primates, and weak to indirect in human fetuses. The evidence that this difference **causes** fold placement is experimental only in ferret and mouse (astrogenesis manipulation; FLRT and progenitor genetics), and those manipulations act on the cortical plate or deep regions rather than cleanly on "core growth". Mechanics papers increasingly treat subcortical growth heterogeneity (fiber growth, astrocyte proliferation, white-matter remodeling) as a **feedback response to folding stress** that reinforces folds. Both readings remain open.

### Cited Findings
- Causal / pre-pattern evidence:
  - Gene expression in germinal layers maps the future folds before folding (ferret, human). — [de Juan Romero 2015](https://doi.org/10.15252/embj.201591176)
  - oSVZ thinner under the presumptive calcarine sulcus before infolding (marmoset). — [Sawada 2014](https://doi.org/10.1016/j.neuroscience.2013.10.067)
  - OSVZ proliferation precedes regional tangential expansion, and IRGC reduction shrinks the surface (ferret). — [Reillo 2011](https://doi.org/10.1093/cercor/bhq238)
  - Local subplate growth initiates the gyrus (ferret, descriptive). — [Smart & McSherry 1986](https://pubmed.ncbi.nlm.nih.gov/3693076/)
  - Astrocyte reduction abolishes gyri, astrocyte increase induces protrusions (ferret, mouse). — [Shinmyo 2022](https://doi.org/10.1126/sciadv.abi5209)
- Consequence / feedback evidence:
  - Folding stress suffices to produce stereotyped fiber organization beneath gyri and sulci (model plus macaque DTI). — [Garcia 2021](https://doi.org/10.1038/s41467-021-26971-9)
  - Stress-driven axon reorientation gives denser axons in gyri. — [Solhtalab 2025](https://doi.org/10.1038/s41467-025-56362-3)
  - Stress-driven astrocyte proliferation is favored over autonomous pushing (preprint). — [Taneja 2025](https://doi.org/10.1101/2025.10.17.683172)
  - Inner zone growth "perhaps in response to stress". — [Kroenke & Bayly 2018](https://doi.org/10.1523/JNEUROSCI.1105-17.2017)
  - Thickness contrast is generic physics. — [Holland 2018](https://doi.org/10.1103/PhysRevLett.121.228002)
- Against axonal tension as the *driver*: tension is not directed across gyri. — [Xu 2010](https://doi.org/10.1115/1.4001683)
- Integrative position: spatio-temporal patterns bias primary fold location, and cortical tangential growth drives the instability. — [Garcia, Kroenke, Bayly 2018](https://doi.org/10.1098/rstb.2017.0321); [Chavoshnejad 2023](https://doi.org/10.1038/s41598-023-40086-9)
- Timing objection to oSVZ-neurogenesis hotspots in primates. — [Rash 2023](https://doi.org/10.1073/pnas.2210967120)

### Inferences
Plausible readings of our δ (crest +, fundus −, |δ|≤0.3), roughly ordered by literature support for the fetal-human, dHCP age range:
1. **Stress-driven subcortical growth** (fiber elongation, astrocyte proliferation in gyral white matter, in response to tension under gyri and compression under sulci). This is a *consequence* that reinforces folds and has the same sign as δ. Supported by Garcia 2021, Solhtalab 2025, Taneja 2025 and the residual-stress data. A testable prediction is that δ should correlate with the model's own subcortical stress or stretch field.
2. **Pre-patterned germinal-zone or subplate expansion under future gyri** (OSVZ, subplate, localized astrogenesis). Same sign. Direct evidence exists in ferret and marmoset (including calcarine), but human gyrus-vs-sulcus data are lacking.
3. **Model artifact or compensation.** δ could absorb missing cortical-growth heterogeneity (curvature-coupled growth; Wang/Holland 2021), anisotropy of fiber growth, or a stiffness contrast. Co-location with the thickness contrast makes this a serious alternative, since Holland 2018 shows thickness asymmetry arises without any subcortical heterogeneity.
- A stiffness reading of δ is not well supported. I found no fetal data on gyral versus sulcal subcortical stiffness. Measured white-matter heterogeneity is mostly anisotropic and fiber-related (Chavoshnejad 2024 is theoretical).

### Gaps
- No human fetal study directly shows gyral-vs-sulcal subcortical growth-rate differences during 21–36 GW (longitudinal deformation-based morphometry of the dHCP atlas could itself provide this).
- I found no experiment that perturbs subcortical growth *without* affecting cortical-plate neuron number, which would separate the two causal routes cleanly.
- Not verified in this session: Tallinen 2016 (Nat Phys), Kriegstein 2006, Striedter 2015. They are listed above with an [unverified] mark.
