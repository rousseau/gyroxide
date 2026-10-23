# Fitting / calibrating mechanical models of cortical folding to MRI: state of the art (as of Oct. 2026)

Context for the novelty judgement. Our approach: exact adjoint gradient through the quasi-static equilibrium of a Tallinen 2016 model. It is fitted to the dHCP fetal surface atlas (SG 26→28) with vertex-to-vertex correspondence, using per-vertex fields (tangential growth, thickening, core growth) and then a 6-coefficient law. The loss is composite: vertex distance, face normals, Laplacian, area and point-to-plane.

## Q1. Which inverse or data-fitted folding models exist, and what exactly do they fit?

### Takeaway
Only one published work solves a true PDE-constrained inverse problem with an adjoint on a growing fetal brain: Wang, Martin, Weickenmeier & Garikipati 2021, using dolfin-adjoint. It fits a nodal growth tensor F_g to *registration displacement fields* between weekly atlas volumes at 21–25 GW, before or at the very start of folding, with a smooth (non-folding) neo-Hookean FE model. All the other "MRI-informed" models fall into two groups:
- (a) they derive growth laws directly from MRI morphometry (surface area, thickness), then run the forward model with no feedback (Hou/Wang/Li/Kuhl group: Soft Matter 2025, Nat Commun 2026);
- (b) they sweep a few scalar parameters and compare with global, spectral or regional metrics (BrainGrowth Sci Rep 2021, FetalFoldSim arXiv 2026, Alenyà 2022).

### Cited findings

**Adjoint inverse on fetal brain (closest prior work)**
- Wang Z., Martin B., Weickenmeier J., Garikipati K., "An inverse modelling study on the local volume changes during early morphoelastic growth of the fetal human brain", *Brain Multiphysics* 2 (2021) 100023, DOI 10.1016/j.brain.2021.100023. — [PMC8186493](https://pmc.ncbi.nlm.nih.gov/articles/PMC8186493/)
  - **Data:** the Gholipour spatio-temporal fetal MRI atlas, weeks 21–25. Cortex, subcortex and ventricles are segmented and meshed. Displacement fields between consecutive weeks come from B-spline registration (L2 gradient flow). — [PMC8186493](https://pmc.ncbi.nlm.nih.gov/articles/PMC8186493/)
  - **Forward model:** neo-Hookean morphoelasticity, F = F_e·F_g, quasi-static FEM. The reference configuration is updated at each step. — same source
  - **Unknown:** a diagonal anisotropic F_g represented at the nodes, with det F_g as the local volume change. — same source
  - **Optimiser:** Adam on the forward model was compared with an adjoint method (dolfin-adjoint plus SciPy L-BFGS-B). The adjoint was retained for real data because it gave lower error and smoother fields. — same source
  - **Loss:** L2 norm of (MRI displacement − simulated displacement), under the weak-form residual constraint. — same source
  - **Cost:** about 2,000 adjoint iterations and about 900 CPU-hours. Final error is ≤ 2 % of the maximum displacement. det F_g ranges over 0.65–1.35. Tangential growth exceeds normal growth. — same source
  - **Stated limitation:** registration fails when new structures appear between weeks, which restricts the study to 1-week intervals and to the *pre-folding* period. — same source

**Data-driven growth laws, no inverse loop (UGA/UNC/Stanford group)**
- Hou J., Wu Z., Chen X., Wang L., Zhu D., Liu T., Li G., Wang X., "Role of data-driven regional growth model in shaping brain folding patterns", *Soft Matter* 21(4):729–749 (2025), DOI 10.1039/d4sm01194e. — [PMC11718650](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718650/); arXiv preprint [2408.17334](https://www.alphaxiv.org/abs/2408.17334)
  - **Data:** more than 1000 MRI scans from 735 subjects, aged 29 PMW to 2 years (dHCP and BCP). — [PMC11718650](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718650/)
  - **Growth laws:** tangential growth g_t is fitted to surface area and radial growth g_r to thickness, by symbolic regression (PySR) on 5 of 18 regions. Typical form: (a·t+b)·tanh(c·t)+1. — same source
  - **Simulation:** ABAQUS on an 80×80×50 mm *patch*, with growth implemented as thermal expansion. — same source
  - **Comparison metrics:** mean curvature, GI and sulcal depth, scored with R², MAPE and Pearson correlation. Mean curvature is poorly predicted (R² = 0). — same source
- Hou J., Wu Z., Jiang K., Wu T., Zhang L., Zhu D., Gao W., Razavi M.J., Liu T., **Kuhl E.**, Li G., Wang X., "Biophysical modeling of anatomically realistic prenatal cortical folding development", *Nat Commun* 17 (2026), DOI 10.1038/s41467-026-75582-9 (published 16 Jul 2026). — [PubMed 42463653 / PMC13490475](https://pmc.ncbi.nlm.nih.gov/articles/PMC13490475/)
  - The Research Square preprint is DOI 10.21203/rs.3.rs-8033969/v1, dated 12 Jan 2026. — [PMC12869665](https://pmc.ncbi.nlm.nih.gov/articles/PMC12869665/)
  - **Data:** dHCP, 120 scans at 22–40 GW, surfaces from iBEAT V2. Growth laws come from symbolic regression per region (18 per hemisphere). Normalisation: g_t = S_t/S_0 and g_r = T_t/T_0, with virtual time t ∈ [0,1]. — [PMC13490475](https://pmc.ncbi.nlm.nih.gov/articles/PMC13490475/)
  - **Forward model:** Abaqus explicit dynamics with VUEXPAN, F_g = g_t I + (g_r − g_t) n⊗n (Tallinen-type). About 1.6 M hex elements, initial geometry at 22 GW, simulated from 22 to 40 GW. — same source
  - **Calibration:** none on folding. Per our reading of the full text, the paper describes no optimisation of the forward model against folding outcomes. The alternative trajectories are "empirically assigned". — same source
  - **Metrics:** global GI trajectories; a local similarity score exp(−|ln(d_sim/d_real)|) on GI and thickness; sulcus counts (H < −0.05, area > 10 mm²); regional sulcal depth on 18 regions (Wilcoxon); 6 sulci delineated by hand and compared qualitatively (central, postcentral, intraparietal, superior temporal, lateral, inferior frontal); shape-index similarity > 0.8 "at vertex level", against the *average* of real brains. The authors state these are "developmental benchmarks rather than direct subject-level comparisons". — same source

**Metric-based sweeps on MRI or synthetic geometries (BrainGrowth / FetalFoldSim lineage, our team)**
- Wang X., Lefèvre J., Bohi A., Al Harrach M., Dinomais M., Rousseau F., "The influence of biophysical parameters in a biomechanical model of cortical folding patterns", *Sci Rep* 11:7686 (2021), DOI 10.1038/s41598-021-87124-y. — [PMC8032759](https://pmc.ncbi.nlm.nih.gov/articles/PMC8032759/)
  - **Model:** Tallinen model in Python (BrainGrowth), on an ellipsoid. — same source
  - **Parameters varied by hand:** growth mode, thickness 0.03–1.63 mm, elongation. — same source
  - **Metrics:** mean curvature, 3D GI, sulcal depth (convex hull), fold orientation (Fiedler vector vs principal directions, KL divergence). — same source
  - No MRI comparison and no estimation. — same source
- Bohi A., Wang X., Harrach M., Dinomais M., Rousseau F., Lefèvre J., "Global perturbation of initial geometry in a biomechanical model of cortical morphogenesis", *IEEE EMBC* 2019:442–445, DOI 10.1109/EMBC.2019.8856723. Compares simulations started from different initial geometries via adaptive spherical parameterization. — [PubMed 31945933](https://pubmed.ncbi.nlm.nih.gov/31945933/)
- Wang X., Bohi A., Al Harrach M., Dinomais M., Lefèvre J., Rousseau F., "On early brain folding patterns using biomechanical growth modeling", *IEEE EMBC* 2019:146–149, DOI 10.1109/EMBC.2019.8856670. Quantifies folds with 3D GI and spectral analysis (SpAnGy). — [PubMed 31945865](https://pubmed.ncbi.nlm.nih.gov/31945865/)
- Kerachni A., Lavigne T., Urcun S., Alenyà M., Camara O., Lefèvre J., Rousseau F., "Open-source MRI-informed computational model of human cortical folding" (FetalFoldSim), arXiv 2607.08534 (July 2026, preprint). — [arXiv](https://arxiv.org/abs/2607.08534)
  - **Setup:** FEniCS with an interhemispheric collision penalty. Input geometry is the dHCP surface atlas at 21 GW, simulated up to about 28 GW. — same source
  - **"Optimisation":** a grid sweep. Thickness H0 ∈ {0.8, 1.5, 2.25, 6} mm and the stiffness ratio (1–20) are compared with the dHCP atlas at 28–29 GW using SpAnGy band powers (primary, secondary and tertiary folds), plus shape index and curvedness. — same source
  - **Positioning:** the authors claim the metrics "are used for the first time to optimize the similarity of the model to healthy neurodevelopmental data". — same source
  - **Limitations they list:** parameters constant in space and time; heterogeneity of thickness, stiffness ratio and growth rate "essential to study" (future work); collisions beyond 28 GW. — same source
- Alenyà M., Wang X., Lefèvre J., Auzias G., Fouquet B., Eixarch E., Rousseau F., Camara O., "Computational pipeline for the generation and validation of patient-specific mechanical models of brain development", *Brain Multiphysics* 3 (2022) 100045, DOI 10.1016/j.brain.2022.100045.
  - 29 subjects with fetal and neonatal MRI. Simulations start from the fetal MRI and are compared with the neonatal MRI of the *same subject*, in a common space, "using several metrics". — [UPF repository (abstract via search)](https://repositori.upf.edu/items/11a66ca6-2f92-494d-92ea-fb68e31fbc92)
  - The exact metrics are **unverified**: the full text was not accessible (403 / access denied).

**Other MRI-to-model works (kinematics, not inversion of a mechanical model)**
- Garcia K.E., Robinson E.C., …, Bayly P.V., "Dynamic patterns of cortical expansion during folding of the preterm human brain", *PNAS* 115(12):3156–3161 (2018), DOI 10.1073/pnas.1715451115.
  - 30 preterm infants (28–38 wk), longitudinal. Correspondence via anatomically constrained MSM (strain-energy regularised). Local expansion is measured. — [TUM portal record](https://portal.fis.tum.de/en/publications/dynamic-patterns-of-cortical-expansion-during-folding-of-the-pret/); [BALSA](https://balsa.wustl.edu/study/K65Z)
  - It measures growth; no mechanical model is fitted.
- Choi G.P.T., …, Mahadevan L., "Biophysical basis for brain folding and misfolding patterns in ferrets and humans", *eLife* 14 (2025), DOI 10.7554/eLife.107141. Physical gels and simulations from ferret MRI; thickness and expansion rate identified as drivers. — [PubMed 41459648](https://pubmed.ncbi.nlm.nih.gov/41459648/)
- Wang S., Saito K., Kawasaki H., Holland M.A., "Orchestrated neuronal migration and cortical folding", *PLoS Comput Biol* 18:e1010190 (2022), DOI 10.1371/journal.pcbi.1010190. Model "calibrated" on ferret in utero electroporation, with *qualitative* agreement. — [PubMed 35709293](https://pubmed.ncbi.nlm.nih.gov/35709293/)
- Wang S., Holland M.A. et al., "Axonal tension contributes to consistent fold placement", *Soft Matter* (2024), DOI 10.1039/d4sm00129j. Explains why folds appear at particular locations. Mechanistic, with no fitting. — [PubMed 38506323](https://pubmed.ncbi.nlm.nih.gov/38506323/)
- Zhao Y., Song Y., Xu F., Xu Z., "Predicting brain morphogenesis via physics-transfer learning", arXiv 2509.05305 (2025, preprint).
  - Neural networks trained on FE simulations of spheres and ellipsoids, then transferred to brain.
  - Composite loss: 3D coordinates + GI + local curvatures.
  - Curvature-map MAE < 0.5 mm⁻¹ against MRI. This is a learned surrogate, not an inversion of a mechanical model. — [arXiv](https://www.alphaxiv.org/abs/2509.05305)

**Kuhl group**
- Constitutive neural networks (CANN) discover the *constitutive law* of brain tissue from mechanical tests, choosing among 4095 models. Linka, St. Pierre, Kuhl, *Acta Biomaterialia* 2023. — [bioRxiv 2022.11.08.515656](https://www.biorxiv.org/content/10.1101/2022.11.08.515656.full.pdf)
  - This is ex vivo rheology, not folding.
  - Bayesian / Gaussian versions of CANNs exist (Linka, Holzapfel, Kuhl, CMAME; arXiv 2503.12679), but not for folding. — [arXiv 2503.12679](https://arxiv.org/pdf/2503.12679)
- Kuhl's direct contribution to fitting folding to MRI goes through his co-authorship of Nat Commun 2026, which has no inverse loop (see above).
- I found no Bayesian calibration of a folding model against MRI.

### Inferences
- **Closest prior art:** the adjoint idea itself (dolfin-adjoint, nodal growth field, L-BFGS) already exists in Wang/Garikipati 2021. Our work differs in four ways that can be stated explicitly:
  - (i) The equilibrium is *post-buckling / folded* (26→28 SG, emerging sulci). Wang 2021 treats 21–25 GW, before folding, explicitly because registration breaks down when folds appear.
  - (ii) Our forward model is Tallinen 2016: thickness-dependent growth, contact, and a cortex/core distinction. Theirs is a generic neo-Hookean with F_g.
  - (iii) We fit *surface* data from the dHCP atlas (vertex-to-vertex correspondence). They fit volumetric registration displacements.
  - (iv) We separate fields (tangential growth, thickening, core growth), then reduce them to a 6-coefficient law, i.e. we go from field identification to identification of a *law*.
- The Nat Commun 2026 model is the most "realistic" competitor, but it is a forward model with growth prescribed from morphometry. Its comparisons are statistical: GI, regional sulcal depth, shape-index distributions against the average brain. Nothing is optimised to reproduce a given sulcus.
- In our own team's lineage (BrainGrowth → Alenyà → FetalFoldSim), FetalFoldSim's "optimisation" is a sweep over scalars evaluated with SpAnGy. Its stated future work (spatial heterogeneity of growth and thickness) is exactly what the adjoint approach does.

### Gaps
- Exact metrics and possible tuning in Alenyà et al. 2022: full text not accessible here. To verify (DOI 10.1016/j.brain.2022.100045).
- Not checked: possible follow-ups to Wang/Garikipati 2021 in 2022–2026 (e.g. a version with folding); my searches found none.
- Not checked: theses (e.g. Alenyà PhD, UPF 2024) and MICCAI/PIPPI workshop papers, which may contain unindexed calibration attempts.

## Q2. Which cost functions or similarity metrics are used to compare simulation and MRI?

### Takeaway
The literature is dominated by global or statistical metrics: GI or 3D-GI, mean curvature, sulcal depth, shape index and curvedness, SpAnGy spectral bands, and counts or qualitative landmarks of sulci. The only differentiable "data-attachment" loss used in a real inverse loop is the L2 displacement error of Wang 2021. No folding work uses a composite geometric loss like ours (vertex distance + normals + Laplacian + area + point-to-plane) on a surface correspondence.

### Cited findings
- **GI, mean curvature, sulcal depth**, with R²/MAPE/Pearson scores. — [Hou et al. Soft Matter 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718650/)
- **Global GI; local similarity exp(−|ln(d_sim/d_real)|); sulcus counts; regional sulcal depth (18 regions); 6 landmark sulci qualitatively; vertex-level shape-index similarity against the average brain.** — [Hou et al. Nat Commun 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13490475/)
- **Mean curvature, 3D GI, convex-hull sulcal depth, orientation via Fiedler vector + KL.** — [Wang et al. Sci Rep 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8032759/)
- **SpAnGy bands, shape index, curvedness** (FetalFoldSim). Global metrics such as GI are judged insufficient because of the spatio-temporal gradient of folding, hence the preference for local metrics. — [arXiv 2607.08534](https://arxiv.org/abs/2607.08534)
- **SpAnGy itself** decomposes mean curvature on Laplace–Beltrami eigenfunctions into 7 bands (B0–B6). B4–B6 carry 93 % of the power and are tied to folds. — Germanaud et al., *NeuroImage* 63:1257–1272 (2012), PMID 22877579 — [PDF](https://www.unicog.org/publications/germanaudLefevre_spangyFoldingAnalysis_nimg12.pdf)
- **Developmental application of SpAnGy:** three folding waves (primary, secondary, tertiary) in newborns aged 27–62 wk. — Dubois et al., *NeuroImage* 185:934–946 (2019), DOI 10.1016/j.neuroimage.2018.03.005 — [PubMed 29522888](https://pubmed.ncbi.nlm.nih.gov/29522888/)
- **L2 norm of the displacement difference** (volumetric registration) under the residual constraint. — [Wang/Garikipati 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8186493/)
- **NN loss of 3D coordinates + GI + curvatures** (learned surrogate). — [arXiv 2509.05305](https://www.alphaxiv.org/abs/2509.05305)
- **Outside the brain:** a fully differentiable inverse framework (JAX-FEM, exact gradients via AD) with loss = nodal-position misfit + deformation-gradient mismatch, applied to MRI breast data. — Bae, Hu, Buganza Tepole, Gomez, arXiv 2609.18109 (2026, preprint) — [arXiv](https://www.alphaxiv.org/abs/2609.18109)
- **Outside the brain:** PDE-constrained shape registration; energy = shape distance + elastic energy with growth. — Pawar et al., arXiv 2204.02451 — [arXiv](https://www.alphaxiv.org/abs/2204.02451)

### Inferences
- Hausdorff distance and vertex-wise distance are almost absent from the folding literature. Point-to-point comparisons of fold positions are reported qualitatively (landmark sulci by hand) or against an average brain. Our point-wise correspondence with the dHCP atlas (same topology across weeks) is therefore an unusual feature that should be stressed.
- Recommendation for the paper: report the classic metrics (GI, sulcal depth, SpAnGy, shape index) *alongside* our loss, so we are comparable with Hou 2026 and FetalFoldSim.

### Gaps
- No systematic review found specifically on simulation-vs-MRI comparison metrics for folding.

## Q3. Has anyone used an adjoint or AD through a folding equilibrium, or fitted spatial growth fields to reproduce specific sulci?

### Takeaway
**Nothing found** for an adjoint or AD through the equilibrium of a *folded* configuration, and nothing for spatial growth fields fitted to reproduce specific sulci. The only adjoint on a fetal brain (Wang/Garikipati 2021) stops before folding. The available differentiable approaches (JAX-FEM, PINNs, surrogates) have not been applied to cortical folding.

### Cited findings
- Adjoint (dolfin-adjoint) through quasi-static morphoelastic equilibrium, at 21–25 GW, where growth is "mostly homogeneous early". The study is limited to 1-week intervals because of registration when new structures appear. — [PMC8186493](https://pmc.ncbi.nlm.nih.gov/articles/PMC8186493/)
- Searches for "cortical folding / brain folding" + "automatic differentiation / dolfin-adjoint / JAX / PINN" returned no work combining AD or adjoint with folding fitted to MRI surfaces. — searches of 23 Oct 2026; leading results [Nat Commun 2026](https://www.nature.com/articles/s41467-026-75582-9), [arXiv 2607.08534](https://arxiv.org/html/2607.08534), [arXiv 2607.21840](https://arxiv.org/pdf/2607.21840)
- The Research Square preprint of Nat Commun 2026 mentions PINNs among ML approaches to morphogenesis and suggests "reverse inference" as a perspective, but implements neither. — (search summary; [Research Square](https://www.researchsquare.com/article/rs-8033969/v1); exact wording **not verified** in full text)
- Regional growth heterogeneity is *prescribed* from morphometry and yields "key anatomical landmarks", but is not inverted to reproduce them. — [PMC13490475](https://pmc.ncbi.nlm.nih.gov/articles/PMC13490475/)
- FetalFoldSim identifies spatial heterogeneity (thickness, stiffness ratio, growth) as necessary future work, with constant parameters for now. — [arXiv 2607.08534](https://arxiv.org/abs/2607.08534)
- A June 2026 comparison of adjoint vs PINN for PDE-constrained inverse problems in mechanics exists, but is generic and not applied to the brain. — [arXiv 2606.12337](https://www.alphaxiv.org/abs/2606.12337)
- Neighbouring differentiable mechanics in brain/soft tissue:
  - differentiable solver for tumour mass effect (TUM, Oct 2026) — [arXiv 2610.06097](https://www.alphaxiv.org/abs/2610.06097)
  - JAX-FEM for soft tissue (breast) — [arXiv 2609.18109](https://www.alphaxiv.org/abs/2609.18109)
- Developmental CORTET (Robinson group, KCL) generates simulation-ready tetrahedral meshes of the fetal cortex (about 200 subjects) and shows a stable morphoelastic folding simulation on a real subject. The abstract describes no fitting. — [arXiv 2607.12157](https://arxiv.org/abs/2607.12157)

### Inferences
- **Novelty, defensible:** within the limits of these searches, we would be the first to:
  - (1) differentiate exactly (adjoint) *through the folded equilibrium* of a Tallinen-type model;
  - (2) identify spatially resolved fields (per vertex) of tangential growth, thickening and core growth that reproduce the *specific geometry* of sulci (central sulcus, calcarine) in the dHCP atlas;
  - (3) reduce these fields to a compact law (6 coefficients), identified by the same adjoint and transferable between regions.
- **Prior art to cite as direct forerunners:**
  - Wang/Garikipati 2021 (adjoint, pre-folding, volume);
  - Hou et al. Nat Commun 2026 (realistic regional growth, forward model);
  - FetalFoldSim 2026 (dHCP atlas, scalar calibration by spectral metric);
  - Garcia et al. PNAS 2018 (measured expansion fields).
- **Risk:** the CORTET group (KCL, dHCP data) and the Garikipati/Weickenmeier groups are well placed to publish an inverse version soon. Monitor arXiv.

### Gaps
- Not fully checked: MICCAI/PIPPI proceedings 2023–2026 and *J Mech Phys Solids* / *CMAME* / *Acta Biomaterialia* for possible inverse folding work in 2D or on patches (e.g. inverse buckling, wavelength estimation). No search was dedicated to "inverse buckling wrinkling growth identification" in general.
- The full texts of Nat Commun 2026 (last 34k characters) and Wang 2021 (last 25k characters) were not read entirely. The key elements come from the methods and results sections.
- The Research Square wording on PINNs and "reverse inference" comes from a search snippet and is not verified.
