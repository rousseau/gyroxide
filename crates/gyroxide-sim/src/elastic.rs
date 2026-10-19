//! Tetrahedral elasticity of Tallinen et al. (2016), reference tensor
//! implementation. Line-by-line port of the "Deformations" loop of
//! Brains.cpp, vectorised over tetrahedra.

use crate::algebra::{M3, T1, V3};
use crate::setup::Precomputed;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use gyroxide_model::Material;

/// What one evaluation of the elastic forces produces.
pub struct ElasticOutput<B: Backend> {
    /// Assembled nodal forces [N].
    pub force: V3<B>,
    /// Grown reference nodal volumes Vn0 [N] (also the lumped masses / rho).
    pub vn0: T1<B>,
    /// Deformed nodal volumes Vn [N].
    pub vn: T1<B>,
    /// Cortical indicator g_m per tetrahedron [T].
    pub gm: T1<B>,
    /// Elastic energy and number of inverted elements, when requested.
    pub energy: Option<(T1<B>, T1<B>)>,
}

/// Sums per-vertex tetrahedral contributions into nodes through the ELL
/// incidence table (gather, deterministic, no atomics).
pub fn gather_nodes<B: Backend>(pre: &Precomputed<B>, per_vertex: [T1<B>; 4]) -> T1<B> {
    let device = per_vertex[0].device();
    let mut parts: Vec<T1<B>> = per_vertex.into();
    parts.push(T1::<B>::zeros([1], &device)); // padding slot
    let flat = T1::cat(parts, 0).select(0, pre.ell_tet.clone());
    let n = pre.n_nodes;
    flat.reshape([n, pre.ell_tet_width]).sum_dim(1).reshape([n])
}

fn gather_v3<B: Backend>(pre: &Precomputed<B>, f: [V3<B>; 4]) -> V3<B> {
    let [a, b, c, d] = f;
    V3::new(
        gather_nodes(pre, [a.x, b.x, c.x, d.x]),
        gather_nodes(pre, [a.y, b.y, c.y, d.y]),
        gather_nodes(pre, [a.z, b.z, c.z, d.z]),
    )
}

/// Nodal forces of the first Piola-Kirchhoff stress `p` on the grown reference
/// edges `ar`: P (N_a + N_b + N_c) / 6 per vertex, then gathered.
fn nodal_forces<B: Backend>(pre: &Precomputed<B>, ar: &M3<B>, p: &M3<B>) -> V3<B> {
    // Undeformed (grown) face normals.
    let (xr1, xr2, xr3) = (ar.col(0), ar.col(1), ar.col(2));
    let n1 = xr3.cross(&xr1);
    let n2 = xr2.cross(&xr3);
    let n3 = xr1.cross(&xr2);
    let n4 = xr2.sub(&xr3).cross(&xr1.sub(&xr3));
    let sixth = 1.0 / 6.0;
    let f_local = [
        p.mul_vec(&n1.add(&n2).add(&n3)).mul_scalar(sixth),
        p.mul_vec(&n1.add(&n3).add(&n4)).mul_scalar(sixth),
        p.mul_vec(&n2.add(&n3).add(&n4)).mul_scalar(sixth),
        p.mul_vec(&n1.add(&n2).add(&n4)).mul_scalar(sixth),
    ];
    gather_v3(pre, f_local)
}

/// Thickness H of the growing layer: one value, or one per tetrahedron.
pub enum Thickness<B: Backend> {
    Uniform(f64),
    PerTet(T1<B>),
}

/// Elastic forces for positions `x`, given the growth stretch `growth_prev`
/// (= g_m a_t of the previous step, as in the C code where G is updated at
/// the end of the loop and used at the next step) and thickness `h`.
pub fn forces<B: Backend>(
    pre: &Precomputed<B>,
    mat: &Material,
    x: &V3<B>,
    growth_prev: &T1<B>,
    thickness: f64,
    with_energy: bool,
) -> ElasticOutput<B> {
    forces_with(pre, mat, x, growth_prev, &Thickness::Uniform(thickness), with_energy)
}

/// [`forces`] with a uniform or per-tetrahedron thickness.
pub fn forces_with<B: Backend>(
    pre: &Precomputed<B>,
    mat: &Material,
    x: &V3<B>,
    growth_prev: &T1<B>,
    thickness: &Thickness<B>,
    with_energy: bool,
) -> ElasticOutput<B> {
    forces_impl(pre, mat, x, growth_prev, None, None, thickness, with_energy, false)
}

/// [`forces_with`] with an anisotropic growth tensor T per tetrahedron:
/// G = I + g T, with `g` (= g_m of the previous step) in `growth_prev`.
pub fn forces_tensor<B: Backend>(
    pre: &Precomputed<B>,
    mat: &Material,
    x: &V3<B>,
    growth_prev: &T1<B>,
    tensor: &M3<B>,
    thickness: &Thickness<B>,
    with_energy: bool,
) -> ElasticOutput<B> {
    forces_impl(pre, mat, x, growth_prev, Some(tensor), None, thickness, with_energy, false)
}

/// Nodal forces from the regular (neo-Hookean) branch only, for every
/// tetrahedron. Equal to [`forces_with`] when no element is inverted or below
/// the stretch floor; meant for reverse-mode differentiation, where the
/// unselected inverted branch would otherwise propagate NaN (0 * NaN).
pub fn forces_regular<B: Backend>(
    pre: &Precomputed<B>,
    mat: &Material,
    x: &V3<B>,
    growth: &T1<B>,
    thickness: &Thickness<B>,
    core: Option<&M3<B>>,
) -> V3<B> {
    forces_impl(pre, mat, x, growth, None, core, thickness, false, true).force
}

/// Elastic forces with every growth term: tangential growth (scalar `growth_prev` along
/// I - n n^T, or the anisotropic `tensor`) and the growth tensor U of the core,
/// G = I + g_m a (I - n n^T) + (1 - g_m)(U - I).
#[allow(clippy::too_many_arguments)]
pub fn forces_full<B: Backend>(
    pre: &Precomputed<B>,
    mat: &Material,
    x: &V3<B>,
    growth_prev: &T1<B>,
    tensor: Option<&M3<B>>,
    core: Option<&M3<B>>,
    thickness: &Thickness<B>,
    with_energy: bool,
) -> ElasticOutput<B> {
    forces_impl(pre, mat, x, growth_prev, tensor, core, thickness, with_energy, false)
}

/// Directional derivative dF_reg/dx . v of the regular-branch elastic force (forward mode,
/// exact): K v = -force_tangent(..). Same growth and thickness conventions as
/// [`forces_regular`]; elements outside the regular branch use its formula too.
pub fn force_tangent<B: Backend>(
    pre: &Precomputed<B>,
    mat: &Material,
    x: &V3<B>,
    v: &V3<B>,
    growth: &T1<B>,
    thickness: &Thickness<B>,
    core: Option<&M3<B>>,
) -> V3<B> {
    let nt = pre.n_tets;
    let device = x.x.device();
    let frob = |a: &M3<B>, b: &M3<B>| -> T1<B> {
        (0..3)
            .flat_map(|r| (0..3).map(move |c| (r, c)))
            .map(|(r, c)| a.get(r, c) * b.get(r, c))
            .reduce(|s, t| s + t)
            .unwrap()
    };
    let d_over_h = match thickness {
        Thickness::Uniform(h) => pre.d_mean.clone() / *h,
        Thickness::PerTet(h) => pre.d_mean.clone() / h.clone(),
    };
    let gm = pre.gr_mean.clone() / ((d_over_h - 1.0) * 10.0).exp().add_scalar(1.0);
    let mu = (gm.clone() * (mat.mu_gray - mat.mu_white)).add_scalar(mat.mu_white);
    let n = &pre.growth_normal;
    let nn = M3::from_cols(&n.mul(&n.x), &n.mul(&n.y), &n.mul(&n.z));
    let eye = M3::<B>::identity(nt, &device);
    let mut g = eye.add(&eye.sub(&nn).mul_s(growth));
    if mat.wm_growth != 0.0 {
        g = g.add(&eye.mul_s(&((-gm.clone()).add_scalar(1.0) * mat.wm_growth)));
    }
    if let Some(u) = core {
        g = g.add(&u.sub(&eye).mul_s(&(-gm).add_scalar(1.0)));
    }
    let ar = g.matmul(&pre.ar0);

    let xs: [V3<B>; 4] = std::array::from_fn(|k| x.select(&pre.tet_nodes[k]));
    let vs: [V3<B>; 4] = std::array::from_fn(|k| v.select(&pre.tet_nodes[k]));
    let a = M3::from_cols(&xs[1].sub(&xs[0]), &xs[2].sub(&xs[0]), &xs[3].sub(&xs[0]));
    let da = M3::from_cols(&vs[1].sub(&vs[0]), &vs[2].sub(&vs[0]), &vs[3].sub(&vs[0]));

    // Nodal-averaged J and its derivative (d det A = adj(A)^T : dA).
    let q0 = ar.det() / 24.0;
    let q = a.det() / 24.0;
    let dq = frob(&a.adjugate().transpose(), &da) / 24.0;
    let vn0 = gather_nodes(pre, [q0.clone(), q0.clone(), q0.clone(), q0]);
    let vn = gather_nodes(pre, [q.clone(), q.clone(), q.clone(), q]);
    let dvn = gather_nodes(pre, [dq.clone(), dq.clone(), dq.clone(), dq]);
    let jn = vn / vn0.clone();
    let djn = dvn / vn0;
    let mean4 = |t: &T1<B>| {
        let s: [T1<B>; 4] = std::array::from_fn(|k| t.clone().select(0, pre.tet_nodes[k].clone()));
        (s[0].clone() + s[1].clone() + s[2].clone() + s[3].clone()) / 4.0
    };
    let kja = (mean4(&jn) - 1.0) * mat.bulk;
    let dkja = mean4(&djn) * mat.bulk;

    let ari = ar.inverse();
    let f = a.matmul(&ari);
    let df = da.matmul(&ari);
    let j = f.det();
    let adjf = f.adjugate();
    let dj = frob(&adjf.transpose(), &df);
    let b = f.matmul(&f.transpose());
    let db = df.matmul(&f.transpose()).add(&f.matmul(&df.transpose()));
    let coef = mu / (j.clone() * j.clone().powf_scalar(2.0 / 3.0));
    let dcoef = coef.clone() * dj.clone() / j.clone() * (-5.0 / 3.0);
    let dev_b = b.sub(&eye.mul_s(&(b.trace() / 3.0)));
    let ddev_b = db.sub(&eye.mul_s(&(db.trace() / 3.0)));
    let sigma = dev_b.mul_s(&coef).add(&eye.mul_s(&kja));
    let dsigma = dev_b.mul_s(&dcoef).add(&ddev_b.mul_s(&coef)).add(&eye.mul_s(&dkja));
    // d adj(F) = adj(F) tr(F^-1 dF) - F^-1 dF adj(F), F^-1 = adj(F) / J.
    let finv = adjf.mul_s(&j.clone().recip());
    let dadj = adjf.mul_s(&(dj / j)).sub(&finv.matmul(&df).matmul(&adjf));
    let dp = dsigma.matmul(&adjf.transpose()).add(&sigma.matmul(&dadj.transpose()));
    nodal_forces(pre, &ar, &dp)
}

fn forces_impl<B: Backend>(
    pre: &Precomputed<B>,
    mat: &Material,
    x: &V3<B>,
    growth_prev: &T1<B>,
    tensor: Option<&M3<B>>,
    core: Option<&M3<B>>,
    thickness: &Thickness<B>,
    with_energy: bool,
    regular_only: bool,
) -> ElasticOutput<B> {
    let nt = pre.n_tets;
    let device = x.x.device();

    // Grey / white matter indicator and shear modulus.
    let d_over_h = match thickness {
        Thickness::Uniform(h) => pre.d_mean.clone() / *h,
        Thickness::PerTet(h) => pre.d_mean.clone() / h.clone(),
    };
    let gm = pre.gr_mean.clone() / ((d_over_h - 1.0) * 10.0).exp().add_scalar(1.0);
    let mu = (gm.clone() * (mat.mu_gray - mat.mu_white)).add_scalar(mat.mu_white);

    // Growth tensor G = I + s (I - n n^T), applied to the reference edges.
    let n = &pre.growth_normal;
    let nn = M3::from_cols(&n.mul(&n.x), &n.mul(&n.y), &n.mul(&n.z));
    let eye = M3::<B>::identity(nt, &device);
    let mut g = match tensor {
        None => eye.add(&eye.sub(&nn).mul_s(growth_prev)),
        Some(t) => eye.add(&t.mul_s(growth_prev)),
    };
    if mat.wm_growth != 0.0 {
        g = g.add(&eye.mul_s(&((-gm.clone()).add_scalar(1.0) * mat.wm_growth)));
    }
    if let Some(u) = core {
        g = g.add(&u.sub(&eye).mul_s(&(-gm.clone()).add_scalar(1.0)));
    }
    let ar = g.matmul(&pre.ar0);

    // Current edges.
    let xs: [V3<B>; 4] = std::array::from_fn(|k| x.select(&pre.tet_nodes[k]));
    let a = M3::from_cols(&xs[1].sub(&xs[0]), &xs[2].sub(&xs[0]), &xs[3].sub(&xs[0]));

    // Nodal volumes (reference grown and deformed) for the averaged J.
    let vol0 = ar.det() / 6.0;
    let vol = a.det() / 6.0;
    let q0 = vol0 / 4.0;
    let q = vol.clone() / 4.0;
    let vn0 = gather_nodes(pre, [q0.clone(), q0.clone(), q0.clone(), q0]);
    let vn = gather_nodes(pre, [q.clone(), q.clone(), q.clone(), q]);
    let jn = vn.clone() / vn0.clone();
    let js: [T1<B>; 4] = std::array::from_fn(|k| jn.clone().select(0, pre.tet_nodes[k].clone()));
    let ja = (js[0].clone() + js[1].clone() + js[2].clone() + js[3].clone()) / 4.0;
    let kja = (ja.clone() - 1.0) * mat.bulk; // K (Ja - 1)

    // Kinematics.
    let f = a.matmul(&ar.inverse());
    let j = f.det();
    let b = f.matmul(&f.transpose());
    let c = f.transpose().matmul(&f);
    let tr_b = b.trace();

    // --- Regular branch: sigma = mu/J^{5/3} dev(B) + K (Ja - 1) I,  P = J sigma F^{-T}.
    // J F^{-T} = adj(F)^T, so P = sigma' adj(F)^T without dividing by J again.
    let pow_j23 = j.clone().powf_scalar(2.0 / 3.0);
    let coef = mu.clone() / (j.clone() * pow_j23.clone());
    let dev_b = b.sub(&eye.mul_s(&(tr_b.clone() / 3.0)));
    let sigma = dev_b.mul_s(&coef).add(&eye.mul_s(&kja));
    let p_reg = sigma.matmul(&f.adjugate().transpose());

    if regular_only {
        let force = nodal_forces(pre, &ar, &p_reg);
        return ElasticOutput {
            force,
            vn0,
            vn,
            gm,
            energy: None,
        };
    }

    // --- Inverted / strongly compressed branch (diagonalised, stretch floor eps).
    let eig = sym_eigen(&c);
    let (p_inv, w_inv_dev, ninv_mask) = inverted_branch(&f, &j, &eig, &mu, &kja, mat);

    let lambda_min = eig.lambda_min.clone();
    let regular = lambda_min
        .greater_equal_elem(mat.eps * mat.eps)
        .bool_and(j.clone().greater_elem(0.0));
    let p = p_inv.where_(&regular, &p_reg);

    let force = nodal_forces(pre, &ar, &p);

    let energy = with_energy.then(|| {
        let vol_term = js
            .iter()
            .map(|ji| (ji.clone() - 1.0).powf_scalar(2.0))
            .reduce(|s, v| s + v)
            .unwrap()
            * (0.125 * mat.bulk); // 0.5 K mean((Ji - 1)^2)
        let w_reg = (tr_b / pow_j23 - 3.0) * mu.clone() * 0.5;
        let w = w_inv_dev.mask_where(regular.clone(), w_reg) + vol_term;
        let ok = (j.clone() * j.clone()).greater_elem(1e-50);
        let ue = (w * vol / j.clone()).mask_fill(ok.bool_not(), 0.0).sum();
        let ninv = ninv_mask.bool_and(regular.bool_not()).float().sum();
        (ue, ninv)
    });

    ElasticOutput {
        force,
        vn0,
        vn,
        gm,
        energy,
    }
}

/// Eigen-decomposition of a batch of symmetric 3x3 matrices.
pub struct SymEigen<B: Backend> {
    /// Eigenvalues in ascending order (Rayleigh quotients of `vectors`).
    pub values: [T1<B>; 3],
    /// Unit eigenvectors as columns, same order as `values`.
    pub vectors: M3<B>,
    /// Smallest eigenvalue from the closed form (`Matrix::EV().z` in vema.h),
    /// used for the branch test exactly as in the C code.
    pub lambda_min: T1<B>,
}

/// Closed-form eigenvalues (Kopp's trigonometric form, as `EV()` in vema.h)
/// plus robust eigenvectors: the best-separated eigenvalue's vector from the
/// largest cross product of the rows of (C - lambda I), the two others from
/// the 2x2 problem in its orthogonal complement (Eberly's construction).
pub fn sym_eigen<B: Backend>(c: &M3<B>) -> SymEigen<B> {
    let (a, b, cc) = (c.get(0, 0), c.get(0, 1), c.get(0, 2));
    let (e, f, i) = (c.get(1, 1), c.get(1, 2), c.get(2, 2));
    let tr = a.clone() + e.clone() + i.clone();
    let c1 = a.clone() * e.clone() + a.clone() * i.clone() + e.clone() * i.clone()
        - b.clone() * b.clone()
        - f.clone() * f.clone()
        - cc.clone() * cc.clone();
    let c0 =
        i.clone() * b.clone() * b.clone() + a.clone() * f.clone() * f.clone() + e.clone() * cc.clone() * cc.clone()
            - a.clone() * e.clone() * i.clone()
            - cc.clone() * b.clone() * f.clone() * 2.0;
    let p = tr.clone() * tr.clone() - c1.clone() * 3.0;
    let q = tr.clone() * (p.clone() - c1.clone() * 1.5) - c0.clone() * 13.5;
    let phi = ((c1.clone() * c1.clone() * 0.25 * (p.clone() - c1.clone())
        + c0.clone() * (q.clone() + c0.clone() * 6.75))
        * 27.0)
        .abs()
        .sqrt()
        .atan2(q)
        / 3.0;
    let sp = p.abs().sqrt();
    let t = sp.clone() * phi.clone().cos();
    let s = sp * phi.sin() / 3f64.sqrt();
    let l_lo = (tr - t.clone()) / 3.0 - s.clone();
    let l_mid = l_lo.clone() + s.clone() * 2.0;
    let l_hi = l_lo.clone() + t + s;

    // Which end of the spectrum is better separated.
    let low_first = (l_mid.clone() - l_lo.clone()).greater_equal(l_hi.clone() - l_mid);
    let lam = l_hi.clone().mask_where(low_first.clone(), l_lo.clone());
    let v = eigvec_from_rows(c, &lam);

    // Orthonormal complement {u, w} of v.
    let x_dominant = v.x.clone().abs().greater(v.y.clone().abs());
    let zero = v.x.zeros_like();
    let u_a = V3::new(-v.z.clone(), zero.clone(), v.x.clone());
    let u_b = V3::new(zero, v.z.clone(), -v.y.clone());
    let u = u_b.where_(&x_dominant, &u_a).normalized();
    let w = v.cross(&u);

    // 2x2 problem.
    let cu = c.mul_vec(&u);
    let cw = c.mul_vec(&w);
    let m00 = u.dot(&cu);
    let m01 = u.dot(&cw);
    let m11 = w.dot(&cw);
    let theta = (m01 * 2.0).atan2(m00 - m11) / 2.0;
    let (ct, st) = (theta.clone().cos(), theta.sin());
    let ea = u.mul(&ct).add(&w.mul(&st));
    let eb = w.mul(&ct).sub(&u.mul(&st));
    let ra = ea.dot(&c.mul_vec(&ea));
    let rb = eb.dot(&c.mul_vec(&eb));
    let a_lower = ra.clone().lower(rb.clone());
    let lo2 = eb.where_(&a_lower, &ea);
    let hi2 = ea.where_(&a_lower, &eb);

    // Ascending columns: [v, lo2, hi2] if the low end was solved first, else [lo2, hi2, v].
    let c0_ = lo2.where_(&low_first, &v);
    let c1_ = hi2.where_(&low_first, &lo2);
    let c2_ = v.where_(&low_first, &hi2);
    let vectors = M3::from_cols(&c0_, &c1_, &c2_);
    let values = std::array::from_fn(|k| {
        let col = vectors.col(k);
        col.dot(&c.mul_vec(&col))
    });
    SymEigen {
        values,
        vectors,
        lambda_min: l_lo,
    }
}

/// Unit eigenvector of symmetric `c` for eigenvalue `lam`: the largest of the
/// cross products of the rows of (C - lam I); e_x if all vanish.
fn eigvec_from_rows<B: Backend>(c: &M3<B>, lam: &T1<B>) -> V3<B> {
    let row = |r: usize| {
        let mut v = [c.get(r, 0), c.get(r, 1), c.get(r, 2)];
        v[r] = v[r].clone() - lam.clone();
        V3::new(v[0].clone(), v[1].clone(), v[2].clone())
    };
    let (r0, r1, r2) = (row(0), row(1), row(2));
    let x01 = r0.cross(&r1);
    let x02 = r0.cross(&r2);
    let x12 = r1.cross(&r2);
    let (d01, d02, d12) = (x01.dot(&x01), x02.dot(&x02), x12.dot(&x12));
    let pick02 = d02.clone().greater(d01.clone());
    let best = x01.where_(&pick02, &x02);
    let dbest = d01.mask_where(pick02, d02);
    let pick12 = d12.clone().greater(dbest.clone());
    let best = best.where_(&pick12, &x12);
    let dbest = dbest.mask_where(pick12, d12);
    let degenerate = dbest.lower_equal_elem(0.0);
    let ex = V3::new(best.x.ones_like(), best.x.zeros_like(), best.x.zeros_like());
    best.normalized().where_(&degenerate, &ex)
}

/// Port of the "Needs SVD" branch of Brains.cpp. Returns P, the deviatoric +
/// floor part of W (the volumetric term is added by the caller), and the
/// mask of truly inverted elements (det F < 0).
pub fn inverted_branch<B: Backend>(
    f: &M3<B>,
    j: &T1<B>,
    eig: &SymEigen<B>,
    mu: &T1<B>,
    kja: &T1<B>,
    mat: &Material,
) -> (M3<B>, T1<B>, Tensor<B, 1, burn::tensor::Bool>) {
    let eps = mat.eps;
    let tiny = 1e-25;
    let l: [T1<B>; 3] = std::array::from_fn(|k| eig.values[k].clone().clamp_min(0.0).sqrt());

    // Right-handed eigenvector basis.
    let mut v = eig.vectors.clone();
    let flip_v = v.det().lower_elem(0.0);
    v.set_col(0, &v.col(0).where_(&flip_v, &v.col(0).neg()));

    let l1_small = l[0].clone().lower_elem(tiny);
    let inv_l1 = l[0].clone().recip().mask_fill(l1_small.clone(), 0.0);
    let mut u = f
        .matmul(&v)
        .scale_cols([&inv_l1, &l[1].clone().recip(), &l[2].clone().recip()]);
    let u_c0 = u.col(1).cross(&u.col(2));
    u.set_col(0, &u.col(0).where_(&l1_small, &u_c0));

    let inverted = j.clone().lower_elem(0.0);
    let l1 = l[0].clone().mask_where(inverted.clone(), -l[0].clone());
    u.set_col(0, &u.col(0).where_(&inverted, &u.col(0).neg()));

    let (l2, l3) = (l[1].clone(), l[2].clone());
    let l2s = l2.clone() * l2.clone();
    let l3s = l3.clone() * l3.clone();
    let pow23 = ((l2.clone() * l3.clone()) * eps).powf_scalar(2.0 / 3.0);
    let m3 = mu.clone() / pow23 / 3.0; // mu / (3 pow23)
    let m9 = m3.clone() / 3.0; // mu / (9 pow23)
    let dl1 = l1.clone() - eps;
    let pa = m3.clone() * (-(l2s.clone() + l3s.clone()) / eps + 2.0 * eps)
        + dl1.clone() * mat.k_inverted
        + kja.clone() * l2.clone() * l3.clone();
    let pe = m3.clone() * ((-l3s.clone() - eps * eps) / l2.clone() + l2.clone() * 2.0)
        + m9.clone() * ((l3s.clone() * (2.0 / eps) - 4.0 * eps) / l2.clone() - l2.clone() * (4.0 / eps)) * dl1.clone()
        + kja.clone() * l1.clone() * l3.clone();
    let pi = m3.clone() * ((-l2s.clone() - eps * eps) / l3.clone() + l3.clone() * 2.0)
        + m9 * ((l2s.clone() * (2.0 / eps) - 4.0 * eps) / l3.clone() - l3.clone() * (4.0 / eps)) * dl1.clone()
        + kja.clone() * l1 * l2;
    let w = (m3.clone() * 1.5 * (l2s.clone() + l3s.clone()).add_scalar(eps * eps)) - mu.clone() * 1.5
        + m3 * (-(l2s + l3s) / eps + 2.0 * eps) * dl1.clone()
        + dl1.clone() * dl1 * (0.5 * mat.k_inverted);
    let p = u.scale_cols([&pa, &pe, &pi]).matmul(&v.transpose());
    (p, w, inverted)
}
