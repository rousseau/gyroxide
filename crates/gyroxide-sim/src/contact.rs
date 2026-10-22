//! Self-contact: penalty between surface points and surface triangles
//! (Tallinen et al. 2016), plus the optional repulsive midplane.
//!
//! Version 1 follows the plan of the design note: candidate pairs are built
//! on the CPU with a linked-cell grid (positions read back only at that
//! moment), uploaded, and the penalty forces are evaluated on the device
//! every step and gathered to the nodes through an ELL table built together
//! with the pairs. A rebuild is triggered when a surface node has moved by
//! more than half the gap between proximity and repulsive skins.

use crate::algebra::{floats, ints, I1, T1, V3};
use burn::tensor::backend::Backend;
use burn::tensor::Bool;
use burn::tensor::Tensor;
use gyroxide_mesh::topology::Ell;
use gyroxide_mesh::{add, dot, norm, scale, sub, TetMesh, Vec3};
use std::collections::HashMap;

/// Candidate (surface point, triangle) pairs on the device.
pub struct ContactPairs<B: Backend> {
    pub n_pairs: usize,
    pub point: I1<B>,
    pub tri: [I1<B>; 3],
    /// 1.0 where `gr[pt] + gr[tri.n1] > 0` (the C code's condition), else 0.
    pub active: T1<B>,
    /// Node <- (role, pair) incidence, slot = role * n_pairs + pair,
    /// role 0 = point, 1..3 = triangle vertices.
    pub ell: I1<B>,
    pub ell_width: usize,
}

/// Builds the candidate lists exactly like `createNNLtriangle`, except
/// that the cell grid is unbounded (hash map) instead of a fixed box.
pub fn build_pairs(mesh: &TetMesh, surface_nodes: &[u32], x: &[Vec3], cell: f64, skin: f64) -> Vec<(u32, u32)> {
    let key = |p: Vec3| -> [i64; 3] { std::array::from_fn(|k| (p[k] / cell).floor() as i64) };
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for (i, f) in mesh.faces.iter().enumerate() {
        let c = scale(
            add(add(x[f[0] as usize], x[f[1] as usize]), x[f[2] as usize]),
            1.0 / 3.0,
        );
        grid.entry(key(c)).or_default().push(i as u32);
    }
    let mut pairs = Vec::new();
    for &pt in surface_nodes {
        let p = x[pt as usize];
        let c = key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(list) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                        continue;
                    };
                    for &tri in list {
                        let f = mesh.faces[tri as usize];
                        if f.contains(&pt) {
                            continue;
                        }
                        let (q, _) = closest_point_triangle(p, x[f[0] as usize], x[f[1] as usize], x[f[2] as usize]);
                        if norm(sub(q, p)) < skin {
                            pairs.push((pt, tri));
                        }
                    }
                }
            }
        }
    }
    pairs
}

pub fn upload_pairs<B: Backend>(
    mesh: &TetMesh,
    gr: &[f64],
    pairs: &[(u32, u32)],
    device: &B::Device,
) -> Option<ContactPairs<B>> {
    if pairs.is_empty() {
        return None;
    }
    let np = pairs.len();
    let point: Vec<u32> = pairs.iter().map(|p| p.0).collect();
    let tri: [Vec<u32>; 3] = std::array::from_fn(|k| pairs.iter().map(|p| mesh.faces[p.1 as usize][k]).collect());
    let active: Vec<f64> = pairs
        .iter()
        .map(|&(pt, t)| (gr[pt as usize] + gr[mesh.faces[t as usize][0] as usize] > 0.0) as u8 as f64)
        .collect();
    let incidence = (0..np).flat_map(|i| {
        let (pt, t) = pairs[i];
        let f = mesh.faces[t as usize];
        [
            (pt, i as u32),
            (f[0], (np + i) as u32),
            (f[1], (2 * np + i) as u32),
            (f[2], (3 * np + i) as u32),
        ]
    });
    let ell = Ell::from_pairs(mesh.n_nodes(), 4 * np, incidence.collect::<Vec<_>>().into_iter());
    Some(ContactPairs {
        n_pairs: np,
        point: ints(&point, device),
        tri: std::array::from_fn(|k| ints(&tri[k], device)),
        active: floats(&active, device),
        ell: ints(&ell.idx, device),
        ell_width: ell.width,
    })
}

/// Penalty forces on all nodes [N] for the current positions.
pub fn penalty_forces<B: Backend>(
    pairs: &ContactPairs<B>,
    x: &V3<B>,
    n_nodes: usize,
    hc: f64,
    kc: f64,
    a: f64,
) -> V3<B> {
    let p = x.select(&pairs.point);
    let ta = x.select(&pairs.tri[0]);
    let tb = x.select(&pairs.tri[1]);
    let tc = x.select(&pairs.tri[2]);
    let (q, [u, v, w]) = closest_point_triangle_t(&p, &ta, &tb, &tc);
    let cc = q.sub(&p);
    let rc = cc.norm();
    let within = rc.clone().lower_elem(hc);
    let gate = pairs.active.clone().mask_fill(within.bool_not(), 0.0);
    let ntri = tb.sub(&ta).cross(&tc.sub(&ta)).normalized();
    let fn_ = cc.normalized().mul(&((rc - hc) * (kc * a * a / hc)));
    let fdn = fn_.dot(&ntri);
    let reflected = fn_.sub(&ntri.mul(&(fdn.clone() * 2.0)));
    let fn_ = fn_.where_(&fdn.lower_elem(0.0), &reflected).mul(&gate);
    let device = x.x.device();
    let mut out = [vec![], vec![], vec![]];
    let parts = [fn_.clone(), fn_.mul(&u).neg(), fn_.mul(&v).neg(), fn_.mul(&w).neg()];
    for part in parts {
        out[0].push(part.x);
        out[1].push(part.y);
        out[2].push(part.z);
    }
    let gather = |mut cols: Vec<T1<B>>| -> T1<B> {
        cols.push(T1::<B>::zeros([1], &device));
        T1::cat(cols, 0)
            .select(0, pairs.ell.clone())
            .reshape([n_nodes, pairs.ell_width])
            .sum_dim(1)
            .reshape([n_nodes])
    };
    let [ox, oy, oz] = out;
    V3::new(gather(ox), gather(oy), gather(oz))
}

/// Midplane force (y component only) on surface nodes that started on one
/// side and are now on the other. `restoring = false` is Brains.cpp's
/// `Ft.y -= (mpy - y)/hc * a^2 K`, which points away from the plane.
#[allow(clippy::too_many_arguments)]
pub fn midplane_force<B: Backend>(
    y: &T1<B>,
    low: &T1<B>,
    high: &T1<B>,
    mpy: f64,
    hc: f64,
    k: f64,
    a: f64,
    restoring: bool,
) -> T1<B> {
    let above = y.clone().greater_elem(mpy).float();
    let below = y.clone().lower_elem(mpy).float();
    let on = low.clone() * above + high.clone() * below;
    let sign = if restoring { -1.0 } else { 1.0 };
    (y.clone().sub_scalar(mpy) * (sign * a * a * k / hc)) * on
}

/// Ericson's closest point on triangle abc to p, with barycentric weights
/// (u, v, w) of (a, b, c). Same region order as Brains.cpp.
pub fn closest_point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f64; 3]) {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (add(a, scale(ab, v)), [1.0 - v, v, 0.0]);
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (add(a, scale(ac, w)), [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (add(b, scale(sub(c, b), w)), [0.0, 1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (add(add(a, scale(ab, v)), scale(ac, w)), [1.0 - v - w, v, w])
}

/// Batched version: all regions are evaluated and the first matching one
/// (in Ericson's order) is selected, by overriding in reverse priority.
fn closest_point_triangle_t<B: Backend>(p: &V3<B>, a: &V3<B>, b: &V3<B>, c: &V3<B>) -> (V3<B>, [T1<B>; 3]) {
    let ab = b.sub(a);
    let ac = c.sub(a);
    let ap = p.sub(a);
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    let bp = p.sub(b);
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    let cp = p.sub(c);
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    let vc = d1.clone() * d4.clone() - d3.clone() * d2.clone();
    let vb = d5.clone() * d2.clone() - d1.clone() * d6.clone();
    let va = d3.clone() * d6.clone() - d5.clone() * d4.clone();
    let one = d1.ones_like();
    let zero = d1.zeros_like();

    let and = |m: Vec<Tensor<B, 1, Bool>>| m.into_iter().reduce(|x, y| x.bool_and(y)).unwrap();

    // Interior (lowest priority).
    let denom = (va.clone() + vb.clone() + vc.clone()).recip();
    let v_in = vb.clone() * denom.clone();
    let w_in = vc.clone() * denom;
    let mut q = a.add(&ab.mul(&v_in)).add(&ac.mul(&w_in));
    let mut bary = [one.clone() - v_in.clone() - w_in.clone(), v_in, w_in];

    let mut apply = |mask: Tensor<B, 1, Bool>, qq: V3<B>, uvw: [T1<B>; 3]| {
        q = q.where_(&mask, &qq);
        for k in 0..3 {
            bary[k] = bary[k].clone().mask_where(mask.clone(), uvw[k].clone());
        }
    };
    // Edge bc.
    let e43 = d4.clone() - d3.clone();
    let e56 = d5.clone() - d6.clone();
    let w_bc = e43.clone() / (e43.clone() + e56.clone());
    apply(
        and(vec![
            va.lower_equal_elem(0.0),
            e43.greater_equal_elem(0.0),
            e56.greater_equal_elem(0.0),
        ]),
        b.add(&c.sub(b).mul(&w_bc)),
        [zero.clone(), one.clone() - w_bc.clone(), w_bc],
    );
    // Edge ac.
    let w_ac = d2.clone() / (d2.clone() - d6.clone());
    apply(
        and(vec![
            vb.lower_equal_elem(0.0),
            d2.clone().greater_equal_elem(0.0),
            d6.clone().lower_equal_elem(0.0),
        ]),
        a.add(&ac.mul(&w_ac)),
        [one.clone() - w_ac.clone(), zero.clone(), w_ac],
    );
    // Vertex c.
    apply(
        and(vec![d6.clone().greater_equal_elem(0.0), d5.clone().lower_equal(d6)]),
        c.clone(),
        [zero.clone(), zero.clone(), one.clone()],
    );
    // Edge ab.
    let v_ab = d1.clone() / (d1.clone() - d3.clone());
    apply(
        and(vec![
            vc.lower_equal_elem(0.0),
            d1.clone().greater_equal_elem(0.0),
            d3.clone().lower_equal_elem(0.0),
        ]),
        a.add(&ab.mul(&v_ab)),
        [one.clone() - v_ab.clone(), v_ab, zero.clone()],
    );
    // Vertex b.
    apply(
        and(vec![d3.clone().greater_equal_elem(0.0), d4.lower_equal(d3)]),
        b.clone(),
        [zero.clone(), one.clone(), zero.clone()],
    );
    // Vertex a (highest priority).
    apply(
        and(vec![d1.lower_equal_elem(0.0), d2.lower_equal_elem(0.0)]),
        a.clone(),
        [one, zero.clone(), zero],
    );
    (q, bary)
}

/// Number of active contact pairs: points closer to their triangle than the repulsive skin `hc`
/// (the pairs that [`penalty_forces`] pushes apart; one scalar read).
pub fn count_active<B: Backend>(pairs: &ContactPairs<B>, x: &V3<B>, hc: f64) -> usize {
    let p = x.select(&pairs.point);
    let ta = x.select(&pairs.tri[0]);
    let tb = x.select(&pairs.tri[1]);
    let tc = x.select(&pairs.tri[2]);
    let (q, _) = closest_point_triangle_t(&p, &ta, &tb, &tc);
    let within = q.sub(&p).norm().lower_elem(hc);
    let gate = pairs.active.clone().mask_fill(within.bool_not(), 0.0);
    crate::algebra::scalar(gate.sum()).round() as usize
}

/// Maximum displacement of the surface nodes since `x_old` (one scalar read).
pub fn max_surface_displacement<B: Backend>(x: &V3<B>, surface: &I1<B>, x_old: &V3<B>) -> f64 {
    crate::algebra::scalar(x.select(surface).sub(x_old).norm().max())
}
