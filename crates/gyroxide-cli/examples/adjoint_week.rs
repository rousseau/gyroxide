//! Per-vertex growth of one week fitted to the atlas by an adjoint at quasi-static equilibrium.
//!
//!     cargo run --release --features autodiff --example adjoint_week -- \
//!         <config.toml> <mesh> <checkpoint dir> <prep dir> <out dir> [--iters N] [--beta B]
//!         [--relax N] [--check] [--hp k | --chamfer] [--gd --step s]
//!         [--theta0-table] [--write-table a.npy] [--save-checkpoint dir] [--save-surface f.vtk]
//!         [--multiscale "k1:n1,k2:n2,..."] [--min-growth a] [--carry]
//!         [--alpha a --measured a_table.npy] [--fi-weight g]
//!         [--absolute | --rigid] [--mm-per-unit u] [--tol-mm t] [--pull-target t.npy --pull-k k] [--shift s] [--diag] [--fd-k]
//!
//! Multi-resolution (coarse to fine) loss, --multiscale "200:4,50:4,10:4,0:4": n1 L-BFGS
//! iterations on the shape loss of the smoothed surfaces S^k1 x vs S^k1 y (S: neighbour
//! averaging, k iterations ~ a Gaussian of width ~ sqrt(k) edges: 200 -> the lobes and the
//! primary folds, 10 -> the secondary folds), then n2 iterations at k2, etc.; 0 = the plain
//! vertex shape loss. The large folds are fitted first, finer detail is added level by level.
//! A level may carry a weight, "k:n:w". --cumulative keeps the coarser levels in the loss:
//! at level j the loss is sum over levels s <= j of w_s L_s / L_s(start) (each term normalised
//! by its value at the measured-growth equilibrium, so that all scales weigh alike).
//! --param-smooth f restricts the growth corrections at level k to smooth fields,
//! theta = theta_start_of_level + S^m z with m = round(f k) (gradient (S^T)^m dL/dtheta):
//! the parameters get the resolution of the data they are fitted to.
//! --min-growth a: projects theta onto theta >= a at every trial point (a = 0: no tangential
//! shrinkage). --max-step d (default 0.05): the step of an iteration changes theta by at most d;
//! trial points whose equilibrium has inverted elements are rejected (step halved).
//! --ftol r: stops when the loss decreased by less than r (relative) over the last 5 iterations
//! of a level. --carry (with --write-table): the correction theta - theta0 is also added to
//! the later rows of the table, so the next weeks grow from the corrected state instead of
//! jumping back to the measured growth.
//! --alpha a: keeps theta near the measured growth, + a/2 mean((theta - a_mes)^2) / mean(a_mes^2)
//! (a_mes: row of --measured at the checkpoint time, default theta0). --fi-weight g: + g/2
//! ((FI(x) - FI_atlas) / FI_atlas)^2, FI = A / |V|^(2/3) of the surface (prep/faces.npy), so that
//! the fit cannot gain by cutting growth (the folding index of the atlas is imposed).
//!
//! --absolute: the level losses compare the vertex positions themselves, in the common frame of
//! the simulation and the atlas (no centring, no scaling): size, position and folds must all
//! match. Deviation in mm (--mm-per-unit, default 32.99 = atlas maxd of w21_left_610k): at the
//! start, after every iteration and at the end, for every level k of --multiscale, RMS / p50 /
//! p95 / max of |S^k x_i - S^k y_i| over the cortex, absolute and shape-only (normalised surfaces
//! rescaled to the atlas rms radius); final values in <out>/deviation_mm.json. --tol-mm t stops
//! as soon as the absolute RMS deviation of the finest level is below t mm.
//! --rigid: like --absolute after the best rotation + translation of the smoothed surface onto
//! the smoothed target (no scaling: size, lobes and folds count, the placement of the
//! hemisphere in the atlas frame does not); --tol-mm then applies to the rigid deviation.
//! --highpass K: each level k < K compares the band S^k x - S^K x with S^k y - S^K y (plain L2,
//! translation-free): the lobar shape, which dominates the vertex loss, is left out.
//!
//! Week-by-week chaining (tools/adjoint_chain.sh): --theta0-table starts from the row of the
//! config's a table at the checkpoint time (instead of prep/theta0.npy); at the end
//! --write-table writes that table with the row replaced by the fitted theta (interior nodes
//! take their owner's value), --save-checkpoint the fitted equilibrium as a restart point of
//! `gyroxide run --resume` (v = 0), --save-surface its surface.
//!
//! Prep dir (tools/adjoint_prep.py): target.npy (atlas surface of the week, simulation frame),
//! edges.npy, owner.npy (surface vertex of each node), cortex.npy, theta0.npy (measured a).
//!
//! Parameters theta_v = a_t of surface vertex v (interior nodes take their owner's value);
//! growth of tetrahedron e: s_e = g_m,e * mean of theta over its 4 nodes.
//! Forward: hold s fixed, relax the damped dynamics to equilibrium x*(theta) (warm start), then
//! Newton with the exact K until |F| < max(1e-3 |F_0|, --force-tol) (default 1e-5, at most
//! --newton 20 steps); --relax0: damped steps of the first equilibrium (from the checkpoint,
//! where the explicit dynamics lags behind the growth), default --relax.
//! Loss: 1/(2N) sum over cortex vertices |x^_i - y^_i|^2, x^ = (x - centroid) / rms radius
//! (shape only: the atlas also grows in volume, the model does not), or with --hp k the fold
//! loss on high-pass positions x - S^k x (S: neighbour averaging on the surface); + beta/2
//! sum over edges of the squared difference of (theta - theta0) (smooth corrections only).
//! Gradient: K lambda = dL/dx (K = -dF/dx, CG with finite-difference products of the total
//! force), then dL/ds = lambda . dF/ds by one reverse-mode pass of the regular elastic
//! branch (Burn autodiff). Optimiser: L-BFGS with Armijo backtracking.
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Tensor, TensorData};
use gyroxide_mesh::{netgen, vtk};
use gyroxide_model::{NodalTable, Params, Schedule};
use gyroxide_sim::algebra::{to_f64, V3};
use gyroxide_sim::elastic::{forces_regular, Thickness};
use gyroxide_sim::Simulation;
use gyroxide_validate::npy;
use std::path::{Path, PathBuf};
use std::time::Instant;

type Fwd = Wgpu;
type Ad = Autodiff<Wgpu>;

struct Problem {
    sim: Simulation<Fwd>,
    ad: Simulation<Ad>,
    tets: Vec<[u32; 4]>,
    owner: Vec<usize>,
    gm: Vec<f64>,
    target_hat: Vec<[f64; 3]>,
    /// High-pass ("fold") loss: k smoothing iterations (0 = shape loss), surface
    /// neighbours, target detail y - S^k y, reference radius.
    hp: usize,
    chamfer: bool,
    /// Low-pass level of the multi-resolution loss (smoothing iterations, 0 = off) and the
    /// normalised smoothed target of that level.
    lp: usize,
    target_lp: Vec<[f64; 3]>,
    /// Cumulative multi-resolution loss: (level, weight, normalised smoothed target).
    active: Vec<(usize, f64, Vec<[f64; 3]>)>,
    target: Vec<[f64; 3]>,
    theta0: Vec<f64>,
    nbr: Vec<Vec<usize>>,
    target_detail: Vec<[f64; 3]>,
    r_ref: f64,
    cortex: Vec<usize>,
    edges: Vec<[usize; 2]>,
    beta: f64,
    /// Absolute loss (raw positions, no normalisation) and the length unit in mm.
    absolute: bool,
    /// Band-pass levels: compare S^k x - S^K x with K = --highpass (0 = off).
    highpass: usize,
    /// Rigid loss: absolute positions after the best rotation + translation (no scaling).
    rigid: bool,
    mm: f64,
    /// Pull towards the measured growth (relative mean square) and its scale mean(a_mes^2).
    alpha: f64,
    theta_mes: Vec<f64>,
    mes_scale: f64,
    /// Folding-index penalty: weight, atlas FI of the week, surface triangles.
    fi_weight: f64,
    fi_target: f64,
    faces: Vec<[usize; 3]>,
    relax: usize,
    thick: Vec<f64>,
    v0: Vec<[f64; 3]>,
    newton: usize,
    sigma: f64,
    last_residual: (f64, f64),
    /// Node colouring (no two nodes of a colour share a tetrahedron) and the inverted
    /// 3x3 diagonal blocks of K + sigma I (block-Jacobi preconditioner).
    colours: Vec<Vec<usize>>,
    pre_inv: Vec<[[f64; 3]; 3]>,
    /// Orthonormal vectors projected out of the CG solves (deflated unstable modes).
    deflate: Vec<Vec<[f64; 3]>>,
    /// Dirichlet nodes (sub-mesh cut): no loss gradient and no adjoint there.
    fixed: Vec<bool>,
    /// Geometric loss terms (face normals, Laplacian coordinates, face areas), if any.
    geo: Option<Geo>,
    /// Weight of the vertex-distance loss (--w-vertex).
    w_vertex: f64,
    /// Extra fitted fields per surface vertex after the growth a (--fit-fields c,k): correction of the
    /// thickening c and isotropic correction delta I of the core growth U; `na` = number of a values.
    fit_c: bool,
    fit_k: bool,
    na: usize,
    /// K products by finite differences of the total force (old path, --fd-k), and the
    /// growth / thickness tensors of the autodiff path.
    fd_k: bool,
    /// Absolute tolerance on |F| of the Newton iterations of the equilibrium.
    force_tol: f64,
}

fn colour_nodes(n: usize, tets: &[[u32; 4]]) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for t in tets {
        for &a in t {
            for &b in t {
                if a != b {
                    adj[a as usize].push(b);
                }
            }
        }
    }
    let mut colour = vec![usize::MAX; n];
    let mut used = Vec::new();
    for i in 0..n {
        used.clear();
        used.extend(adj[i].iter().map(|&j| colour[j as usize]).filter(|&c| c != usize::MAX));
        used.sort_unstable();
        used.dedup();
        let mut c = 0;
        for &u in &used {
            if u == c {
                c += 1;
            } else if u > c {
                break;
            }
        }
        colour[i] = c;
    }
    let nc = colour.iter().max().unwrap() + 1;
    let mut groups = vec![Vec::new(); nc];
    for (i, &c) in colour.iter().enumerate() {
        groups[c].push(i);
    }
    groups
}

fn inv3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !(det > 0.0) || m[0][0] <= 0.0 || m[1][1] <= 0.0 || m[2][2] <= 0.0 {
        return None;
    }
    let c = |a: usize, b: usize, d: usize, e: usize| m[a][b] * m[d][e] - m[a][e] * m[d][b];
    Some([
        [c(1, 1, 2, 2) / det, -c(0, 1, 2, 2) / det, c(0, 1, 1, 2) / det],
        [-c(1, 0, 2, 2) / det, c(0, 0, 2, 2) / det, -c(0, 0, 1, 2) / det],
        [c(1, 0, 2, 1) / det, -c(0, 0, 2, 1) / det, c(0, 0, 1, 1) / det],
    ])
}

/// Rotation R minimising sum |R x_i - y_i|^2 for centred point sets: orthogonal polar factor
/// of M = sum y_i x_i^T (Newton iteration X <- (X + X^-T) / 2).
fn best_rotation(x: &[[f64; 3]], y: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for (a, b) in x.iter().zip(y) {
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += b[r] * a[c];
            }
        }
    }
    let scale = m.iter().flatten().fold(0.0f64, |s, v| s.max(v.abs())).max(1e-300);
    let mut r = m.map(|row| row.map(|v| v / scale));
    for _ in 0..50 {
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        // X^-T = cofactor(X) / det.
        let cof = |i: usize, j: usize| {
            let (i1, i2, j1, j2) = ((i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3);
            r[i1][j1] * r[i2][j2] - r[i1][j2] * r[i2][j1]
        };
        let next: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (r[i][j] + cof(i, j) / det)));
        let diff = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).map(|(i, j)| (next[i][j] - r[i][j]).abs()).fold(0.0, f64::max);
        r = next;
        if diff < 1e-14 {
            break;
        }
    }
    r
}

fn centred(p: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let n = p.len() as f64;
    let c = [0, 1, 2].map(|d| p.iter().map(|v| v[d]).sum::<f64>() / n);
    p.iter().map(|v| [0, 1, 2].map(|d| v[d] - c[d])).collect()
}

fn read(p: &Path) -> Vec<f64> {
    npy::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).data
}

fn normalise(x: &[[f64; 3]], idx: &[usize]) -> (Vec<[f64; 3]>, [f64; 3], f64) {
    let n = idx.len() as f64;
    let mut c = [0.0; 3];
    for &i in idx {
        for k in 0..3 {
            c[k] += x[i][k] / n;
        }
    }
    let r = (idx
        .iter()
        .map(|&i| (0..3).map(|k| (x[i][k] - c[k]).powi(2)).sum::<f64>())
        .sum::<f64>()
        / n)
        .sqrt();
    (
        idx.iter()
            .map(|&i| [(x[i][0] - c[0]) / r, (x[i][1] - c[1]) / r, (x[i][2] - c[2]) / r])
            .collect(),
        c,
        r,
    )
}

/// Geometric loss terms on the faces / vertices of the loss region, each divided by its value
/// at the start: w_n mean_f A*_f (1 - u_f . nu_f) / mean A*, w_l mean_i |d(x)_i - d(y)_i|^2 / 2
/// (d: umbrella Laplacian coordinates x_i - mean of the neighbours), w_a mean_f log(A_f / A*_f)^2 / 2.
struct Geo {
    faces: Vec<[usize; 3]>,
    nu: Vec<[f64; 3]>,
    area_t: Vec<f64>,
    lap_t: Vec<[f64; 3]>,
    lap_idx: Vec<usize>,
    /// Point-to-plane: target positions and unit vertex normals on the loss vertices.
    plane_y: Vec<[f64; 3]>,
    plane_n: Vec<[f64; 3]>,
    w: [f64; 4],
    scale: [f64; 4],
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Geo {
    fn new(faces_all: &[[usize; 3]], in_loss: &[bool], nbr: &[Vec<usize>], y: &[[f64; 3]], w: [f64; 4]) -> Self {
        let faces: Vec<[usize; 3]> = faces_all.iter().copied().filter(|f| f.iter().all(|&i| in_loss[i])).collect();
        let (mut nu, mut area_t) = (Vec::new(), Vec::new());
        for f in &faces {
            let n = cross3(sub3(y[f[1]], y[f[0]]), sub3(y[f[2]], y[f[0]]));
            let l = dot3(n, n).sqrt();
            nu.push(n.map(|c| c / l));
            area_t.push(0.5 * l);
        }
        let lap_idx: Vec<usize> = (0..nbr.len()).filter(|&i| in_loss[i]).collect();
        let lap_t = lap_idx.iter().map(|&i| Self::umbrella(nbr, y, i)).collect();
        // vertex normals of the target (area weighted, all faces around the vertex)
        let mut vn = vec![[0.0; 3]; nbr.len()];
        for f in faces_all {
            let n = cross3(sub3(y[f[1]], y[f[0]]), sub3(y[f[2]], y[f[0]]));
            for &i in f {
                (0..3).for_each(|d| vn[i][d] += n[d]);
            }
        }
        let plane_y = lap_idx.iter().map(|&i| y[i]).collect();
        let plane_n = lap_idx.iter().map(|&i| { let l = dot3(vn[i], vn[i]).sqrt(); vn[i].map(|c| c / l) }).collect();
        Self { faces, nu, area_t, lap_t, lap_idx, plane_y, plane_n, w, scale: [1.0; 4] }
    }

    fn umbrella(nbr: &[Vec<usize>], x: &[[f64; 3]], i: usize) -> [f64; 3] {
        let k = nbr[i].len() as f64;
        [0, 1, 2].map(|d| x[i][d] - nbr[i].iter().map(|&j| x[j][d]).sum::<f64>() / k)
    }

    /// Raw terms [normal, laplacian, area, point-to-plane] and the gradient of sum_t w_t / scale_t * term_t.
    fn eval(&self, nbr: &[Vec<usize>], x: &[[f64; 3]], n_nodes: usize) -> ([f64; 4], Vec<[f64; 3]>) {
        let mut g = vec![[0.0; 3]; n_nodes];
        let mut add = |i: usize, v: [f64; 3], c: f64| (0..3).for_each(|d| g[i][d] += c * v[d]);
        let nf = self.faces.len() as f64;
        let a_mean = self.area_t.iter().sum::<f64>() / nf;
        let (mut ln, mut la) = (0.0, 0.0);
        let (cn, ca) = (self.w[0] / self.scale[0], self.w[2] / self.scale[2]);
        for (k, f) in self.faces.iter().enumerate() {
            let (e1, e2) = (sub3(x[f[1]], x[f[0]]), sub3(x[f[2]], x[f[0]]));
            let n = cross3(e1, e2);
            let l = dot3(n, n).sqrt();
            let u = n.map(|c| c / l);
            let un = dot3(u, self.nu[k]);
            ln += self.area_t[k] * (1.0 - un) / (nf * a_mean);
            let lr = (0.5 * l / self.area_t[k]).ln();
            la += 0.5 * lr * lr / nf;
            // dL/dn for both terms, then n = e1 x e2: grad_b = e2 x w, grad_c = w x e1, grad_a = -(both).
            let wn = (0..3).map(|d| -cn * self.area_t[k] / (nf * a_mean) * (self.nu[k][d] - un * u[d]) / l);
            let wa = (0..3).map(|d| ca * lr / nf * u[d] / l);
            let w: Vec<f64> = wn.zip(wa).map(|(p, q)| p + q).collect();
            let w = [w[0], w[1], w[2]];
            let (gb, gc) = (cross3(e2, w), cross3(w, e1));
            add(f[1], gb, 1.0);
            add(f[2], gc, 1.0);
            add(f[0], gb, -1.0);
            add(f[0], gc, -1.0);
        }
        // Laplacian coordinates: d = M x, gradient M^T r on the loss vertices.
        let cl = self.w[1] / self.scale[1];
        let nl = self.lap_idx.len() as f64;
        let mut ll = 0.0;
        for (k, &i) in self.lap_idx.iter().enumerate() {
            let r = sub3(Self::umbrella(nbr, x, i), self.lap_t[k]);
            ll += 0.5 * dot3(r, r) / nl;
            let kk = nbr[i].len() as f64;
            add(i, r, cl / nl);
            for &j in &nbr[i] {
                add(j, r, -cl / nl / kk);
            }
        }
        // Point-to-plane: mean_i ((x_i - y_i) . n_i)^2 / 2, tangential sliding free.
        let cp = self.w[3] / self.scale[3];
        let mut lp = 0.0;
        for (k, &i) in self.lap_idx.iter().enumerate() {
            let e = dot3(sub3(x[i], self.plane_y[k]), self.plane_n[k]);
            lp += 0.5 * e * e / nl;
            add(i, self.plane_n[k], cp * e / nl);
        }
        ([ln, ll, la, lp], g)
    }
}

impl Problem {
    fn growth(&self, theta: &[f64]) -> Vec<f64> {
        self.tets
            .iter()
            .zip(&self.gm)
            .map(|(t, g)| g * 0.25 * t.iter().map(|&n| theta[self.owner[n as usize]]).sum::<f64>())
            .collect()
    }

    /// Equilibrium of theta from positions `x0`: `steps` damped steps, then matrix-free
    /// Newton-Krylov on F(x) = 0 (K delta = F by CG, backtracking on |F|, contact lists
    /// refreshed at each trial point). Returns x*.
    /// Offsets of the extra field blocks in theta (c, then k).
    fn field_offsets(&self) -> (Option<usize>, Option<usize>) {
        let oc = self.fit_c.then_some(self.na);
        let ok = self.fit_k.then_some(self.na * (1 + usize::from(self.fit_c)));
        (oc, ok)
    }

    /// Sets the thickening / core corrections of theta's extra blocks in both simulations.
    fn apply_fields(&mut self, theta: &[f64]) {
        let per_tet = |d: &[f64]| -> Vec<f64> {
            self.tets.iter().map(|t| 0.25 * t.iter().map(|&n| d[self.owner[n as usize]]).sum::<f64>()).collect()
        };
        let (oc, ok) = self.field_offsets();
        if let Some(o) = oc {
            let d = per_tet(&theta[o..o + self.na]);
            self.sim.set_radial_delta(Some(&d));
            self.ad.set_radial_delta(Some(&d));
        }
        if let Some(o) = ok {
            let d = per_tet(&theta[o..o + self.na]);
            self.sim.set_core_delta(Some(&d));
            self.ad.set_core_delta(Some(&d));
        }
    }

    fn equilibrium(&mut self, theta: &[f64], x0: &[[f64; 3]], steps: usize) -> Vec<[f64; 3]> {
        self.apply_fields(theta);
        let s = self.growth(theta);
        self.sim.set_state(x0, &self.v0, &s);
        self.sim.set_fixed_growth(Some(&s));
        for _ in 0..steps {
            self.sim.step();
        }
        let mut x = self.sim.positions();
        self.build_preconditioner(&x);
        let norm = |f: &[[f64; 3]]| {
            f.iter()
                .map(|r| r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
                .sum::<f64>()
                .sqrt()
        };
        let mut f = self.sim.total_force(&x);
        let r0 = norm(&f);
        let mut rn = r0;
        for _ in 0..self.newton {
            let (delta, _, _) = self.solve(&x, &f, self.sigma, 100, !self.fd_k);
            let mut accepted = false;
            for alpha in [1.0, 0.5, 0.25] {
                let xt: Vec<[f64; 3]> = x
                    .iter()
                    .zip(&delta)
                    .map(|(a, b)| [0, 1, 2].map(|d| a[d] + alpha * b[d]))
                    .collect();
                self.sim.set_state(&xt, &self.v0, &s);
                self.sim.refresh_contact();
                let ft = self.sim.total_force(&xt);
                let rt = norm(&ft);
                if rt < rn {
                    (x, f, rn) = (xt, ft, rt);
                    accepted = true;
                    break;
                }
            }
            if !accepted || rn < (1e-3 * r0).max(self.force_tol) {
                break;
            }
        }
        self.last_residual = (r0, rn);
        self.sim.set_state(&x, &self.v0, &s);
        self.sim.refresh_contact();
        x
    }

    /// Shape loss, its gradient w.r.t. x (all nodes), and the regulariser.
    /// S v with S = (I + D^-1 A) / 2 on the surface vertices.
    fn smooth(&self, v: &[[f64; 3]]) -> Vec<[f64; 3]> {
        self.nbr
            .iter()
            .enumerate()
            .map(|(i, nb)| {
                let k = nb.len() as f64;
                [0, 1, 2].map(|d| 0.5 * v[i][d] + 0.5 * nb.iter().map(|&j| v[j][d]).sum::<f64>() / k)
            })
            .collect()
    }

    /// S^m z and (S^T)^m z for a scalar field on the surface vertices.
    fn smooth_scalar(&self, z: &[f64], m: usize, transpose: bool) -> Vec<f64> {
        let mut v: Vec<[f64; 3]> = z.iter().map(|&a| [a, 0.0, 0.0]).collect();
        for _ in 0..m {
            v = if transpose { self.smooth_t(&v) } else { self.smooth(&v) };
        }
        v.iter().map(|a| a[0]).collect()
    }

    /// S^T v (A symmetric): 0.5 v_j + 0.5 sum over neighbours i of v_i / deg_i.
    fn smooth_t(&self, v: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let w: Vec<[f64; 3]> = v
            .iter()
            .zip(&self.nbr)
            .map(|(a, nb)| a.map(|c| c / nb.len() as f64))
            .collect();
        self.nbr
            .iter()
            .enumerate()
            .map(|(j, nb)| [0, 1, 2].map(|d| 0.5 * v[j][d] + 0.5 * nb.iter().map(|&i| w[i][d]).sum::<f64>()))
            .collect()
    }

    fn detail(&self, x: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let xs = &x[..self.nbr.len()];
        let mut sm = xs.to_vec();
        for _ in 0..self.hp {
            sm = self.smooth(&sm);
        }
        xs.iter()
            .zip(&sm)
            .map(|(a, b)| [0, 1, 2].map(|d| a[d] - b[d]))
            .collect()
    }

    /// Back-propagates a gradient w.r.t. the normalised cortex points (x^ = (x - c) / r)
    /// to all nodes: d/du_j = (g_j - (sum_i g_i . x^_i) x^_j / N) / r, then centring.
    fn through_normalisation(&self, xh: &[[f64; 3]], r: f64, g: &[[f64; 3]], n_nodes: usize) -> Vec<[f64; 3]> {
        let n = xh.len() as f64;
        let gx: f64 = g
            .iter()
            .zip(xh)
            .map(|(a, b)| a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            .sum();
        let mut du: Vec<[f64; 3]> = g
            .iter()
            .zip(xh)
            .map(|(a, b)| [0, 1, 2].map(|d| (a[d] - gx * b[d] / n) / r))
            .collect();
        let mean = [0, 1, 2].map(|d| du.iter().map(|v| v[d]).sum::<f64>() / n);
        for v in &mut du {
            for d in 0..3 {
                v[d] -= mean[d];
            }
        }
        let mut gfull = vec![[0.0; 3]; n_nodes];
        for (k, &i) in self.cortex.iter().enumerate() {
            gfull[i] = du[k];
        }
        gfull
    }

    /// Symmetric Chamfer distance between the normalised cortex surfaces (no vertex matching):
    /// 1/(2N) sum_i |x^_i - y^_nn(i)|^2 + 1/(2N) sum_j |y^_j - x^_nn(j)|^2.
    fn loss_chamfer(&self, x: &[[f64; 3]], theta: &[f64]) -> (f64, f64, Vec<[f64; 3]>) {
        let (xh, _, r) = normalise(x, &self.cortex);
        let n = xh.len() as f64;
        let to_y = gyroxide_mesh::distance::nearest(&xh, &self.target_hat);
        let to_x = gyroxide_mesh::distance::nearest(&self.target_hat, &xh);
        let mut g = vec![[0.0; 3]; xh.len()];
        let mut ls = 0.0;
        for (i, &j) in to_y.index.iter().enumerate() {
            let y = self.target_hat[j as usize];
            for d in 0..3 {
                let e = xh[i][d] - y[d];
                ls += 0.5 * e * e / n;
                g[i][d] += e / n;
            }
        }
        for (j, &i) in to_x.index.iter().enumerate() {
            let y = self.target_hat[j];
            for d in 0..3 {
                let e = xh[i as usize][d] - y[d];
                ls += 0.5 * e * e / n;
                g[i as usize][d] += e / n;
            }
        }
        let gfull = self.through_normalisation(&xh, r, &g, x.len());
        let lr = 0.5
            * self.beta
            * self
                .edges
                .iter()
                .map(|[i, j]| (theta[*i] - self.theta0[*i] - theta[*j] + self.theta0[*j]).powi(2))
                .sum::<f64>();
        (ls, lr, gfull)
    }

    fn reg(&self, theta: &[f64]) -> f64 {
        let (oc, ok) = self.field_offsets();
        let fields: f64 = [oc, ok]
            .into_iter()
            .flatten()
            .map(|o| self.edges.iter().map(|[i, j]| (theta[o + *i] - theta[o + *j]).powi(2)).sum::<f64>())
            .sum();
        0.5 * self.beta
            * (self
                .edges
                .iter()
                .map(|[i, j]| (theta[*i] - self.theta0[*i] - theta[*j] + self.theta0[*j]).powi(2))
                .sum::<f64>()
                + fields)
    }

    /// S^k of the surface vertices (k = self.lp).
    fn lowpass(&self, x: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let mut sm = x[..self.nbr.len()].to_vec();
        for _ in 0..self.lp {
            sm = self.smooth(&sm);
        }
        sm
    }

    /// Selects the low-pass level k of the multi-resolution loss (k = 0: vertex shape loss).
    fn set_level(&mut self, k: usize) {
        self.lp = k;
        if k > 0 {
            let t = self.target.clone();
            self.target_lp = normalise(&self.lowpass(&t), &self.cortex).0;
        }
    }

    /// Low-pass shape loss: 1/(2N) sum over cortex |x^_i - y^_i|^2 with x^, y^ the normalised
    /// smoothed surfaces S^k x, S^k y; gradient (S^T)^k of the normalisation gradient.
    fn loss_lp(&self, x: &[[f64; 3]], theta: &[f64]) -> (f64, f64, Vec<[f64; 3]>) {
        let ns = self.nbr.len();
        let (xh, _, r) = normalise(&self.lowpass(x), &self.cortex);
        let n = self.cortex.len() as f64;
        let mut ls = 0.0;
        let mut g = vec![[0.0; 3]; xh.len()];
        for (k, (a, b)) in xh.iter().zip(&self.target_lp).enumerate() {
            for d in 0..3 {
                let e = a[d] - b[d];
                ls += 0.5 * e * e / n;
                g[k][d] = e / n;
            }
        }
        let mut st = self.through_normalisation(&xh, r, &g, ns);
        for _ in 0..self.lp {
            st = self.smooth_t(&st);
        }
        let mut gfull = vec![[0.0; 3]; x.len()];
        gfull[..ns].copy_from_slice(&st);
        (ls, self.reg(theta), gfull)
    }

    /// Shape loss of level k (smoothed surfaces S^k x vs normalised smoothed target `t_hat`),
    /// and its gradient on the surface vertices.
    fn level_loss(&self, x: &[[f64; 3]], k: usize, t_hat: &[[f64; 3]]) -> (f64, Vec<[f64; 3]>) {
        let ns = self.nbr.len();
        let mut sm = x[..ns].to_vec();
        for _ in 0..k {
            sm = self.smooth(&sm);
        }
        if self.highpass > k {
            // Band S^k x - S^K x (K = --highpass): the lobar shape is left out, translation-free.
            let mut lo = sm.clone();
            for _ in k..self.highpass {
                lo = self.smooth(&lo);
            }
            let n = self.cortex.len() as f64;
            let mut ls = 0.0;
            let mut g = vec![[0.0; 3]; ns];
            for (j, &i) in self.cortex.iter().enumerate() {
                for d in 0..3 {
                    let e = sm[i][d] - lo[i][d] - t_hat[j][d];
                    ls += 0.5 * e * e / n;
                    g[i][d] = e / n;
                }
            }
            // d/dx = ((S^T)^k - (S^T)^K) g.
            let mut gk = g;
            for _ in 0..k {
                gk = self.smooth_t(&gk);
            }
            let mut gkk = gk.clone();
            for _ in k..self.highpass {
                gkk = self.smooth_t(&gkk);
            }
            let st = gk.iter().zip(&gkk).map(|(a, b)| [0, 1, 2].map(|d| a[d] - b[d])).collect();
            return (ls, st);
        }
        if self.rigid {
            // t_hat: centred smoothed target. L = 1/(2N) sum |R xc_i - t_i|^2, R fixed in the
            // gradient (optimal R: envelope theorem), then centring.
            let xc = centred(&self.cortex.iter().map(|&i| sm[i]).collect::<Vec<_>>());
            let r = best_rotation(&xc, t_hat);
            let n = xc.len() as f64;
            let mut ls = 0.0;
            let mut g: Vec<[f64; 3]> = Vec::with_capacity(xc.len());
            for (a, b) in xc.iter().zip(t_hat) {
                let e: [f64; 3] = std::array::from_fn(|i| r[i][0] * a[0] + r[i][1] * a[1] + r[i][2] * a[2] - b[i]);
                ls += 0.5 * (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) / n;
                g.push(std::array::from_fn(|j| (r[0][j] * e[0] + r[1][j] * e[1] + r[2][j] * e[2]) / n));
            }
            let mean = [0, 1, 2].map(|d| g.iter().map(|v| v[d]).sum::<f64>() / n);
            let mut st = vec![[0.0; 3]; ns];
            for (k, &i) in self.cortex.iter().enumerate() {
                st[i] = [0, 1, 2].map(|d| g[k][d] - mean[d]);
            }
            for _ in 0..k {
                st = self.smooth_t(&st);
            }
            return (ls, st);
        }
        let xh: Vec<[f64; 3]> = if self.absolute {
            self.cortex.iter().map(|&i| sm[i]).collect()
        } else {
            normalise(&sm, &self.cortex).0
        };
        let n = self.cortex.len() as f64;
        let mut ls = 0.0;
        let mut g = vec![[0.0; 3]; xh.len()];
        for (i, (a, b)) in xh.iter().zip(t_hat).enumerate() {
            for d in 0..3 {
                let e = a[d] - b[d];
                ls += 0.5 * e * e / n;
                g[i][d] = e / n;
            }
        }
        let mut st = if self.absolute {
            let mut gs = vec![[0.0; 3]; ns];
            for (k, &i) in self.cortex.iter().enumerate() {
                gs[i] = g[k];
            }
            gs
        } else {
            let r = normalise(&sm, &self.cortex).2;
            self.through_normalisation(&xh, r, &g, ns)
        };
        for _ in 0..k {
            st = self.smooth_t(&st);
        }
        (ls, st)
    }

    /// Smoothed surface S^k of the surface vertices of x.
    fn smoothed(&self, x: &[[f64; 3]], k: usize) -> Vec<[f64; 3]> {
        let mut sm = x[..self.nbr.len()].to_vec();
        for _ in 0..k {
            sm = self.smooth(&sm);
        }
        sm
    }

    /// Deviation in mm from the atlas at level k over the cortex: (absolute, shape-only), each
    /// [rms, p50, p95, max]. Shape-only: normalised surfaces, rescaled to the atlas rms radius.
    fn deviation_mm(&self, x: &[[f64; 3]], k: usize) -> ([f64; 4], [f64; 4], [f64; 4]) {
        let sx = self.smoothed(x, k);
        let sy = self.smoothed(&self.target, k);
        let stats = |mut d: Vec<f64>| {
            let rms = (d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64).sqrt();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |f: f64| d[((d.len() - 1) as f64 * f).round() as usize];
            [rms, q(0.5), q(0.95), d[d.len() - 1]]
        };
        let dist = |a: &[f64; 3], b: &[f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        let abs = stats(self.cortex.iter().map(|&i| self.mm * dist(&sx[i], &sy[i])).collect());
        let xc = centred(&self.cortex.iter().map(|&i| sx[i]).collect::<Vec<_>>());
        let yc = centred(&self.cortex.iter().map(|&i| sy[i]).collect::<Vec<_>>());
        let r = best_rotation(&xc, &yc);
        let rigid = stats(
            xc.iter()
                .zip(&yc)
                .map(|(a, b)| self.mm * dist(&std::array::from_fn(|i| r[i][0] * a[0] + r[i][1] * a[1] + r[i][2] * a[2]), b))
                .collect(),
        );
        let (xh, _, _) = normalise(&sx, &self.cortex);
        let (yh, _, ry) = normalise(&sy, &self.cortex);
        let shape = stats(xh.iter().zip(&yh).map(|(a, b)| self.mm * ry * dist(a, b)).collect());
        (abs, rigid, shape)
    }

    fn report_mm(&self, tag: &str, x: &[[f64; 3]], levels: &[usize]) -> Vec<(usize, [f64; 4], [f64; 4], [f64; 4])> {
        let mut out = Vec::new();
        for &k in levels {
            let (a, g, s) = self.deviation_mm(x, k);
            println!(
                "{tag} mm k={k:>3}: absolute rms {:.3} p95 {:.3} | rigid rms {:.3} p50 {:.3} p95 {:.3} max {:.3} | shape rms {:.3} p95 {:.3}",
                a[0], a[2], g[0], g[1], g[2], g[3], s[0], s[2]
            );
            out.push((k, a, g, s));
        }
        out
    }

    /// Normalised smoothed target of level k.
    fn level_target(&self, k: usize) -> Vec<[f64; 3]> {
        let sm = self.smoothed(&self.target, k);
        if self.highpass > k {
            let lo = self.smoothed(&sm, self.highpass - k);
            return self.cortex.iter().map(|&i| [0, 1, 2].map(|d| sm[i][d] - lo[i][d])).collect();
        }
        if self.rigid {
            return centred(&self.cortex.iter().map(|&i| sm[i]).collect::<Vec<_>>());
        }
        if self.absolute {
            return self.cortex.iter().map(|&i| sm[i]).collect();
        }
        normalise(&sm, &self.cortex).0
    }

    /// FI = A / |V|^(2/3) of the surface triangles and its gradient w.r.t. the surface vertices.
    fn fi_grad(&self, x: &[[f64; 3]]) -> (f64, Vec<[f64; 3]>) {
        let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let ns = self.nbr.len();
        let (mut area, mut vol) = (0.0, 0.0);
        let mut ga = vec![[0.0; 3]; ns];
        let mut gv = vec![[0.0; 3]; ns];
        for f in &self.faces {
            let (a, b, c) = (x[f[0]], x[f[1]], x[f[2]]);
            let cr = cross(sub(b, a), sub(c, a));
            let len = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt().max(1e-30);
            area += 0.5 * len;
            let nh = cr.map(|v| v / len);
            // dA/dp_i = 0.5 n x (p_{i+2} - p_{i+1}); dV/dp_i = (p_{i+1} x p_{i+2}) / 6.
            let pts = [a, b, c];
            for k in 0..3 {
                let (p1, p2) = (pts[(k + 1) % 3], pts[(k + 2) % 3]);
                let da = cross(nh, sub(p2, p1));
                let dv = cross(p1, p2);
                for d in 0..3 {
                    ga[f[k]][d] += 0.5 * da[d];
                    gv[f[k]][d] += dv[d] / 6.0;
                }
            }
            vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
        }
        let av = vol.abs();
        let fi = area / av.powf(2.0 / 3.0);
        let kv = -(2.0 / 3.0) * area * av.powf(-5.0 / 3.0) * vol.signum();
        let g = ga
            .iter()
            .zip(&gv)
            .map(|(a, v)| [0, 1, 2].map(|d| a[d] / av.powf(2.0 / 3.0) + kv * v[d]))
            .collect();
        (fi, g)
    }

    /// Fit loss with the optional folding-index and measured-growth terms.
    fn loss(&self, x: &[[f64; 3]], theta: &[f64]) -> (f64, f64, Vec<[f64; 3]>) {
        let (mut ls, mut lr, mut g) = self.loss_base(x, theta);
        if self.w_vertex != 1.0 {
            ls *= self.w_vertex;
            g.iter_mut().for_each(|r| r.iter_mut().for_each(|v| *v *= self.w_vertex));
        }
        if let Some(geo) = &self.geo {
            let (t, gg) = geo.eval(&self.nbr, x, x.len());
            ls += (0..4).map(|k| geo.w[k] * t[k] / geo.scale[k]).sum::<f64>();
            for (a, b) in g.iter_mut().zip(&gg) {
                (0..3).for_each(|d| a[d] += b[d]);
            }
        }
        if self.fi_weight > 0.0 {
            let (fi, gfi) = self.fi_grad(x);
            let e = (fi - self.fi_target) / self.fi_target;
            ls += 0.5 * self.fi_weight * e * e;
            for (gi, fi_i) in g.iter_mut().zip(&gfi) {
                for d in 0..3 {
                    gi[d] += self.fi_weight * e / self.fi_target * fi_i[d];
                }
            }
        }
        if self.alpha > 0.0 {
            let n = theta.len() as f64;
            lr += 0.5 * self.alpha / self.mes_scale
                * theta.iter().zip(&self.theta_mes).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
                / n;
        }
        (ls, lr, g)
    }

    fn loss_base(&self, x: &[[f64; 3]], theta: &[f64]) -> (f64, f64, Vec<[f64; 3]>) {
        if !self.active.is_empty() {
            let ns = self.nbr.len();
            let mut ls = 0.0;
            let mut gfull = vec![[0.0; 3]; x.len()];
            for (k, w, t_hat) in &self.active {
                let (l, g) = self.level_loss(x, *k, t_hat);
                ls += w * l;
                for i in 0..ns {
                    for d in 0..3 {
                        gfull[i][d] += w * g[i][d];
                    }
                }
            }
            return (ls, self.reg(theta), gfull);
        }
        if self.lp > 0 {
            return self.loss_lp(x, theta);
        }
        if self.chamfer {
            return self.loss_chamfer(x, theta);
        }
        if self.hp > 0 {
            return self.loss_hp(x, theta);
        }
        let (xh, _, r) = normalise(x, &self.cortex);
        let n = self.cortex.len() as f64;
        let mut ls = 0.0;
        let mut g: Vec<[f64; 3]> = vec![[0.0; 3]; xh.len()];
        for (k, (a, b)) in xh.iter().zip(&self.target_hat).enumerate() {
            for d in 0..3 {
                let e = a[d] - b[d];
                ls += 0.5 * e * e / n;
                g[k][d] = e / n;
            }
        }
        // d/du_j = (g_j - (sum_i g_i . xh_i) xh_j / N) / r, then remove the mean (centring).
        let gx: f64 = g
            .iter()
            .zip(&xh)
            .map(|(a, b)| a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            .sum();
        let mut du: Vec<[f64; 3]> = g
            .iter()
            .zip(&xh)
            .map(|(a, b)| [0, 1, 2].map(|d| (a[d] - gx * b[d] / n) / r))
            .collect();
        let mean = [0, 1, 2].map(|d| du.iter().map(|v| v[d]).sum::<f64>() / n);
        for v in &mut du {
            for d in 0..3 {
                v[d] -= mean[d];
            }
        }
        let mut gfull = vec![[0.0; 3]; x.len()];
        for (k, &i) in self.cortex.iter().enumerate() {
            gfull[i] = du[k];
        }
        let lr = 0.5
            * self.beta
            * self
                .edges
                .iter()
                .map(|[i, j]| (theta[*i] - self.theta0[*i] - theta[*j] + self.theta0[*j]).powi(2))
                .sum::<f64>();
        (ls, lr, gfull)
    }

    /// Fold loss: 1/(2N) sum over cortex |(x - S^k x)_i - (y - S^k y)_i|^2 / r^2.
    fn loss_hp(&self, x: &[[f64; 3]], theta: &[f64]) -> (f64, f64, Vec<[f64; 3]>) {
        let d = self.detail(x);
        let n = self.cortex.len() as f64;
        let r2 = self.r_ref * self.r_ref;
        let mut gd = vec![[0.0; 3]; d.len()];
        let mut ls = 0.0;
        for &i in &self.cortex {
            for k in 0..3 {
                let e = d[i][k] - self.target_detail[i][k];
                ls += 0.5 * e * e / (n * r2);
                gd[i][k] = e / (n * r2);
            }
        }
        // d(detail)/dx = I - S^k, so the gradient is (I - (S^T)^k) gd.
        let mut st = gd.clone();
        for _ in 0..self.hp {
            st = self.smooth_t(&st);
        }
        let mut gfull = vec![[0.0; 3]; x.len()];
        for i in 0..gd.len() {
            gfull[i] = [0, 1, 2].map(|k| gd[i][k] - st[i][k]);
        }
        let lr = 0.5
            * self.beta
            * self
                .edges
                .iter()
                .map(|[i, j]| (theta[*i] - self.theta0[*i] - theta[*j] + self.theta0[*j]).powi(2))
                .sum::<f64>();
        (ls, lr, gfull)
    }

    /// K v with K = -dF/dx for the adjoint: elastic part exact (forward-mode tangent),
    /// contact and mid-plane by central differences (central differences of the whole f32
    /// force at equilibrium are dominated by round-off and break CG). --fd-k: old path.
    fn k_mul(&self, x: &[[f64; 3]], v: &[[f64; 3]]) -> Vec<[f64; 3]> {
        if self.fd_k {
            return self.k_mul_fd(x, v);
        }
        self.sim.stiffness_product(x, v)
    }

    fn k_mul_fd(&self, x: &[[f64; 3]], v: &[[f64; 3]]) -> Vec<[f64; 3]> {
        self.k_mul_fd_eps(x, v, 3e-5)
    }

    fn k_mul_fd_eps(&self, x: &[[f64; 3]], v: &[[f64; 3]], e: f64) -> Vec<[f64; 3]> {
        let vmax = v
            .iter()
            .flat_map(|r| r.iter())
            .fold(0.0f64, |m, a| m.max(a.abs()))
            .max(1e-30);
        let eps = e / vmax;
        let shift = |s: f64| -> Vec<[f64; 3]> {
            x.iter()
                .zip(v)
                .map(|(a, b)| [0, 1, 2].map(|d| a[d] + s * b[d]))
                .collect()
        };
        let fp = self.sim.total_force(&shift(eps));
        let fm = self.sim.total_force(&shift(-eps));
        fp.iter()
            .zip(&fm)
            .map(|(a, b)| [0, 1, 2].map(|d| -(a[d] - b[d]) / (2.0 * eps)))
            .collect()
    }

    /// Block-Jacobi preconditioner of K + sigma I at x: one K product per colour and direction.
    fn build_preconditioner(&mut self, x: &[[f64; 3]]) {
        let n = x.len();
        let mut blocks = vec![[[0.0; 3]; 3]; n];
        for group in &self.colours {
            for d in 0..3 {
                let mut v = vec![[0.0; 3]; n];
                for &i in group {
                    v[i][d] = 1.0;
                }
                let kv = self.k_mul_fd(x, &v);
                for &i in group {
                    for r in 0..3 {
                        blocks[i][r][d] = kv[i][r];
                    }
                }
            }
        }
        self.pre_inv = blocks
            .into_iter()
            .map(|mut b| {
                for r in 0..3 {
                    b[r][r] += self.sigma;
                }
                // Symmetrise; fall back to the inverted |diagonal| if not positive definite.
                for r in 0..3 {
                    for c in r + 1..3 {
                        let m = 0.5 * (b[r][c] + b[c][r]);
                        b[r][c] = m;
                        b[c][r] = m;
                    }
                }
                inv3(b).unwrap_or_else(|| {
                    let mut z = [[0.0; 3]; 3];
                    for r in 0..3 {
                        z[r][r] = 1.0 / b[r][r].abs().max(self.sigma);
                    }
                    z
                })
            })
            .collect();
    }

    fn precondition(&self, r: &[[f64; 3]]) -> Vec<[f64; 3]> {
        if self.pre_inv.is_empty() {
            return r.to_vec();
        }
        r.iter()
            .zip(&self.pre_inv)
            .map(|(v, m)| [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]))
            .collect()
    }

    /// Solves (K + sigma I) lambda = b by preconditioned CG; stops on negative curvature or
    /// a non-finite residual and returns the iterate of smallest residual (near buckling K is
    /// indefinite and the residual can grow by orders of magnitude before p.Kp turns negative).
    fn adjoint(&self, x: &[[f64; 3]], b: &[[f64; 3]], sigma: f64, iters: usize) -> (Vec<[f64; 3]>, usize, f64) {
        // Fixed nodes: K is the identity there and the loss does not act on them.
        let b: Vec<[f64; 3]> = b.iter().enumerate().map(|(i, r)| if self.fixed.get(i).copied().unwrap_or(false) { [0.0; 3] } else { *r }).collect();
        let b = &b[..];
        self.solve(x, b, sigma, iters, true)
    }

    /// CG on (K + sigma I) z = b; `exact`: autodiff K products (adjoint), else finite
    /// differences (Newton steps of the forward equilibrium, where accuracy matters less).
    fn solve(&self, x: &[[f64; 3]], b: &[[f64; 3]], sigma: f64, iters: usize, exact: bool) -> (Vec<[f64; 3]>, usize, f64) {
        let dot = |a: &[[f64; 3]], c: &[[f64; 3]]| {
            a.iter()
                .zip(c)
                .map(|(p, q)| p[0] * q[0] + p[1] * q[1] + p[2] * q[2])
                .sum::<f64>()
        };
        let axpy = |y: &mut [[f64; 3]], a: f64, z: &[[f64; 3]]| {
            for (p, q) in y.iter_mut().zip(z) {
                for d in 0..3 {
                    p[d] += a * q[d];
                }
            }
        };
        let mut lam = vec![[0.0; 3]; b.len()];
        // Deflation: CG on P (K + sigma) P with P = I - D D^T (D: self.deflate).
        let project = |v: &mut [[f64; 3]]| {
            for d in &self.deflate {
                let c = dot(d, v);
                axpy(v, -c, d);
            }
        };
        let mut r = b.to_vec();
        project(&mut r);
        let mut z = self.precondition(&r);
        project(&mut z);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let b2 = dot(&r, &r);
        let mut rr = b2;
        let mut it = 0;
        let (mut best, mut best_rr) = (lam.clone(), rr);
        while it < iters && rr > 1e-8 * b2 {
            let mut kp = if exact { self.k_mul(x, &p) } else { self.k_mul_fd(x, &p) };
            axpy(&mut kp, sigma, &p);
            project(&mut kp);
            let pkp = dot(&p, &kp);
            if pkp <= 0.0 {
                if std::env::var_os("CG_DEBUG").is_some() {
                    println!("cg: negative curvature at iteration {it}: p.(K+sigma)p / p.p = {:.4e}", pkp / dot(&p, &p));
                    if let Some(path) = std::env::var_os("CG_DEBUG_SAVE") {
                        save_rows(PathBuf::from(path), &p);
                    }
                }
                break;
            }
            let alpha = rz / pkp;
            axpy(&mut lam, alpha, &p);
            axpy(&mut r, -alpha, &kp);
            rr = dot(&r, &r);
            if !rr.is_finite() {
                break;
            }
            if rr < best_rr {
                best_rr = rr;
                best.clone_from(&lam);
            }
            z = self.precondition(&r);
            project(&mut z);
            let rz2 = dot(&r, &z);
            let beta = rz2 / rz;
            rz = rz2;
            for (pp, q) in p.iter_mut().zip(&z) {
                for d in 0..3 {
                    pp[d] = q[d] + beta * pp[d];
                }
            }
            it += 1;
        }
        (best, it, (best_rr / b2).sqrt())
    }

    /// lambda . dF/ds per tetrahedron (one reverse-mode pass of the regular elastic branch).
    fn vjp_growth(&self, x: &[[f64; 3]], s: &[f64], lam: &[[f64; 3]]) -> Vec<f64> {
        let dev = WgpuDevice::default();
        let xv = V3::<Ad>::from_rows(x, &dev);
        let lv = V3::<Ad>::from_rows(lam, &dev);
        let st: Tensor<Ad, 1> = Tensor::from_data(
            TensorData::new(s.iter().map(|&v| v as f32).collect::<Vec<_>>(), [s.len()]),
            &dev,
        )
        .require_grad();
        let h: Tensor<Ad, 1> = Tensor::from_data(
            TensorData::new(
                self.thick.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                [self.thick.len()],
            ),
            &dev,
        );
        let mat = self.ad.params.material_at(self.sim.t);
        let f = forces_regular(self.ad.precomputed(), &mat, &xv, &st, &Thickness::PerTet(h), self.ad.core_growth().as_ref(), self.ad.radial_growth().as_ref());
        let l = (f.x * lv.x).sum() + (f.y * lv.y).sum() + (f.z * lv.z).sum();
        let grads = l.backward();
        to_f64(st.grad(&grads).expect("gradient w.r.t. growth"))
    }

    /// lambda . dF/ds per tetrahedron, and optionally lambda . dF/dc (thickening c) and
    /// lambda . dF/d delta (isotropic correction delta I of the core growth U) of `ad`.
    fn vjp_fields(&self, x: &[[f64; 3]], s: &[f64], lam: &[[f64; 3]], want_c: bool, want_k: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let dev = WgpuDevice::default();
        let xv = V3::<Ad>::from_rows(x, &dev);
        let lv = V3::<Ad>::from_rows(lam, &dev);
        let t1 = |v: &[f64]| -> Tensor<Ad, 1> {
            Tensor::from_data(TensorData::new(v.iter().map(|&a| a as f32).collect::<Vec<_>>(), [v.len()]), &dev)
        };
        let st = t1(s).require_grad();
        let ct = self.ad.radial_growth().map(|c| {
            let t = t1(&to_f64(c));
            if want_c { t.require_grad() } else { t }
        });
        let nt = s.len();
        let kt = t1(&vec![0.0; nt]).require_grad();
        let core = self.ad.core_growth().map(|mut u| {
            if want_k {
                for k in 0..3 {
                    u.m[k][k] = u.m[k][k].clone() + kt.clone();
                }
            }
            u
        });
        let h = t1(&self.thick);
        let mat = self.ad.params.material_at(self.sim.t);
        let f = forces_regular(self.ad.precomputed(), &mat, &xv, &st, &Thickness::PerTet(h), core.as_ref(), ct.as_ref());
        let l = (f.x * lv.x).sum() + (f.y * lv.y).sum() + (f.z * lv.z).sum();
        let grads = l.backward();
        let gc = if want_c { to_f64(ct.as_ref().expect("field c needs thickening = true").grad(&grads).expect("gradient w.r.t. c")) } else { Vec::new() };
        let gk = if want_k { to_f64(kt.grad(&grads).expect("field k needs a core growth table")) } else { Vec::new() };
        (to_f64(st.grad(&grads).expect("gradient w.r.t. growth")), gc, gk)
    }

    /// Full gradient w.r.t. theta at the equilibrium x of theta.
    fn gradient(
        &self,
        x: &[[f64; 3]],
        theta: &[f64],
        gx: &[[f64; 3]],
        sigma: f64,
        cg: usize,
    ) -> (Vec<f64>, usize, f64) {
        let (lam, it, res) = self.adjoint(x, gx, sigma, cg);
        let (oc, ok) = self.field_offsets();
        let (gs, gc, gk) = if oc.is_some() || ok.is_some() {
            self.vjp_fields(x, &self.growth(theta), &lam, oc.is_some(), ok.is_some())
        } else {
            (self.vjp_growth(x, &self.growth(theta), &lam), Vec::new(), Vec::new())
        };
        let mut g = vec![0.0; theta.len()];
        for ((t, gm), gse) in self.tets.iter().zip(&self.gm).zip(&gs) {
            for &n in t {
                g[self.owner[n as usize]] += gse * gm * 0.25;
            }
        }
        for (o, gf) in [(oc, &gc), (ok, &gk)] {
            if let Some(o) = o {
                for (t, ge) in self.tets.iter().zip(gf.iter()) {
                    for &n in t {
                        g[o + self.owner[n as usize]] += ge * 0.25;
                    }
                }
                for [i, j] in &self.edges {
                    let d = self.beta * (theta[o + *i] - theta[o + *j]);
                    g[o + *i] += d;
                    g[o + *j] -= d;
                }
            }
        }
        for [i, j] in &self.edges {
            let d = self.beta * (theta[*i] - self.theta0[*i] - theta[*j] + self.theta0[*j]);
            g[*i] += d;
            g[*j] -= d;
        }
        if self.alpha > 0.0 {
            let n = theta.len() as f64;
            for (gi, (a, b)) in g.iter_mut().zip(theta.iter().zip(&self.theta_mes)) {
                *gi += self.alpha / self.mes_scale * (a - b) / n;
            }
        }
        (g, it, res)
    }

    /// Number of inverted elements at positions x with the growth of theta (one recorded
    /// step under fixed growth; the simulation state is reset to x afterwards).
    fn n_inverted(&mut self, x: &[[f64; 3]], theta: &[f64]) -> usize {
        self.apply_fields(theta);
        let s = self.growth(theta);
        self.sim.set_state(x, &self.v0, &s);
        self.sim.set_fixed_growth(Some(&s));
        let n = self.sim.step_recorded().diagnostics.n_inverted;
        self.sim.set_state(x, &self.v0, &s);
        self.sim.refresh_contact();
        n
    }

    fn rayleigh(&self, x: &[[f64; 3]]) -> f64 {
        let v: Vec<[f64; 3]> = (0..x.len())
            .map(|i| [0, 1, 2].map(|d| (((i * 7 + d * 13) % 17) as f64 / 8.0) - 1.0))
            .collect();
        let kv = self.k_mul_fd(x, &v);
        let num: f64 = v
            .iter()
            .zip(&kv)
            .map(|(a, b)| a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            .sum();
        let den: f64 = v.iter().map(|a| a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sum();
        num / den
    }
}

fn save_rows(p: PathBuf, x: &[[f64; 3]]) {
    let flat: Vec<f64> = x.iter().flatten().copied().collect();
    npy::write_f64(p, &[x.len(), 3], &flat).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str, def: f64| -> f64 {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].parse().unwrap())
            .unwrap_or(def)
    };
    let (cfg, mesh_path, ckpt, prep, out) = (
        &args[1],
        &args[2],
        Path::new(&args[3]),
        Path::new(&args[4]),
        PathBuf::from(&args[5]),
    );
    let iters = flag("--iters", 10.0) as usize;
    let beta = flag("--beta", 1e-4);
    let relax = flag("--relax", 1500.0) as usize;
    let cg = flag("--cg", 200.0) as usize;
    let check = args.iter().any(|a| a == "--check");
    let gd = args.iter().any(|a| a == "--gd");
    let gd_step = flag("--step", 0.01);
    let opt = |name: &str| args.iter().position(|a| a == name).map(|i| PathBuf::from(&args[i + 1]));
    let theta0_table = args.iter().any(|a| a == "--theta0-table");
    let (write_table, save_ckpt, save_surface) = (opt("--write-table"), opt("--save-checkpoint"), opt("--save-surface"));
    std::fs::create_dir_all(&out).unwrap();

    let params = Params::load(cfg).unwrap();
    let mut dirichlet_mask: Vec<bool> = Vec::new();
    let Schedule::Nodal { t: knots, a: a_path, a_scale, tensor, .. } = &params.schedule else {
        panic!("adjoint_week needs the per-node growth law (law = \"nodal\")");
    };
    assert!(tensor.is_none(), "adjoint_week fits the scalar growth a_t: remove `tensor` from the config");
    let (knots, a_path, a_scale) = (knots.clone(), a_path.clone(), *a_scale);
    let mesh = netgen::read(mesh_path).unwrap();
    let t0 = Instant::now();
    let mut sim = Simulation::<Fwd>::new(mesh.clone(), params.clone(), WgpuDevice::default()).unwrap();
    let mut ad = Simulation::<Ad>::new(mesh.clone(), params.clone(), WgpuDevice::default()).unwrap();
    if let Schedule::Nodal { a, h, core, .. } = &params.schedule {
        let table = || {
            let t = NodalTable::new(&params.schedule, &npy::read(a).unwrap().data, &npy::read(h).unwrap().data, mesh.n_nodes())
                .unwrap();
            match core {
                Some(path) => t.with_core(&npy::read(path).unwrap().data).unwrap(),
                None => t,
            }
        };
        sim.set_nodal_table(table());
        ad.set_nodal_table(table());
    }
    if let Some(cp) = opt("--core-override") {
        // Remodelled core: growth tensor U per tetrahedron (tools/absorb_core.py).
        let u = read(&cp);
        assert_eq!(u.len(), 6 * mesh.n_tets(), "--core-override needs [n_tets, 6]");
        sim.set_core_override(Some(&u));
        ad.set_core_override(Some(&u));
        println!("core growth replaced per tetrahedron ({})", cp.display());
    }
    if let Some(fp) = opt("--dirichlet") {
        // Sub-mesh: nodes of the cut faces fixed at --dirichlet-pos (tools/make_slab.py).
        let fixed: Vec<bool> = read(&fp).iter().map(|&v| v > 0.5).collect();
        let pos = npy::read(opt("--dirichlet-pos").expect("--dirichlet needs --dirichlet-pos")).unwrap().rows3().unwrap();
        sim.set_dirichlet(Some((&fixed, &pos)));
        ad.set_dirichlet(Some((&fixed, &pos)));
        println!("Dirichlet: {} fixed nodes", fixed.iter().filter(|f| **f).count());
        dirichlet_mask = fixed;
    }
    let xs = npy::read(ckpt.join("x.npy")).unwrap().rows3().unwrap();
    let vs = npy::read(ckpt.join("v.npy")).unwrap().rows3().unwrap();
    let gp = read(&ckpt.join("growth_prev.npy"));
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ckpt.join("state.json")).unwrap()).unwrap();
    sim.restore(
        &xs,
        &vs,
        &gp,
        state["t"].as_f64().unwrap(),
        state["step"].as_u64().unwrap() as usize,
    );
    let t = sim.t;
    let gm = sim.cortical_indicator(t);
    let thick = sim.tet_thickness(t);
    ad.t = t;
    println!(
        "setup {:.1}s, t = {t:.4}, {} nodes, {} tets",
        t0.elapsed().as_secs_f64(),
        mesh.n_nodes(),
        mesh.n_tets()
    );

    let target = npy::read(prep.join("target.npy")).unwrap().rows3().unwrap();
    let cortex: Vec<usize> = read(&prep.join("cortex.npy"))
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0.5)
        .map(|(i, _)| i)
        .collect();
    let (target_hat, _, r_ref) = normalise(&target, &cortex);
    let e = read(&prep.join("edges.npy"));
    let edges: Vec<[usize; 2]> = e.chunks_exact(2).map(|c| [c[0] as usize, c[1] as usize]).collect();
    let owner: Vec<usize> = read(&prep.join("owner.npy")).iter().map(|&v| v as usize).collect();
    // Knot of the checkpoint time: the row of the a table that theta replaces.
    let knot = knots
        .iter()
        .position(|&k| (k - t).abs() < 1e-3)
        .unwrap_or_else(|| panic!("checkpoint t = {t} is not a knot of the a table"));
    let mut theta = if theta0_table {
        let a = npy::read(&a_path).unwrap();
        let n = a.shape[1];
        a.data[knot * n..knot * n + target.len()].iter().map(|v| v * a_scale).collect()
    } else {
        read(&prep.join("theta0.npy"))
    };
    let theta0 = theta.clone();
    let theta_mes: Vec<f64> = match opt("--measured") {
        Some(path) => {
            let a = npy::read(&path).unwrap();
            let n = a.shape[1];
            a.data[knot * n..knot * n + target.len()].iter().map(|v| v * a_scale).collect()
        }
        None => theta0.clone(),
    };
    let mes_scale = (theta_mes.iter().map(|v| v * v).sum::<f64>() / theta_mes.len() as f64).max(1e-12);
    let fi_weight = flag("--fi-weight", 0.0);
    let faces: Vec<[usize; 3]> = if fi_weight > 0.0 {
        read(&prep.join("faces.npy"))
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect()
    } else {
        Vec::new()
    };
    let mut p = Problem {
        tets: mesh.tets.clone(),
        sim,
        ad,
        owner,
        gm,
        target_hat,
        cortex,
        edges,
        beta,
        absolute: args.iter().any(|a| a == "--absolute" || a == "--rigid"),
        rigid: args.iter().any(|a| a == "--rigid"),
        highpass: flag("--highpass", 0.0) as usize,
        mm: flag("--mm-per-unit", 32.99),
        alpha: flag("--alpha", 0.0),
        theta_mes,
        mes_scale,
        fi_weight,
        fi_target: 0.0,
        faces,
        relax,
        hp: flag("--hp", 0.0) as usize,
        chamfer: args.iter().any(|a| a == "--chamfer"),
        lp: 0,
        target_lp: Vec::new(),
        active: Vec::new(),
        target: target.clone(),
        theta0: theta0.clone(),
        nbr: Vec::new(),
        target_detail: Vec::new(),
        r_ref,
        thick,
        v0: vec![[0.0; 3]; xs.len()],
        newton: flag("--newton", 20.0) as usize,
        sigma: 0.0,
        last_residual: (0.0, 0.0),
        colours: colour_nodes(mesh.n_nodes(), &mesh.tets),
        pre_inv: Vec::new(),
        deflate: Vec::new(),
        fixed: dirichlet_mask.clone(),
        geo: None,
        w_vertex: flag("--w-vertex", 1.0),
        fit_c: opt("--fit-fields").is_some_and(|f| f.to_string_lossy().contains('c')),
        fit_k: opt("--fit-fields").is_some_and(|f| f.to_string_lossy().contains('k')),
        na: 0,
        fd_k: args.iter().any(|a| a == "--fd-k"),
        force_tol: flag("--force-tol", 1e-5),
    };
    println!("{} colours for the block-Jacobi preconditioner", p.colours.len());
    p.na = theta.len();
    let n_fields = usize::from(p.fit_c) + usize::from(p.fit_k);
    if n_fields > 0 {
        theta.extend(std::iter::repeat_n(0.0, n_fields * p.na));
        println!(
            "fitted fields: a{}{} ({} values per field), field band +-{}",
            if p.fit_c { ", c (thickening correction)" } else { "" },
            if p.fit_k { ", k (core growth correction)" } else { "" },
            p.na,
            flag("--field-band", 0.3)
        );
    }
    let ns = target.len();
    let mut nbr = vec![Vec::new(); ns];
    for [i, j] in &p.edges {
        nbr[*i].push(*j);
        nbr[*j].push(*i);
    }
    p.nbr = nbr;
    let wgeo = [flag("--w-normal", 0.0), flag("--w-lap", 0.0), flag("--w-area", 0.0), flag("--w-plane", 0.0)];
    if wgeo.iter().any(|w| *w > 0.0) {
        let faces: Vec<[usize; 3]> = read(&prep.join("faces.npy")).chunks_exact(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect();
        let mut in_loss = vec![false; ns];
        p.cortex.iter().for_each(|&i| in_loss[i] = true);
        let mut geo = Geo::new(&faces, &in_loss, &p.nbr, &target, wgeo);
        let (t0, _) = geo.eval(&p.nbr, &xs, xs.len());
        geo.scale = t0.map(|v| v.max(1e-30));
        println!(
            "geometric loss: weights normal {} laplacian {} area {} plane {}, start values {:.4e} {:.4e} {:.4e} {:.4e} ({} faces, vertex weight {})",
            wgeo[0], wgeo[1], wgeo[2], wgeo[3], t0[0], t0[1], t0[2], t0[3], geo.faces.len(), p.w_vertex
        );
        p.geo = Some(geo);
    }
    if p.fi_weight > 0.0 {
        p.fi_target = p.fi_grad(&target).0;
        println!("folding-index penalty: weight {}, atlas FI {:.4}, start FI {:.4}", p.fi_weight, p.fi_target, p.fi_grad(&xs).0);
    }
    if p.alpha > 0.0 {
        println!("pull towards the measured growth: alpha {}, mean a_mes^2 {:.3e}", p.alpha, p.mes_scale);
    }
    if p.hp > 0 {
        p.target_detail = p.detail(&target);
        println!("fold (high-pass) loss, {} smoothing iterations", p.hp);
    }
    let mut weights = Vec::new();
    let levels: Vec<(usize, usize)> = match args.iter().position(|a| a == "--multiscale") {
        Some(i) => args[i + 1]
            .split(',')
            .map(|l| {
                let f: Vec<&str> = l.split(':').map(str::trim).collect();
                assert!(f.len() >= 2, "--multiscale k:n[:w],k:n[:w],...");
                weights.push(f.get(2).map_or(1.0, |w| w.parse().unwrap()));
                (f[0].parse().unwrap(), f[1].parse().unwrap())
            })
            .collect(),
        None => {
            weights.push(1.0);
            vec![(0, iters)]
        }
    };
    let cumulative = args.iter().any(|a| a == "--cumulative") || p.absolute;
    let tol_mm = flag("--tol-mm", 0.0);
    let only_last = !args.iter().any(|a| a == "--cumulative");
    let mut mm_levels: Vec<usize> = levels.iter().map(|l| l.0).collect();
    mm_levels.sort_unstable_by(|a, b| b.cmp(a));
    mm_levels.dedup();
    if !mm_levels.contains(&0) {
        mm_levels.push(0);
    }
    let param_smooth = flag("--param-smooth", 0.0);
    let min_growth = args.iter().position(|a| a == "--min-growth").map(|i| args[i + 1].parse::<f64>().unwrap());
    let carry = args.iter().any(|a| a == "--carry");
    let growth_band = flag("--growth-band", 0.0);
    let field_band = flag("--field-band", 0.3);
    let max_step = flag("--max-step", 0.05);
    let ftol = flag("--ftol", 0.0);
    let mut hist_f: Vec<f64> = Vec::new();
    if levels.len() > 1 || levels[0].0 > 0 {
        assert!(p.hp == 0 && !p.chamfer, "--multiscale replaces --hp / --chamfer");
        println!("multi-resolution loss, levels (smoothing iterations : L-BFGS iterations) {levels:?}");
    }
    p.set_level(levels[0].0);

    // Equilibrium of the measured growth, from the checkpoint state.
    p.sim.set_state(&xs, &p.v0.clone(), &p.growth(&theta));
    let ray = p.rayleigh(&xs);
    let sigma = flag("--shift", 1e-3) * ray.abs();
    p.sigma = sigma;
    let mut x = p.equilibrium(&theta, &xs, flag("--relax0", relax as f64) as usize);
    // Cumulative loss: weights w_s / L_s(start) for every level, first level active.
    let mut norm_levels: Vec<(usize, f64, Vec<[f64; 3]>)> = Vec::new();
    if cumulative {
        for (&(k, _), &w) in levels.iter().zip(&weights) {
            let t_hat = p.level_target(k);
            let l0 = p.level_loss(&x, k, &t_hat).0;
            println!("level k={k}: start loss {l0:.4e}, weight {w}");
            norm_levels.push((k, w / l0, t_hat));
        }
        p.active = norm_levels[..1].to_vec();
    }
    p.report_mm("start", &x, &mm_levels);
    let (mut ls, mut lr, mut gx) = p.loss(&x, &theta);
    println!(
        "start: shape loss {ls:.6e}, reg {lr:.3e}; K Rayleigh {ray:.3e}, shift {sigma:.3e}; |F| {:.3e} -> {:.3e}",
        p.last_residual.0, p.last_residual.1
    );
    save_rows(out.join("x_0.npy"), &x);
    if let Some(path) = opt("--pull-target") {
        // Branch test: equilibrium pulled onto the target (surface springs), then released.
        let tgt = npy::read(&path).unwrap().rows3().unwrap();
        let k = flag("--pull-k", 1.0);
        // Continuation: the spring anchors move from x to the target in `stages` Newton solves
        // (no damped steps: explicit dynamics is unstable with stiff springs).
        let stages = flag("--pull-stages", 10.0) as usize;
        // Per-node weight of the springs (0: target not imposed there), e.g. around crushed elements.
        let pull_mask: Option<Vec<f64>> = opt("--pull-mask").map(|m| read(&m));
        if let Some(np_) = opt("--pull-normals") {
            // Springs along the target normals only: no vertex correspondence imposed tangentially.
            let nr = npy::read(&np_).unwrap().rows3().unwrap();
            p.sim.set_pull_normals(Some(&nr));
            println!("pull along the target normals only ({} nodes)", nr.len());
        }
        let x_start = x.clone();
        for s in 1..=stages {
            let f = s as f64 / stages as f64;
            let anchors: Vec<[f64; 3]> =
                tgt.iter().zip(&x_start).map(|(t, a)| [0, 1, 2].map(|d| a[d] + f * (t[d] - a[d]))).collect();
            match &pull_mask {
                Some(m) => p.sim.set_pull_weighted(&anchors, &m.iter().map(|w| w * k).collect::<Vec<_>>()),
                None => p.sim.set_pull(Some((&anchors, k))),
            }
            let xs_ = p.equilibrium(&theta, &x, 0);
            let inv = p.n_inverted(&xs_, &theta);
            println!("pull stage {s}/{stages}: |F| {:.3e} -> {:.3e}, inverted {inv}", p.last_residual.0, p.last_residual.1);
            if inv > 0 || p.last_residual.1 > 1e-2 * p.last_residual.0 {
                println!("pull stopped at stage {}: last admissible, converged state kept", s - 1);
                break;
            }
            x = xs_;
            p.report_mm(&format!("stage {s}"), &x, &[0]);
        }
        println!("pulled (k = {k}): |F| {:.3e} -> {:.3e}", p.last_residual.0, p.last_residual.1);
        p.report_mm("pulled", &x, &mm_levels);
        save_rows(out.join("x_pulled.npy"), &x);
        p.sim.set_pull(None);
        x = p.equilibrium(&theta, &x, relax);
        println!("released: |F| {:.3e} -> {:.3e}", p.last_residual.0, p.last_residual.1);
        p.report_mm("released", &x, &mm_levels);
        save_rows(out.join("x_released.npy"), &x);
        return;
    }
    if let Some(path) = opt("--inverse-eq") {
        // Inverse equilibrium: geometry fixed at x_g (e.g. the pulled state), growth fitted so that
        // x_g is an equilibrium: min R = |F(x_g; theta)|^2 / 2 + reg, dR/ds = F . dF/ds (one VJP,
        // no equilibrium solve). Then released from x_g with the fitted growth.
        let xg = npy::read(&path).unwrap().rows3().unwrap();
        let v0 = p.v0.clone();
        if let Some(cp) = opt("--core-override") {
            // Remodelled core: growth tensor U per tetrahedron (tools/absorb_core.py).
            let u = read(&cp);
            assert_eq!(u.len(), 6 * p.tets.len(), "--core-override needs [n_tets, 6]");
            p.sim.set_core_override(Some(&u));
            p.ad.set_core_override(Some(&u));
            println!("core growth replaced per tetrahedron ({})", cp.display());
        }
        let energy = args.iter().any(|a| a == "--energy-norm");
        let fields = opt("--inverse-fields").map_or(String::new(), |f| f.to_string_lossy().into_owned());
        let (with_c, with_k) = (fields.contains('c'), fields.contains('k'));
        let na = theta.len();
        // z = [a per vertex; correction of the thickening c (if c); core correction delta (if k)].
        let (oc, ok) = (na, na * (1 + with_c as usize));
        let nz = na * (1 + with_c as usize + with_k as usize);
        let per_tet = |p: &Problem, d: &[f64]| -> Vec<f64> {
            p.tets.iter().map(|t| 0.25 * t.iter().map(|&n| d[p.owner[n as usize]]).sum::<f64>()).collect()
        };
        let reg_d = |p: &Problem, d: &[f64]| 0.5 * p.beta * p.edges.iter().map(|[i, j]| (d[*i] - d[*j]).powi(2)).sum::<f64>();
        let residual = |p: &mut Problem, z: &[f64]| -> (f64, Vec<[f64; 3]>) {
            let s = p.growth(&z[..na]);
            let mut rd = 0.0;
            if with_c {
                let d = per_tet(p, &z[oc..oc + na]);
                p.sim.set_radial_delta(Some(&d));
                p.ad.set_radial_delta(Some(&d));
                rd += reg_d(p, &z[oc..oc + na]);
            }
            if with_k {
                let d = per_tet(p, &z[ok..ok + na]);
                p.sim.set_core_delta(Some(&d));
                p.ad.set_core_delta(Some(&d));
                rd += reg_d(p, &z[ok..ok + na]);
            }
            p.sim.set_state(&xg, &v0, &s);
            p.sim.set_fixed_growth(Some(&s));
            let f = p.sim.total_force(&xg);
            if energy {
                // Energy norm F.K^-1 F / 2: u = K^-1 F is the linearised displacement on release;
                // its VJP u . dF/dtheta is the Gauss-Newton gradient (dK/dtheta term dropped).
                p.sim.refresh_contact();
                let (u, it, res) = p.solve(&xg, &f, p.sigma, cg, true);
                if std::env::var_os("CG_DEBUG").is_some() {
                    println!("cg: {it} iterations, relative residual {res:.3e}");
                }
                let e: f64 = u.iter().zip(&f).map(|(a, b)| a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).sum();
                return (0.5 * e + p.reg(&z[..na]) + rd, u);
            }
            (0.5 * f.iter().flatten().map(|v| v * v).sum::<f64>() + p.reg(&z[..na]) + rd, f)
        };
        let grad = |p: &Problem, z: &[f64], f: &[[f64; 3]]| -> Vec<f64> {
            let th = &z[..na];
            let (gs, gc, gk) = p.vjp_fields(&xg, &p.growth(th), f, with_c, with_k);
            let mut g = vec![0.0; z.len()];
            for ((t, gm), gse) in p.tets.iter().zip(&p.gm).zip(&gs) {
                for &n in t {
                    g[p.owner[n as usize]] += gse * gm * 0.25;
                }
            }
            for (off, gf) in [(oc, &gc), (ok, &gk)] {
                for (t, ge) in p.tets.iter().zip(gf.iter()) {
                    for &n in t {
                        g[off + p.owner[n as usize]] += ge * 0.25;
                    }
                }
            }
            for [i, j] in &p.edges {
                let d = p.beta * (th[*i] - p.theta0[*i] - th[*j] + p.theta0[*j]);
                g[*i] += d;
                g[*j] -= d;
                for (on, off) in [(with_c, oc), (with_k, ok)] {
                    if on {
                        let d = p.beta * (z[off + *i] - z[off + *j]);
                        g[off + *i] += d;
                        g[off + *j] -= d;
                    }
                }
            }
            g
        };
        let mut theta: Vec<f64> = theta.iter().copied().chain(std::iter::repeat(0.0)).take(nz).collect();
        if let Some(zp) = opt("--inverse-z0") {
            theta = read(&zp);
            assert_eq!(theta.len(), nz, "--inverse-z0 does not match --inverse-fields");
        }
        let surf_norm = |f: &[[f64; 3]]| f[..ns].iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        let full_norm = |f: &[[f64; 3]]| f.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        p.sim.set_state(&xg, &v0, &p.growth(&theta));
        p.sim.refresh_contact();
        if energy {
            let s_ = p.growth(&theta);
            p.sim.set_state(&xg, &v0, &s_);
            p.sim.set_fixed_growth(Some(&s_));
            p.build_preconditioner(&xg);
        }
        if let Some(list) = opt("--deflate") {
            // Unstable modes (e.g. Ritz vectors of negative eigenvalues), Gram-Schmidt orthonormalised.
            let dot = |a: &[[f64; 3]], b: &[[f64; 3]]| a.iter().zip(b).map(|(p, q)| p[0] * q[0] + p[1] * q[1] + p[2] * q[2]).sum::<f64>();
            for path in list.to_string_lossy().split(',') {
                let mut v = npy::read(path).unwrap().rows3().unwrap();
                for d in &p.deflate {
                    let c = dot(d, &v);
                    v.iter_mut().zip(d).for_each(|(a, b)| (0..3).for_each(|k| a[k] -= c * b[k]));
                }
                let nv = dot(&v, &v).sqrt();
                v.iter_mut().for_each(|a| a.iter_mut().for_each(|c| *c /= nv));
                p.deflate.push(v);
            }
            println!("CG deflation: {} modes", p.deflate.len());
        }
        let u_mm = |p: &Problem, u: &[[f64; 3]]| p.mm * (u[..ns].iter().flatten().map(|v| v * v).sum::<f64>() / ns as f64).sqrt();
        let (mut r, mut f) = residual(&mut p, &theta);
        let (r_start, f_start) = (r, f.clone());
        save_rows(out.join("force_start.npy"), &f);
        println!("inverse-eq start: R {r:.6e}, |F| {:.4e} (surface {:.4e})", full_norm(&f), surf_norm(&f));
        if energy {
            println!("energy norm: predicted release displacement (surface rms) {:.3} mm", u_mm(&p, &f));
        }
        let mut g = grad(&p, &theta, &f);
        {
            // Directional check of the gradient: R(theta - e g/|g|_inf) - R(theta) vs -e |g|^2/|g|_inf.
            let gi = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let g2: f64 = g.iter().map(|v| v * v).sum();
            for e in [1e-2, 1e-3, 1e-4, 1e-5] {
                let tp: Vec<f64> = theta.iter().zip(&g).map(|(a, d)| a - e * d / gi).collect();
                let tm: Vec<f64> = theta.iter().zip(&g).map(|(a, d)| a + e * d / gi).collect();
                let (rp, rm) = (residual(&mut p, &tp).0, residual(&mut p, &tm).0);
                println!(
                    "gradient check e {e:.0e}: R(-) - R {:+.4e}, R(+) - R {:+.4e}, predicted -/+ {:.4e}, central ratio {:.4}",
                    rp - r,
                    rm - r,
                    e * g2 / gi,
                    (rm - rp) / (2.0 * e * g2 / gi)
                );
            }
        }
        let mut hist: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        for it in 1..=iters {
            let mut q = g.clone();
            let mut al = Vec::new();
            for (s, y) in hist.iter().rev() {
                let rho = 1.0 / y.iter().zip(s).map(|(a, b)| a * b).sum::<f64>();
                let a = rho * s.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>();
                q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
                al.push((a, rho));
            }
            let gamma = match hist.last() {
                Some((s, y)) => s.iter().zip(y).map(|(a, b)| a * b).sum::<f64>() / y.iter().map(|v| v * v).sum::<f64>(),
                None => max_step / g.iter().fold(0.0f64, |m, v| m.max(v.abs())),
            };
            q.iter_mut().for_each(|v| *v *= gamma);
            for ((s, y), (a, rho)) in hist.iter().zip(al.iter().rev()) {
                let b = rho * y.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>();
                q.iter_mut().zip(s).for_each(|(qi, si)| *qi += (a - b) * si);
            }
            let mut dir: Vec<f64> = q.iter().map(|v| -v).collect();
            let dmax = dir.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if dmax > max_step {
                dir.iter_mut().for_each(|v| *v *= max_step / dmax);
            }
            let slope: f64 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
            if slope >= 0.0 {
                hist.clear();
                println!("iter {it}: not a descent direction, history reset");
                continue;
            }
            let mut step = 1.0;
            let mut accepted = None;
            for _ in 0..10 {
                let mut tn: Vec<f64> = theta.iter().zip(&dir).map(|(a, d)| a + step * d).collect();
                if let Some(lo) = min_growth {
                    tn[..na].iter_mut().for_each(|v| *v = v.max(lo));
                }
                let (rn, fnew) = residual(&mut p, &tn);
                if rn.is_finite() && rn <= r + 1e-4 * step * slope {
                    accepted = Some((tn, rn, fnew));
                    break;
                }
                step *= 0.5;
            }
            let Some((tn, rn, fnew)) = accepted else {
                println!("iter {it}: no decrease, stopping");
                break;
            };
            let gn = grad(&p, &tn, &fnew);
            let s: Vec<f64> = tn.iter().zip(&theta).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = gn.iter().zip(&g).map(|(a, b)| a - b).collect();
            if s.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() > 0.0 {
                hist.push((s, y));
                if hist.len() > 10 {
                    hist.remove(0);
                }
            }
            (theta, r, f, g) = (tn, rn, fnew, gn);
            if it % 10 == 0 || it == 1 || energy {
                let range = |v: &[f64]| v.iter().fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
                let (lo, hi) = range(&theta[..na]);
                let mut c = String::new();
                if with_c {
                    let (l, h) = range(&theta[oc..oc + na]);
                    c += &format!(", dc in [{l:.3}, {h:.3}]");
                }
                if with_k {
                    let (l, h) = range(&theta[ok..ok + na]);
                    c += &format!(", dk in [{l:.3}, {h:.3}]");
                }
                println!(
                    "iter {it}: R {r:.6e} ({:.4} of start), |F| {:.4e} (surface {:.4e}), a in [{lo:.3}, {hi:.3}]{c}, step {step}",
                    r / r_start,
                    full_norm(&f),
                    surf_norm(&f)
                );
                if energy {
                    println!("iter {it}: predicted release displacement {:.3} mm", u_mm(&p, &f));
                }
            }
        }
        let fs = surf_norm(&f_start);
        println!("inverse-eq end: R {:.4} of start, surface |F| {:.4} of start", r / r_start, surf_norm(&f) / fs);
        save_rows(out.join("force_end.npy"), &f);
        npy::write_f64(out.join("theta_inv.npy"), &[theta.len()], &theta).unwrap();
        let th = theta[..na].to_vec();
        if let Some(nl) = args.iter().position(|a| a == "--lanczos").map(|i| args[i + 1].parse::<usize>().unwrap()) {
            // Extreme eigenpairs of K at x_g with the fitted fields: Lanczos with full
            // reorthogonalisation, Ritz values from the tridiagonal matrix (cyclic Jacobi).
            let s_ = p.growth(&th);
            p.sim.set_state(&xg, &v0, &s_);
            p.sim.set_fixed_growth(Some(&s_));
            p.sim.refresh_contact();
            let dot = |a: &[[f64; 3]], b: &[[f64; 3]]| a.iter().zip(b).map(|(p, q)| p[0] * q[0] + p[1] * q[1] + p[2] * q[2]).sum::<f64>();
            let axpy = |y: &mut [[f64; 3]], a: f64, z: &[[f64; 3]]| y.iter_mut().zip(z).for_each(|(p, q)| (0..3).for_each(|d| p[d] += a * q[d]));
            let mut q: Vec<[f64; 3]> = (0..xg.len()).map(|i| [0, 1, 2].map(|d| (((i * 7 + d * 13) % 17) as f64 / 8.0) - 1.0)).collect();
            let nq = dot(&q, &q).sqrt();
            q.iter_mut().for_each(|r| r.iter_mut().for_each(|v| *v /= nq));
            let (mut basis, mut al, mut be) = (vec![q], Vec::<f64>::new(), Vec::<f64>::new());
            for j in 0..nl {
                let mut w = p.k_mul(&xg, &basis[j]);
                let a = dot(&basis[j], &w);
                al.push(a);
                for _ in 0..2 {
                    for b in &basis {
                        let c = dot(b, &w);
                        axpy(&mut w, -c, b);
                    }
                }
                let b = dot(&w, &w).sqrt();
                if j + 1 == nl || b < 1e-14 {
                    break;
                }
                be.push(b);
                w.iter_mut().for_each(|r| r.iter_mut().for_each(|v| *v /= b));
                basis.push(w);
            }
            let m = al.len();
            let mut t = vec![vec![0.0; m]; m];
            let mut s = vec![vec![0.0; m]; m];
            for i in 0..m {
                t[i][i] = al[i];
                s[i][i] = 1.0;
                if i + 1 < m {
                    t[i][i + 1] = be[i];
                    t[i + 1][i] = be[i];
                }
            }
            for _sweep in 0..100 {
                let off: f64 = (0..m).flat_map(|i| (0..m).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| t[i][j] * t[i][j]).sum();
                if off < 1e-30 {
                    break;
                }
                for pp in 0..m {
                    for qq in pp + 1..m {
                        if t[pp][qq].abs() < 1e-300 {
                            continue;
                        }
                        let theta = (t[qq][qq] - t[pp][pp]) / (2.0 * t[pp][qq]);
                        let tt = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                        let tt = if theta == 0.0 { 1.0 } else { tt };
                        let (c, sn) = (1.0 / (tt * tt + 1.0).sqrt(), tt / (tt * tt + 1.0).sqrt());
                        for k in 0..m {
                            let (a, b) = (t[k][pp], t[k][qq]);
                            t[k][pp] = c * a - sn * b;
                            t[k][qq] = sn * a + c * b;
                        }
                        for k in 0..m {
                            let (a, b) = (t[pp][k], t[qq][k]);
                            t[pp][k] = c * a - sn * b;
                            t[qq][k] = sn * a + c * b;
                        }
                        for k in 0..m {
                            let (a, b) = (s[k][pp], s[k][qq]);
                            s[k][pp] = c * a - sn * b;
                            s[k][qq] = sn * a + c * b;
                        }
                    }
                }
            }
            let mut order: Vec<usize> = (0..m).collect();
            order.sort_by(|&a, &b| t[a][a].partial_cmp(&t[b][b]).unwrap());
            let ev: Vec<f64> = order.iter().map(|&i| t[i][i]).collect();
            println!("lanczos {m}: smallest Ritz values {:?}", ev.iter().take(6).map(|v| format!("{v:.4e}")).collect::<Vec<_>>());
            println!("lanczos {m}: largest Ritz value {:.4e}", ev[m - 1]);
            let n_neg = ev.iter().filter(|v| **v < 0.0).count();
            for (r, &col) in order.iter().take(n_neg.clamp(3, 16)).enumerate() {
                let mut y = vec![[0.0; 3]; xg.len()];
                for (k, b) in basis.iter().enumerate() {
                    axpy(&mut y, s[k][col], b);
                }
                let ky = p.k_mul(&xg, &y);
                let lam = t[col][col];
                let res: f64 = ky.iter().zip(&y).map(|(a, b)| (0..3).map(|d| (a[d] - lam * b[d]).powi(2)).sum::<f64>()).sum::<f64>().sqrt();
                println!("ritz {r}: lambda {lam:.4e}, residual |Ky - lambda y| {res:.3e}");
                save_rows(out.join(format!("ritz{r}.npy")), &y);
            }
        }
        if let Some(rp) = opt("--fold-ref") {
            // Curvature of the energy along the fold direction v = x_g - x_ref (x_ref: unfolded
            // equilibrium), at x_g with the fitted fields: v.Kv / v.v vs a generic direction.
            let xr = npy::read(&rp).unwrap().rows3().unwrap();
            let v: Vec<[f64; 3]> = xg.iter().zip(&xr).map(|(a, b)| [0, 1, 2].map(|d| a[d] - b[d])).collect();
            let s_ = p.growth(&th);
            p.sim.set_state(&xg, &v0, &s_);
            p.sim.set_fixed_growth(Some(&s_));
            p.sim.refresh_contact();
            let kv = p.k_mul(&xg, &v);
            let dot = |a: &[[f64; 3]], b: &[[f64; 3]]| a.iter().zip(b).map(|(p, q)| p[0] * q[0] + p[1] * q[1] + p[2] * q[2]).sum::<f64>();
            println!("fold-direction Rayleigh v.Kv/v.v {:.4e} (generic direction {:.4e})", dot(&v, &kv) / dot(&v, &v), p.rayleigh(&xg));
        }
        let rr = flag("--release-relax", relax as f64) as usize;
        x = p.equilibrium(&th, &xg, rr);
        println!("released with fitted growth: |F| {:.3e} -> {:.3e}", p.last_residual.0, p.last_residual.1);
        p.report_mm("released", &x, &mm_levels);
        println!("inverted elements: {}", p.n_inverted(&x, &th));
        save_rows(out.join("x_released.npy"), &x);
        return;
    }
    if args.iter().any(|a| a == "--diag") {
        // Loss gradient w.r.t. x vs central differences of the loss (no equilibrium).
        let d: Vec<[f64; 3]> = (0..x.len()).map(|i| [0, 1, 2].map(|c| (((i * 7 + c * 13) % 17) as f64 / 8.0) - 1.0)).collect();
        let an: f64 = gx.iter().zip(&d).map(|(a, b)| a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).sum();
        for e in [1e-4, 1e-5] {
            let xp: Vec<[f64; 3]> = x.iter().zip(&d).map(|(a, b)| [0, 1, 2].map(|c| a[c] + e * b[c])).collect();
            let xm: Vec<[f64; 3]> = x.iter().zip(&d).map(|(a, b)| [0, 1, 2].map(|c| a[c] - e * b[c])).collect();
            let fd = (p.loss(&xp, &theta).0 - p.loss(&xm, &theta).0) / (2.0 * e);
            println!("diag loss gradient eps {e:.0e}: finite difference {fd:.6e}, analytic {an:.6e}, ratio {:.6}", fd / an);
        }
        for _ in 0..2 {
            let tk = Instant::now();
            let _ = p.k_mul(&x, &gx);
            let t_ad = tk.elapsed().as_secs_f64();
            let tk = Instant::now();
            let _ = p.k_mul_fd(&x, &gx);
            println!("diag K product: exact {t_ad:.2}s, finite differences {:.2}s", tk.elapsed().as_secs_f64());
        }
        let dot = |a: &[[f64; 3]], b: &[[f64; 3]]| a.iter().zip(b).map(|(p, q)| p[0] * q[0] + p[1] * q[1] + p[2] * q[2]).sum::<f64>();
        let rel = |a: &[[f64; 3]], b: &[[f64; 3]]| {
            let d: Vec<[f64; 3]> = a.iter().zip(b).map(|(p, q)| [0, 1, 2].map(|k| p[k] - q[k])).collect();
            (dot(&d, &d) / dot(b, b)).sqrt()
        };
        let v = &gx;
        let base = p.k_mul(&x, v);
        println!("diag K determinism: {:.3e}", rel(&p.k_mul(&x, v), &base));
        let v2: Vec<[f64; 3]> = v.iter().map(|r| r.map(|c| 2.0 * c)).collect();
        let k2: Vec<[f64; 3]> = p.k_mul(&x, &v2).iter().map(|r| r.map(|c| 0.5 * c)).collect();
        println!("diag K linearity K(2v)/2 vs K(v): {:.3e}", rel(&k2, &base));
        for e in [1e-3, 1e-4, 1e-5] {
            let k = p.k_mul_fd_eps(&x, v, e);
            println!("diag f32 finite-difference K, eps {e:.0e}: rel diff to exact {:.3e}", rel(&k, &base));
        }
    }

    let tg = Instant::now();
    let (mut g, it, res) = p.gradient(&x, &theta, &gx, sigma, cg);
    let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!(
        "gradient: CG {it} it (residual {res:.1e}), |g| {gnorm:.3e}, {:.1}s",
        tg.elapsed().as_secs_f64()
    );

    if args.iter().any(|a| a == "--diag") {
        // Link-by-link test of the adjoint at the start equilibrium.
        let dot = |a: &[[f64; 3]], b: &[[f64; 3]]| a.iter().zip(b).map(|(p, q)| p[0] * q[0] + p[1] * q[1] + p[2] * q[2]).sum::<f64>();
        let (lam, it, res) = p.adjoint(&x, &gx, sigma, cg);
        let klam = p.k_mul(&x, &lam);
        let kres: Vec<[f64; 3]> = klam.iter().zip(&lam).zip(&gx).map(|((k, l), b)| [0, 1, 2].map(|d| k[d] + sigma * l[d] - b[d])).collect();
        println!("diag adjoint: CG {it} (res {res:.1e}); |K lam - b| / |b| = {:.3e} (with shift)", (dot(&kres, &kres) / dot(&gx, &gx)).sqrt());
        // Symmetry of K.
        let n = x.len();
        let v: Vec<[f64; 3]> = (0..n).map(|i| [0, 1, 2].map(|d| ((i * 7 + d * 13) % 17) as f64 / 8.0 - 1.0)).collect();
        let w: Vec<[f64; 3]> = (0..n).map(|i| [0, 1, 2].map(|d| ((i * 11 + d * 5) % 19) as f64 / 9.0 - 1.0)).collect();
        let (vkw, wkv) = (dot(&v, &p.k_mul(&x, &w)), dot(&w, &p.k_mul(&x, &v)));
        println!("diag symmetry: v.Kw {vkw:.6e}, w.Kv {wkv:.6e}, rel diff {:.2e}", (vkw - wkv).abs() / vkw.abs().max(wkv.abs()));
        let sv: Vec<[f64; 3]> = v.iter().enumerate().map(|(i, a)| if i < ns { *a } else { [0.0; 3] }).collect();
        let sw: Vec<[f64; 3]> = w.iter().enumerate().map(|(i, a)| if i < ns { *a } else { [0.0; 3] }).collect();
        let (vkw, wkv) = (dot(&sv, &p.k_mul(&x, &sw)), dot(&sw, &p.k_mul(&x, &sv)));
        println!("diag symmetry (surface only): v.Kw {vkw:.6e}, w.Kv {wkv:.6e}, rel diff {:.2e}", (vkw - wkv).abs() / vkw.abs().max(wkv.abs()));
        // VJP lambda . dF/ds: autodiff vs central difference of the total force.
        let s0 = p.growth(&theta);
        let gs = p.vjp_growth(&x, &s0, &lam);
        let gmax = g.iter().fold(0.0f64, |m, a| m.max(a.abs()));
        let dth: Vec<f64> = g.iter().map(|a| -a / gmax).collect();
        let ds = p.growth(&dth);
        let an: f64 = gs.iter().zip(&ds).map(|(a, b)| a * b).sum();
        for eps in [1e-3, 3e-3, 1e-2] {
            let sp: Vec<f64> = s0.iter().zip(&ds).map(|(a, b)| a + eps * b).collect();
            let sm: Vec<f64> = s0.iter().zip(&ds).map(|(a, b)| a - eps * b).collect();
            p.sim.set_state(&x, &p.v0.clone(), &sp);
            let fp = p.sim.total_force(&x);
            p.sim.set_state(&x, &p.v0.clone(), &sm);
            let fm = p.sim.total_force(&x);
            let fd = (dot(&lam, &fp) - dot(&lam, &fm)) / (2.0 * eps);
            println!("diag vjp eps {eps}: finite difference {fd:.5e}, autodiff {an:.5e}, ratio {:.4}", fd / an);
        }
        // Linearised response: dx = K^-1 dF/ds ds vs equilibrium difference; dL = gx . dx.
        p.sim.set_state(&x, &p.v0.clone(), &s0);
        let eps = 3e-3;
        let sp: Vec<f64> = s0.iter().zip(&ds).map(|(a, b)| a + eps * b).collect();
        let sm: Vec<f64> = s0.iter().zip(&ds).map(|(a, b)| a - eps * b).collect();
        p.sim.set_state(&x, &p.v0.clone(), &sp);
        let fp = p.sim.total_force(&x);
        p.sim.set_state(&x, &p.v0.clone(), &sm);
        let fm = p.sim.total_force(&x);
        let rhs: Vec<[f64; 3]> = fp.iter().zip(&fm).map(|(a, b)| [0, 1, 2].map(|d| (a[d] - b[d]) / (2.0 * eps))).collect();
        p.sim.set_state(&x, &p.v0.clone(), &s0);
        p.sim.refresh_contact();
        let (dx, it2, res2) = p.adjoint(&x, &rhs, sigma, cg);
        let tp: Vec<f64> = theta.iter().zip(&dth).map(|(a, b)| a + eps * b).collect();
        let tm: Vec<f64> = theta.iter().zip(&dth).map(|(a, b)| a - eps * b).collect();
        let xp = p.equilibrium(&tp, &x, 2 * relax);
        let rp = p.last_residual;
        let xm = p.equilibrium(&tm, &x, 2 * relax);
        let dxe: Vec<[f64; 3]> = xp.iter().zip(&xm).map(|(a, b)| [0, 1, 2].map(|d| (a[d] - b[d]) / (2.0 * eps))).collect();
        let diff: Vec<[f64; 3]> = dxe.iter().zip(&dx).map(|(a, b)| [0, 1, 2].map(|d| a[d] - b[d])).collect();
        println!(
            "diag response: CG {it2} (res {res2:.1e}); |dx_eq| {:.4e}, |dx_lin| {:.4e}, |diff|/|dx_eq| {:.3e}; gx.dx_eq {:.5e}, gx.dx_lin {:.5e}; residuals {rp:?} {:?}",
            dot(&dxe, &dxe).sqrt(), dot(&dx, &dx).sqrt(), (dot(&diff, &diff) / dot(&dxe, &dxe)).sqrt(),
            dot(&gx, &dxe), dot(&gx, &dx), p.last_residual
        );
        let surf = |a: &[[f64; 3]]| a[..ns].iter().map(|r| r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sum::<f64>().sqrt();
        println!("diag response surface: |dx_eq| {:.4e}, |dx_lin| {:.4e}, |diff| {:.4e}", surf(&dxe), surf(&dx), surf(&diff));
        return;
    }

    if check {
        // Directional finite difference along -g, normalised to max |d| = 0.01 (x --check-scale).
        let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let dir: Vec<f64> = g.iter().map(|v| -0.01 * flag("--check-scale", 1.0) * v / gmax).collect();
        for h in [0.5, 1.0, 2.0] {
            let tp: Vec<f64> = theta.iter().zip(&dir).map(|(a, d)| a + h * d).collect();
            let tm: Vec<f64> = theta.iter().zip(&dir).map(|(a, d)| a - h * d).collect();
            let xp = p.equilibrium(&tp, &x, 2 * relax);
            let lp = {
                let (a, b, _) = p.loss(&xp, &tp);
                a + b
            };
            let xm = p.equilibrium(&tm, &x, 2 * relax);
            let lm = {
                let (a, b, _) = p.loss(&xm, &tm);
                a + b
            };
            let fd = (lp - lm) / (2.0 * h);
            let an: f64 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
            println!(
                "check h = {h}: finite difference {fd:.4e}, adjoint {an:.4e}, ratio {:.3}; |F| {:.2e} -> {:.2e}",
                fd / an,
                p.last_residual.0,
                p.last_residual.1
            );
        }
        return;
    }

    if levels.len() > 1 {
        let report: Vec<String> = levels
            .iter()
            .map(|&(l, _)| format!("k={l}: {:.4e}", p.level_loss(&x, l, &p.level_target(l)).0))
            .collect();
        println!("start loss per level: {}", report.join(", "));
    }
    // L-BFGS, one run per resolution level (history reset when the loss changes).
    let m = 5;
    let mut hist: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut k = 0;
    'levels: for (li, &(level, level_iters)) in levels.iter().enumerate() {
    if cumulative {
        // --absolute without --cumulative: only the current level is in the loss.
        p.active = if only_last { norm_levels[li..=li].to_vec() } else { norm_levels[..=li].to_vec() };
    }
    if li > 0 {
        p.set_level(level);
        hist.clear();
        (ls, lr, gx) = p.loss(&x, &theta);
        let (gn, _, _) = p.gradient(&x, &theta, &gx, sigma, cg);
        g = gn;
    }
    // Parameter resolution of this level: theta = theta + S^m z (m = 0: theta itself).
    let pm = (param_smooth * level as f64).round() as usize;
    let mut gz = p.smooth_scalar(&g, pm, true);
    if pm > 0 {
        println!("level {li}: growth corrections smoothed by S^{pm}");
    }
    if levels.len() > 1 {
        println!("level {li} (smoothing {level}): loss {ls:.6e}, reg {lr:.3e}");
    }
    for _ in 0..level_iters {
        if !gz.iter().any(|v| *v != 0.0) || !gz.iter().all(|v| v.is_finite()) {
            // CG made no progress (indefinite K during buckling): no usable gradient.
            println!("level {li}: no usable gradient, stopping this level");
            break;
        }
        k += 1;
        let ti = Instant::now();
        let mut q = gz.clone();
        let mut alphas = Vec::new();
        for (s, y) in hist.iter().rev() {
            let rho = 1.0 / y.iter().zip(s).map(|(a, b)| a * b).sum::<f64>();
            let a = rho * s.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>();
            for (qi, yi) in q.iter_mut().zip(y) {
                *qi -= a * yi;
            }
            alphas.push((a, rho));
        }
        if gd {
            hist.clear(); // normalised steepest descent: max |step| = gd_step
        }
        let gamma = match hist.last() {
            Some((s, y)) => s.iter().zip(y).map(|(a, b)| a * b).sum::<f64>() / y.iter().map(|v| v * v).sum::<f64>(),
            None => gd_step / gz.iter().fold(0.0f64, |m, v| m.max(v.abs())),
        };
        for qi in q.iter_mut() {
            *qi *= gamma;
        }
        for ((s, y), (a, rho)) in hist.iter().zip(alphas.iter().rev()) {
            let b = rho * y.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>();
            for (qi, si) in q.iter_mut().zip(s) {
                *qi += (a - b) * si;
            }
        }
        let dir_z: Vec<f64> = q.iter().map(|v| -v).collect();
        let slope: f64 = gz.iter().zip(&dir_z).map(|(a, b)| a * b).sum();
        let mut dir = p.smooth_scalar(&dir_z, pm, false);
        // Trust region on the growth: max |step 1| <= --max-step.
        let dmax = dir.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let cap = if dmax > max_step { max_step / dmax } else { 1.0 };
        if cap < 1.0 {
            for v in dir.iter_mut() {
                *v *= cap;
            }
        }
        let slope = slope * cap;
        // Reference value relaxed exactly like the trial points (same start, same steps),
        // so that residual drift of the dynamics cancels in the comparison.
        let x_ref = p.equilibrium(&theta, &x, p.relax);
        let f0 = {
            let (a, b, _) = p.loss(&x_ref, &theta);
            a + b
        };
        println!(
            "iter {k}: reference {f0:.6e} (drift {:+.2e} from last), expected decrease {:.2e} at step 1",
            f0 - (ls + lr),
            -slope
        );
        let mut step = 1.0;
        let (mut tn, mut xn, mut fn_, mut parts) = (theta.clone(), x.clone(), f0, (ls, lr, gx.clone()));
        for _ in 0..6 {
            tn = theta.iter().zip(&dir).map(|(a, d)| a + step * d).collect();
            if let Some(lo) = min_growth {
                for v in tn[..p.na].iter_mut() {
                    *v = v.max(lo);
                }
            }
            for v in tn[p.na..].iter_mut() {
                *v = v.clamp(-field_band, field_band);
            }
            if growth_band > 0.0 {
                // Growth kept within +-band x the measured value (theta0).
                for (v, m) in tn.iter_mut().zip(&theta0) {
                    *v = v.clamp(m * (1.0 - growth_band), m * (1.0 + growth_band));
                }
            }
            xn = p.equilibrium(&tn, &x, p.relax);
            parts = p.loss(&xn, &tn);
            fn_ = parts.0 + parts.1;
            let finite = fn_.is_finite() && xn.iter().flatten().all(|v| v.is_finite());
            if !finite || p.n_inverted(&xn, &tn) > 0 {
                fn_ = f64::INFINITY;
            } else if fn_ <= f0 + 1e-4 * step * slope {
                break;
            }
            step *= 0.5;
        }
        if !(fn_ <= f0) {
            println!("iter {k}: no decrease, stopping");
            break;
        }
        let (gn, it, res) = p.gradient(&xn, &tn, &parts.2, sigma, cg);
        if !gn.iter().all(|v| v.is_finite()) {
            println!("iter {k}: non-finite gradient, stopping (accepted step kept)");
            theta = tn;
            x = xn;
            (ls, lr, gx) = parts;
            break;
        }
        let gzn = p.smooth_scalar(&gn, pm, true);
        let s: Vec<f64> = dir_z.iter().map(|d| step * cap * d).collect();
        let y: Vec<f64> = gzn.iter().zip(&gz).map(|(a, b)| a - b).collect();
        if s.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() > 0.0 {
            hist.push((s, y));
            if hist.len() > m {
                hist.remove(0);
            }
        }
        theta = tn;
        x = xn;
        (ls, lr, gx) = parts;
        g = gn;
        gz = gzn;
        let dth: Vec<f64> = theta.iter().zip(&theta0).map(|(a, b)| a - b).collect();
        let dmax = dth.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let drms = (dth.iter().map(|v| v * v).sum::<f64>() / dth.len() as f64).sqrt();
        println!(
            "iter {k}: shape loss {ls:.6e}, reg {lr:.3e}, step {step}, CG {it} (res {res:.1e}), |theta - theta0| max {dmax:.4} rms {drms:.4}, {:.0}s",
            ti.elapsed().as_secs_f64()
        );
        save_rows(out.join(format!("x_{k}.npy")), &x);
        npy::write_f64(out.join(format!("theta_{k}.npy")), &[theta.len()], &theta).unwrap();
        let dev = p.report_mm(&format!("iter {k}"), &x, &mm_levels);
        let last = dev.last().unwrap();
        let crit = if p.rigid { last.2[0] } else { last.1[0] };
        if tol_mm > 0.0 && crit < tol_mm {
            println!("iter {k}: {} rms deviation below {tol_mm} mm, done", if p.rigid { "rigid" } else { "absolute" });
            break 'levels;
        }
        // Convergence: relative decrease of the loss over the last 5 iterations below --ftol.
        hist_f.push(ls + lr);
        if ftol > 0.0 && hist_f.len() > 5 {
            let old = hist_f[hist_f.len() - 6];
            if (old - (ls + lr)) / old.abs() < ftol {
                println!("iter {k}: converged (relative decrease {:.2e} over 5 iterations < {ftol:.0e})", (old - (ls + lr)) / old.abs());
                break 'levels;
            }
        }
    }
    hist_f.clear();
    }
    if levels.len() > 1 {
        // Final loss at every level, for the report.
        let report: Vec<String> = levels
            .iter()
            .map(|&(l, _)| format!("k={l}: {:.4e}", p.level_loss(&x, l, &p.level_target(l)).0))
            .collect();
        println!("final loss per level: {}", report.join(", "));
    }

    let dev = p.report_mm("final", &x, &mm_levels);
    // Admissibility of the fit: inverted elements at the fitted equilibrium, growth range.
    let s_fit = p.growth(&theta);
    p.sim.set_state(&x, &p.v0.clone(), &s_fit);
    p.sim.set_fixed_growth(Some(&s_fit));
    let n_inv = p.sim.step_recorded().diagnostics.n_inverted;
    let (tmin, tmax) = theta.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    println!("inverted elements at the fitted equilibrium: {n_inv}; fitted a in [{tmin:.3}, {tmax:.3}]");
    let json: Vec<serde_json::Value> = dev
        .iter()
        .map(|(k, a, g, s)| serde_json::json!({ "k": k, "absolute_mm": { "rms": a[0], "p50": a[1], "p95": a[2], "max": a[3] },
                                               "rigid_mm": { "rms": g[0], "p50": g[1], "p95": g[2], "max": g[3] },
                                               "shape_mm": { "rms": s[0], "p50": s[1], "p95": s[2], "max": s[3] } }))
        .collect();
    std::fs::write(out.join("deviation_mm.json"), serde_json::to_string_pretty(&json).unwrap()).unwrap();

    // Chaining outputs: fitted theta into the a table, fitted equilibrium as a restart point.
    npy::write_f64(out.join("theta_final.npy"), &[theta.len()], &theta).unwrap();
    if let Some(path) = write_table {
        let mut a = npy::read(&a_path).unwrap();
        let n = a.shape[1];
        for (node, &o) in p.owner.iter().enumerate() {
            a.data[knot * n + node] = theta[o] / a_scale;
            if carry {
                for later in knot + 1..a.shape[0] {
                    a.data[later * n + node] += (theta[o] - theta0[o]) / a_scale;
                }
            }
        }
        npy::write_f64(&path, &a.shape, &a.data).unwrap();
        println!("table row {knot} (t = {:.4}) written to {}", knots[knot], path.display());
    }
    if let Some(dir) = save_ckpt {
        std::fs::create_dir_all(&dir).unwrap();
        save_rows(dir.join("x.npy"), &x);
        save_rows(dir.join("v.npy"), &p.v0);
        let s = p.growth(&theta);
        npy::write_f64(dir.join("growth_prev.npy"), &[s.len()], &s).unwrap();
        std::fs::write(
            dir.join("state.json"),
            serde_json::json!({ "step": state["step"], "t": t }).to_string(),
        )
        .unwrap();
        println!("restart point written to {}", dir.display());
    }
    if let Some(path) = save_surface {
        vtk::write_surface(&path, &mesh, &x, &[]).unwrap();
    }
}
