//! Per-vertex growth of one week fitted to the atlas by an adjoint at quasi-static equilibrium.
//!
//!     cargo run --release --features autodiff --example adjoint_week -- \
//!         <config.toml> <mesh> <checkpoint dir> <prep dir> <out dir> [--iters N] [--beta B]
//!         [--relax N] [--check] [--hp k | --chamfer] [--gd --step s]
//!         [--theta0-table] [--write-table a.npy] [--save-checkpoint dir] [--save-surface f.vtk]
//!         [--multiscale "k1:n1,k2:n2,..."] [--min-growth a] [--carry]
//!         [--alpha a --measured a_table.npy] [--fi-weight g]
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
//! shrinkage). --carry (with --write-table): the correction theta - theta0 is also added to
//! the later rows of the table, so the next weeks grow from the corrected state instead of
//! jumping back to the measured growth.
//! --alpha a: keeps theta near the measured growth, + a/2 mean((theta - a_mes)^2) / mean(a_mes^2)
//! (a_mes: row of --measured at the checkpoint time, default theta0). --fi-weight g: + g/2
//! ((FI(x) - FI_atlas) / FI_atlas)^2, FI = A / |V|^(2/3) of the surface (prep/faces.npy), so that
//! the fit cannot gain by cutting growth (the folding index of the atlas is imposed).
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
//! Forward: hold s fixed, relax the damped dynamics to equilibrium x*(theta) (warm start).
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
    fn equilibrium(&mut self, theta: &[f64], x0: &[[f64; 3]], steps: usize) -> Vec<[f64; 3]> {
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
            let (delta, _, _) = self.adjoint(&x, &f, self.sigma, 100);
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
            if !accepted || rn < 1e-3 * r0 {
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
        0.5 * self.beta
            * self
                .edges
                .iter()
                .map(|[i, j]| (theta[*i] - self.theta0[*i] - theta[*j] + self.theta0[*j]).powi(2))
                .sum::<f64>()
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
        let (xh, _, r) = normalise(&sm, &self.cortex);
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
        let mut st = self.through_normalisation(&xh, r, &g, ns);
        for _ in 0..k {
            st = self.smooth_t(&st);
        }
        (ls, st)
    }

    /// Normalised smoothed target of level k.
    fn level_target(&self, k: usize) -> Vec<[f64; 3]> {
        let mut sm = self.target.clone();
        for _ in 0..k {
            sm = self.smooth(&sm);
        }
        normalise(&sm, &self.cortex).0
    }

    /// FI = A / |V|^(2/3) of the surface triangles and its gradient w.r.t. the surface vertices.
    fn fi_grad(&self, x: &[[f64; 3]]) -> (f64, Vec<[f64; 3]>) {
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
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
            vol += (a[0] * (b[1] * c[2] - b[2] * c[1])
                + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
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
                * theta
                    .iter()
                    .zip(&self.theta_mes)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
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

    fn k_mul(&self, x: &[[f64; 3]], v: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let vmax = v
            .iter()
            .flat_map(|r| r.iter())
            .fold(0.0f64, |m, a| m.max(a.abs()))
            .max(1e-30);
        let eps = 3e-5 / vmax;
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
                let kv = self.k_mul(x, &v);
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
        let mut r = b.to_vec();
        let mut z = self.precondition(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let b2 = dot(&r, &r);
        let mut rr = b2;
        let mut it = 0;
        let (mut best, mut best_rr) = (lam.clone(), rr);
        while it < iters && rr > 1e-8 * b2 {
            let mut kp = self.k_mul(x, &p);
            axpy(&mut kp, sigma, &p);
            let pkp = dot(&p, &kp);
            if pkp <= 0.0 {
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
        let f = forces_regular(self.ad.precomputed(), &mat, &xv, &st, &Thickness::PerTet(h));
        let l = (f.x * lv.x).sum() + (f.y * lv.y).sum() + (f.z * lv.z).sum();
        let grads = l.backward();
        to_f64(st.grad(&grads).expect("gradient w.r.t. growth"))
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
        let gs = self.vjp_growth(x, &self.growth(theta), &lam);
        let mut g = vec![0.0; theta.len()];
        for ((t, gm), gse) in self.tets.iter().zip(&self.gm).zip(&gs) {
            for &n in t {
                g[self.owner[n as usize]] += gse * gm * 0.25;
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

    fn rayleigh(&self, x: &[[f64; 3]]) -> f64 {
        let v: Vec<[f64; 3]> = (0..x.len())
            .map(|i| [0, 1, 2].map(|d| (((i * 7 + d * 13) % 17) as f64 / 8.0) - 1.0))
            .collect();
        let kv = self.k_mul(x, &v);
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
    let (write_table, save_ckpt, save_surface) =
        (opt("--write-table"), opt("--save-checkpoint"), opt("--save-surface"));
    std::fs::create_dir_all(&out).unwrap();

    let params = Params::load(cfg).unwrap();
    let Schedule::Nodal {
        t: knots,
        a: a_path,
        a_scale,
        tensor,
        ..
    } = &params.schedule
    else {
        panic!("adjoint_week needs the per-node growth law (law = \"nodal\")");
    };
    assert!(
        tensor.is_none(),
        "adjoint_week fits the scalar growth a_t: remove `tensor` from the config"
    );
    let (knots, a_path, a_scale) = (knots.clone(), a_path.clone(), *a_scale);
    let mesh = netgen::read(mesh_path).unwrap();
    let t0 = Instant::now();
    let mut sim = Simulation::<Fwd>::new(mesh.clone(), params.clone(), WgpuDevice::default()).unwrap();
    if let Schedule::Nodal { a, h, .. } = &params.schedule {
        let table = NodalTable::new(
            &params.schedule,
            &npy::read(a).unwrap().data,
            &npy::read(h).unwrap().data,
            mesh.n_nodes(),
        )
        .unwrap();
        sim.set_nodal_table(table);
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
    let ad = Simulation::<Ad>::new(mesh.clone(), params.clone(), WgpuDevice::default()).unwrap();
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
        a.data[knot * n..knot * n + target.len()]
            .iter()
            .map(|v| v * a_scale)
            .collect()
    } else {
        read(&prep.join("theta0.npy"))
    };
    let theta0 = theta.clone();
    let theta_mes: Vec<f64> = match opt("--measured") {
        Some(path) => {
            let a = npy::read(&path).unwrap();
            let n = a.shape[1];
            a.data[knot * n..knot * n + target.len()]
                .iter()
                .map(|v| v * a_scale)
                .collect()
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
        newton: flag("--newton", 6.0) as usize,
        sigma: 0.0,
        last_residual: (0.0, 0.0),
        colours: colour_nodes(mesh.n_nodes(), &mesh.tets),
        pre_inv: Vec::new(),
    };
    println!("{} colours for the block-Jacobi preconditioner", p.colours.len());
    let ns = target.len();
    let mut nbr = vec![Vec::new(); ns];
    for [i, j] in &p.edges {
        nbr[*i].push(*j);
        nbr[*j].push(*i);
    }
    p.nbr = nbr;
    if p.fi_weight > 0.0 {
        p.fi_target = p.fi_grad(&target).0;
        println!(
            "folding-index penalty: weight {}, atlas FI {:.4}, start FI {:.4}",
            p.fi_weight,
            p.fi_target,
            p.fi_grad(&xs).0
        );
    }
    if p.alpha > 0.0 {
        println!(
            "pull towards the measured growth: alpha {}, mean a_mes^2 {:.3e}",
            p.alpha, p.mes_scale
        );
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
    let cumulative = args.iter().any(|a| a == "--cumulative");
    let param_smooth = flag("--param-smooth", 0.0);
    let min_growth = args
        .iter()
        .position(|a| a == "--min-growth")
        .map(|i| args[i + 1].parse::<f64>().unwrap());
    let carry = args.iter().any(|a| a == "--carry");
    if levels.len() > 1 || levels[0].0 > 0 {
        assert!(p.hp == 0 && !p.chamfer, "--multiscale replaces --hp / --chamfer");
        println!("multi-resolution loss, levels (smoothing iterations : L-BFGS iterations) {levels:?}");
    }
    p.set_level(levels[0].0);

    // Equilibrium of the measured growth, from the checkpoint state.
    p.sim.set_state(&xs, &p.v0.clone(), &p.growth(&theta));
    let ray = p.rayleigh(&xs);
    let sigma = 1e-3 * ray.abs();
    p.sigma = sigma;
    let mut x = p.equilibrium(&theta, &xs, relax);
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
    let (mut ls, mut lr, mut gx) = p.loss(&x, &theta);
    println!(
        "start: shape loss {ls:.6e}, reg {lr:.3e}; K Rayleigh {ray:.3e}, shift {sigma:.3e}; |F| {:.3e} -> {:.3e}",
        p.last_residual.0, p.last_residual.1
    );
    save_rows(out.join("x_0.npy"), &x);

    let tg = Instant::now();
    let (mut g, it, res) = p.gradient(&x, &theta, &gx, sigma, cg);
    let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!(
        "gradient: CG {it} it (residual {res:.1e}), |g| {gnorm:.3e}, {:.1}s",
        tg.elapsed().as_secs_f64()
    );

    if check {
        // Directional finite difference along -g, normalised to max |d| = 0.01 (x --check-scale).
        let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let dir: Vec<f64> = g
            .iter()
            .map(|v| -0.01 * flag("--check-scale", 1.0) * v / gmax)
            .collect();
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
    for (li, &(level, level_iters)) in levels.iter().enumerate() {
        if cumulative {
            p.active = norm_levels[..=li].to_vec();
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
            let dir = p.smooth_scalar(&dir_z, pm, false);
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
                    for v in tn.iter_mut() {
                        *v = v.max(lo);
                    }
                }
                xn = p.equilibrium(&tn, &x, p.relax);
                parts = p.loss(&xn, &tn);
                fn_ = parts.0 + parts.1;
                let finite = fn_.is_finite() && xn.iter().flatten().all(|v| v.is_finite());
                if !finite {
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
            let s: Vec<f64> = dir_z.iter().map(|d| step * d).collect();
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
        }
    }
    if levels.len() > 1 {
        // Final loss at every level, for the report.
        let report: Vec<String> = levels
            .iter()
            .map(|&(l, _)| format!("k={l}: {:.4e}", p.level_loss(&x, l, &p.level_target(l)).0))
            .collect();
        println!("final loss per level: {}", report.join(", "));
    }

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
        println!(
            "table row {knot} (t = {:.4}) written to {}",
            knots[knot],
            path.display()
        );
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
