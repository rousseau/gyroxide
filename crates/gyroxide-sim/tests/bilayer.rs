//! Vérification numérique (P03, tâche 2) : bicouche plane, cortex d'épaisseur H sur substrat épais, croissance
//! tangentielle selon x en déformation plane. Croissance critique et longueur d'onde du premier flambage, comparées à
//! la théorie linéaire du film mince sur substrat épais (Allen 1969 ; Chen et Hutchinson 2004 ; Cao et Hutchinson 2012,
//! J. Appl. Mech. 79, 031019) :
//!
//!     lambda = 2 pi H (E_f / 3 E_s)^(1/3),   eps_c = (1/4) (3 E_s / E_f)^(2/3),   E = module de déformation plane
//!
//! avec E = E_young / (1 - nu^2) ; pour deux matériaux incompressibles E = 4 mu / 1, d'où la forme
//! lambda = 2 pi H (mu_f / 3 mu_s)^(1/3) de l'énoncé de P03. Ici K = 5 mu_g : le film est compressible (nu = 0,406),
//! on prend les modules de déformation plane des matériaux réels du code.
//!
//! Méthode : l'état plat (rouleaux sur les côtés, bas encastré, u_y = 0) est calculé sur une colonne d'une cellule par
//! Newton-Krylov, puis recopié sur le long maillage ; le plus petit mode propre de la tangente K = -dF/dx (Lanczos,
//! produits matrice-vecteur par différences finies centrées de la force) change de signe à la croissance critique,
//! trouvée par bissection. La longueur d'onde est celle du mode (spectre du déplacement vertical de la surface).
//! Le test (`flambage_bicouche`) est ignoré par défaut : `cargo test --release -- --ignored`.

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use burn::tensor::{Device, FloatDType};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Side};
use gyroxide_mesh::{topology::extract_boundary, TetMesh};
use gyroxide_model::{GrowthMask, Params, Preset};
use gyroxide_sim::algebra::{floats, to_f64, M3, V3};
use gyroxide_sim::elastic::{self, Thickness};
use gyroxide_sim::Simulation;

#[allow(deprecated)] // burn-ndarray est déprécié depuis Burn 0.22 (Device::flex() le remplace)
fn device() -> Device {
    static F64: std::sync::Once = std::sync::Once::new();
    let mut device = Device::ndarray();
    F64.call_once(|| device.configure(FloatDType::F64).expect("NdArray en f64"));
    device
}

/// Module de déformation plane E / (1 - nu^2) d'un matériau (cisaillement mu, compressibilité K).
fn plane_strain_modulus(mu: f64, k: f64) -> f64 {
    let e = 9.0 * k * mu / (3.0 * k + mu);
    let nu = (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu));
    e / (1.0 - nu * nu)
}

struct Theory {
    wavelength: f64,
    strain: f64,
}

fn theory(r: f64, bulk: f64, h: f64) -> Theory {
    let ratio = plane_strain_modulus(1.0, bulk) / plane_strain_modulus(1.0 / r, bulk);
    Theory {
        wavelength: 2.0 * std::f64::consts::PI * h * (ratio / 3.0).powf(1.0 / 3.0),
        strain: 0.25 * (3.0 / ratio).powf(2.0 / 3.0),
    }
}

/// Brique [x_0, x_n] x [0, w] x [z_0, 0], six tétraèdres par cellule ; seules les faces du dessus (z = 0) sont
/// déclarées surface (couche corticale, normales de croissance), comme `tools/make_slab.py`.
fn brick(xs: &[f64], zs: &[f64], w: f64) -> TetMesh {
    let (nx, nz) = (xs.len(), zs.len());
    let id = |i: usize, j: usize, k: usize| ((i * 2 + j) * nz + k) as u32;
    let mut m = TetMesh::default();
    for i in 0..nx {
        for j in 0..2 {
            for k in 0..nz {
                m.nodes.push([xs[i], j as f64 * w, zs[k]]);
            }
        }
    }
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for i in 0..nx - 1 {
        for k in 0..nz - 1 {
            for p in perms {
                let mut c = [i, 0, k];
                let mut t = [id(c[0], c[1], c[2]); 4];
                for (s, &axis) in p.iter().enumerate() {
                    c[axis] += 1;
                    t[s + 1] = id(c[0], c[1], c[2]);
                }
                let [a, b, cc, d] = t.map(|v| m.nodes[v as usize]);
                let det = (b[0] - a[0]) * ((cc[1] - a[1]) * (d[2] - a[2]) - (cc[2] - a[2]) * (d[1] - a[1]))
                    - (b[1] - a[1]) * ((cc[0] - a[0]) * (d[2] - a[2]) - (cc[2] - a[2]) * (d[0] - a[0]))
                    + (b[2] - a[2]) * ((cc[0] - a[0]) * (d[1] - a[1]) - (cc[1] - a[1]) * (d[0] - a[0]));
                if det < 0.0 {
                    t.swap(2, 3);
                }
                m.tets.push(t);
            }
        }
    }
    let all = extract_boundary(&m);
    m.faces = all
        .into_iter()
        .filter(|f| f.iter().all(|&n| m.nodes[n as usize][2].abs() < 1e-12))
        .collect();
    m
}

/// Cotes z (de 0 vers le bas) : `nh` couches par épaisseur H dans le film, puis un pas qui grandit de 20 % jusqu'à `dmax`.
fn z_levels(h: f64, nh: usize, depth: f64, dzmax: f64) -> Vec<f64> {
    let dz0 = h / nh as f64;
    let (mut z, mut step): (Vec<f64>, f64) = (vec![0.0], dz0);
    while -z.last().unwrap() < depth {
        let d = -z.last().unwrap();
        if d >= 1.5 * h {
            step = (step * 1.2).min(dzmax);
        }
        z.push(-(d + step));
    }
    z.reverse();
    z
}

struct Bilayer {
    sim: Simulation,
    pre: gyroxide_sim::setup::Precomputed,
    mesh: TetMesh,
    /// Composantes bloquées par nœud.
    fixed: Vec<[bool; 3]>,
    gm: Vec<f64>,
    tensor: M3,
    h: f64,
}

fn build(r: f64, xs: &[f64], zs: &[f64], h: f64, sharp: bool) -> Bilayer {
    let mesh = brick(xs, zs, 0.5 * h);
    let mut p: Params = Preset::Tallinen2016.params();
    p.growth_mask = GrowthMask::Uniform;
    p.contact.enabled = false;
    p.contact.midplane = None;
    p.material.mu_gray = 1.0;
    p.material.mu_white = 1.0 / r;
    p.material.bulk = 5.0;
    let sim = Simulation::new(mesh.clone(), p, device()).unwrap();
    let nt = mesh.n_tets();
    let (x0, x1, zb) = (xs[0], *xs.last().unwrap(), *zs.first().unwrap());
    let fixed = mesh
        .nodes
        .iter()
        .map(|q| {
            let bottom = (q[2] - zb).abs() < 1e-12;
            let side = (q[0] - x0).abs() < 1e-12 || (q[0] - x1).abs() < 1e-12;
            [side || bottom, true, bottom]
        })
        .collect();
    let zero = floats(&vec![0.0; nt], &device());
    // `sharp` : transition 10 fois plus étroite entre cortex et substrat (la loi du code est
    // g_m = 1/(1 + exp(10 (d/H - 1))), de 0,4 H entre 10 et 90 % ; ici d' = H + 10 (d - H), soit 0,04 H),
    // pour isoler l'erreur de discrétisation de l'effet de la transition.
    let mut pre = sim.precomputed().clone();
    if sharp {
        pre.d_mean = (pre.d_mean.clone() - h) * 10.0 + h;
    }
    let gm = to_f64(elastic::forces_with(&pre, &sim.params.material, &sim.x, &zero, &Thickness::Uniform(h), false).gm);
    let tensor = M3 {
        m: std::array::from_fn(|i| {
            std::array::from_fn(|j| floats(&vec![(i == 0 && j == 0) as u8 as f64; nt], &device()))
        }),
    };
    Bilayer {
        sim,
        pre,
        mesh,
        fixed,
        gm,
        tensor,
        h,
    }
}

impl Bilayer {
    fn force(&self, x: &[[f64; 3]], s: f64) -> Vec<[f64; 3]> {
        let g = floats(&self.gm.iter().map(|g| g * s).collect::<Vec<_>>(), &device());
        let xv = V3::from_rows(x, &device());
        let mut f = elastic::forces_tensor(
            &self.pre,
            &self.sim.params.material,
            &xv,
            &g,
            &self.tensor,
            &Thickness::Uniform(self.h),
            false,
        )
        .force
        .to_rows();
        self.mask(&mut f);
        f
    }
    fn mask(&self, v: &mut [[f64; 3]]) {
        for (r, fx) in v.iter_mut().zip(&self.fixed) {
            for k in 0..3 {
                if fx[k] {
                    r[k] = 0.0;
                }
            }
        }
    }
    /// K v = -dF/dx v par différences finies centrées (f64).
    fn kmul(&self, x: &[[f64; 3]], s: f64, v: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let vmax = v.iter().flatten().fold(0.0f64, |m, a| m.max(a.abs())).max(1e-300);
        let e = 1e-6 / vmax;
        let shift = |sg: f64| -> Vec<[f64; 3]> {
            x.iter()
                .zip(v)
                .map(|(p, d)| std::array::from_fn(|k| p[k] + sg * e * d[k]))
                .collect()
        };
        let (fp, fm) = (self.force(&shift(1.0), s), self.force(&shift(-1.0), s));
        fp.iter()
            .zip(&fm)
            .map(|(a, b)| std::array::from_fn(|k| -(a[k] - b[k]) / (2.0 * e)))
            .collect()
    }
    /// Diagonale de K par sondes colorées (deux nœuds de même couleur ne partagent aucun tétraèdre) : 24 produits.
    fn diagonal(&self, x: &[[f64; 3]], s: f64, nz: usize) -> Vec<[f64; 3]> {
        let n = x.len();
        let color = |idx: usize| {
            let (k, ij) = (idx % nz, idx / nz);
            (ij / 2 % 2) * 4 + (ij % 2) * 2 + k % 2
        };
        let mut d = vec![[1.0; 3]; n];
        for c in 0..8 {
            for comp in 0..3 {
                let mut v = vec![[0.0; 3]; n];
                for idx in (0..n).filter(|&i| color(i) == c) {
                    v[idx][comp] = 1.0;
                }
                self.mask(&mut v);
                let kv = self.kmul(x, s, &v);
                for idx in (0..n).filter(|&i| color(i) == c && !self.fixed[i][comp]) {
                    d[idx][comp] = kv[idx][comp].abs().max(1e-12);
                }
            }
        }
        d
    }
    /// Matrice creuse de K sur les degrés de liberté libres, par sondes colorées. La force d'un nœud dépend des nœuds
    /// à deux mailles (la pression nodale moyenne J_a couple les voisins des voisins) : deux nœuds de même couleur
    /// (période 5 en x et en z, y différent) n'ont aucune ligne commune, une sonde par couleur et composante donne donc
    /// leurs colonnes entières (150 sondes, chacune deux évaluations de la force).
    fn assemble(&self, x: &[[f64; 3]], s: f64, nz: usize) -> (SparseColMat<usize, f64>, Vec<[usize; 3]>) {
        let n = x.len();
        let mut free = vec![[usize::MAX; 3]; n];
        let mut nfree = 0;
        for (i, f) in self.fixed.iter().enumerate() {
            for k in 0..3 {
                if !f[k] {
                    free[i][k] = nfree;
                    nfree += 1;
                }
            }
        }
        let cell = |idx: usize| {
            let (ij, k) = (idx / nz, idx % nz);
            (ij / 2, ij % 2, k)
        };
        let nx = n / (2 * nz);
        let color = |idx: usize| {
            let (i, j, k) = cell(idx);
            (i % 5) * 10 + j * 5 + k % 5
        };
        let mut triplets = Vec::new();
        for c in 0..50 {
            for comp in 0..3 {
                let cols: Vec<usize> = (0..n)
                    .filter(|&i| color(i) == c && free[i][comp] != usize::MAX)
                    .collect();
                if cols.is_empty() {
                    continue;
                }
                let mut v = vec![[0.0; 3]; n];
                for &j in &cols {
                    v[j][comp] = 1.0;
                }
                let kv = self.kmul(x, s, &v);
                for &j in &cols {
                    let (ij, jj, kj) = cell(j);
                    for di in -2i64..=2 {
                        for dj in -1i64..=1 {
                            for dk in -2i64..=2 {
                                let (ii, jn, kn) = (ij as i64 + di, jj as i64 + dj, kj as i64 + dk);
                                if ii < 0 || ii >= nx as i64 || !(0..2).contains(&jn) || kn < 0 || kn >= nz as i64 {
                                    continue;
                                }
                                let row = ((ii as usize) * 2 + jn as usize) * nz + kn as usize;
                                for r in 0..3 {
                                    if free[row][r] != usize::MAX && kv[row][r] != 0.0 {
                                        triplets.push(Triplet::new(free[row][r], free[j][comp], kv[row][r]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Symétrisation : K_sym = (K + K^T) / 2 (la différence finie centrée est symétrique à 1e-10 près).
        let sym: Vec<Triplet<usize, usize, f64>> = triplets
            .iter()
            .flat_map(|t| {
                [
                    Triplet::new(t.row, t.col, 0.5 * t.val),
                    Triplet::new(t.col, t.row, 0.5 * t.val),
                ]
            })
            .collect();
        (SparseColMat::try_new_from_triplets(nfree, nfree, &sym).unwrap(), free)
    }
    /// Equilibre par Newton-Krylov (CG sur K), depuis `x`.
    fn equilibrium(&self, mut x: Vec<[f64; 3]>, s: f64) -> Vec<[f64; 3]> {
        let n = x.len();
        for _ in 0..15 {
            let r = self.force(&x, s);
            let rn = dot(&r, &r).sqrt();
            if rn < 1e-11 {
                break;
            }
            let mut d = vec![[0.0; 3]; n];
            let (mut res, mut p) = (r.clone(), r.clone());
            let mut rr = dot(&res, &res);
            for _ in 0..1500 {
                let kp = self.kmul(&x, s, &p);
                let pk = dot(&p, &kp);
                assert!(pk > 0.0, "tangente non définie positive dans le calcul de l'état plat");
                let a = rr / pk;
                for i in 0..n {
                    for k in 0..3 {
                        d[i][k] += a * p[i][k];
                        res[i][k] -= a * kp[i][k];
                    }
                }
                let rr2 = dot(&res, &res);
                if rr2.sqrt() < 1e-9 * rn {
                    break;
                }
                let b = rr2 / rr;
                rr = rr2;
                for i in 0..n {
                    for k in 0..3 {
                        p[i][k] = res[i][k] + b * p[i][k];
                    }
                }
            }
            for i in 0..n {
                for k in 0..3 {
                    x[i][k] += d[i][k];
                }
            }
        }
        x
    }
}

fn dot(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| p[0] * q[0] + p[1] * q[1] + p[2] * q[2])
        .sum()
}

/// Plus petite valeur propre de Ritz (et vecteur) de `op`, Lanczos à réorthogonalisation complète.
fn lanczos_min(
    n: usize,
    m: usize,
    mask: &dyn Fn(&mut Vec<[f64; 3]>),
    op: &dyn Fn(&[[f64; 3]]) -> Vec<[f64; 3]>,
) -> (f64, Vec<[f64; 3]>) {
    let mut s = 12345u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut q: Vec<Vec<[f64; 3]>> = Vec::new();
    let mut v: Vec<[f64; 3]> = (0..n).map(|_| [rnd(), rnd(), rnd()]).collect();
    mask(&mut v);
    let nv = dot(&v, &v).sqrt();
    v.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c /= nv));
    q.push(v);
    let (mut alpha, mut beta) = (Vec::new(), Vec::new());
    for j in 0..m {
        let mut w = op(&q[j]);
        let a = dot(&w, &q[j]);
        alpha.push(a);
        for qi in &q {
            let c = dot(&w, qi);
            for i in 0..n {
                for k in 0..3 {
                    w[i][k] -= c * qi[i][k];
                }
            }
        }
        for qi in &q {
            let c = dot(&w, qi);
            for i in 0..n {
                for k in 0..3 {
                    w[i][k] -= c * qi[i][k];
                }
            }
        }
        let b = dot(&w, &w).sqrt();
        if j + 1 == m || b < 1e-12 {
            break;
        }
        beta.push(b);
        w.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c /= b));
        q.push(w);
    }
    let k = alpha.len();
    let mut t = vec![vec![0.0; k]; k];
    for i in 0..k {
        t[i][i] = alpha[i];
        if i + 1 < k {
            t[i][i + 1] = beta[i];
            t[i + 1][i] = beta[i];
        }
    }
    let (vals, vecs) = jacobi(t);
    let imin = (0..k).min_by(|&a, &b| vals[a].total_cmp(&vals[b])).unwrap();
    let mut vec_min = vec![[0.0; 3]; n];
    for (j, qj) in q.iter().enumerate().take(k) {
        let c = vecs[j][imin];
        for i in 0..n {
            for d in 0..3 {
                vec_min[i][d] += c * qj[i][d];
            }
        }
    }
    (vals[imin], vec_min)
}

/// Valeurs et vecteurs propres d'une matrice symétrique (Jacobi cyclique).
fn jacobi(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..60 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-28 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let th = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

struct Result {
    s_crit: f64,
}

/// Croissance critique (bissection sur le signe de la plus petite valeur propre) et longueur d'onde du mode.
fn measure(r: f64, length: f64, hx: f64, nh: usize, depth: f64, lanczos: usize, sharp: bool) -> Result {
    let h = 1.0;
    let th = theory(r, 5.0, h);
    let nx = (length / hx).round() as usize;
    let xs: Vec<f64> = (0..=nx).map(|i| length * i as f64 / nx as f64).collect();
    let zs = z_levels(h, nh, depth, 1.0);
    let col = build(r, &[0.0, length / nx as f64], &zs, h, sharp);
    let long = build(r, &xs, &zs, h, sharp);
    let n = long.mesh.n_nodes();
    let nz = zs.len();
    let s_th = th.strain / (1.0 - th.strain);
    let mut x_col = col.mesh.nodes.clone();
    let flat = |s: f64, x_col: &mut Vec<[f64; 3]>| -> Vec<[f64; 3]> {
        *x_col = col.equilibrium(x_col.clone(), s);
        let uz: Vec<f64> = (0..nz).map(|k| x_col[k][2] - col.mesh.nodes[k][2]).collect();
        long.mesh
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, p)| [p[0], p[1], p[2] + uz[idx % nz]])
            .collect()
    };
    let unstable = |s: f64, x_col: &mut Vec<[f64; 3]>| -> (f64, Vec<[f64; 3]>) {
        let x = flat(s, x_col);
        let mask = |v: &mut Vec<[f64; 3]>| long.mask(v);
        // Mise à l'échelle de Jacobi : le signe de la plus petite valeur propre de D^-1/2 K D^-1/2 est celui de K.
        let dg = long.diagonal(&x, s, nz);
        let sc: Vec<[f64; 3]> = dg.iter().map(|r| r.map(|c| 1.0 / c.sqrt())).collect();
        let scale = |v: &[[f64; 3]]| -> Vec<[f64; 3]> {
            v.iter()
                .zip(&sc)
                .map(|(a, b)| std::array::from_fn(|k| a[k] * b[k]))
                .collect()
        };
        let (lam, w) = lanczos_min(n, lanczos, &mask, &|w| {
            let mut o = long.kmul(&x, s, &scale(w));
            long.mask(&mut o);
            scale(&o)
        });
        let vec = scale(&w);
        (lam, vec)
    };
    let (mut lo, mut hi) = (0.4 * s_th, 2.0 * s_th);
    let (l_lo, _) = unstable(lo, &mut x_col);
    assert!(l_lo > 0.0, "déjà instable à {lo}");
    let (l_hi, _) = unstable(hi, &mut x_col);
    assert!(l_hi < 0.0, "encore stable à {hi} : valeur propre {l_hi}");
    for _ in 0..9 {
        let mid = 0.5 * (lo + hi);
        let (l, _) = unstable(mid, &mut x_col);
        if l < 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Result {
        s_crit: 0.5 * (lo + hi),
    }
}

/// Croissance critique par factorisation creuse : K est définie positive tant que la structure est stable, la
/// factorisation de Cholesky de `faer` (dépendance de développement) réussit alors ; à la croissance critique la plus
/// petite valeur propre passe par zéro et elle échoue (pivot non positif). Bissection sur ce test, sans Lanczos.
fn measure_sparse(r: f64, length: f64, hx: f64, nh: usize, depth: f64, sharp: bool, steps: usize) -> Result {
    let h = 1.0;
    let th = theory(r, 5.0, h);
    let nx = (length / hx).round() as usize;
    let xs: Vec<f64> = (0..=nx).map(|i| length * i as f64 / nx as f64).collect();
    let zs = z_levels(h, nh, depth, 1.0);
    let col = build(r, &[0.0, length / nx as f64], &zs, h, sharp);
    let long = build(r, &xs, &zs, h, sharp);
    let nz = zs.len();
    let s_th = th.strain / (1.0 - th.strain);
    let mut x_col = col.mesh.nodes.clone();
    let mut stable = |s: f64| -> bool {
        x_col = col.equilibrium(x_col.clone(), s);
        let uz: Vec<f64> = (0..nz).map(|k| x_col[k][2] - col.mesh.nodes[k][2]).collect();
        let x: Vec<[f64; 3]> = long
            .mesh
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, p)| [p[0], p[1], p[2] + uz[idx % nz]])
            .collect();
        let (k, _) = long.assemble(&x, s, nz);
        k.sp_cholesky(Side::Lower).is_ok()
    };
    let (mut lo, mut hi) = (0.4 * s_th, 2.0 * s_th);
    assert!(stable(lo), "déjà instable à {lo}");
    assert!(!stable(hi), "encore stable à {hi}");
    for _ in 0..steps {
        let mid = 0.5 * (lo + hi);
        if stable(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Result {
        s_crit: 0.5 * (lo + hi),
    }
}

/// Croissance critique du mode n = 1 (longueur d'onde 2 L) pour trois longueurs d'onde autour de la théorie ; renvoie
/// (longueur d'onde du minimum par ajustement parabolique de s_c(lambda), s_c minimale).
fn scan(r: f64, sharp: bool) -> (f64, f64) {
    let th = theory(r, 5.0, 1.0);
    let fs = [0.8, 1.0, 1.25];
    let sc: Vec<f64> = fs
        .iter()
        .map(|&f| measure(r, 0.5 * f * th.wavelength, 0.5, 3, 0.8 * th.wavelength, 240, sharp).s_crit)
        .collect();
    // parabole en ln(lambda) par les trois points
    let x: Vec<f64> = fs.iter().map(|f: &f64| f.ln()).collect();
    let d1 = (sc[1] - sc[0]) / (x[1] - x[0]);
    let d2 = (sc[2] - sc[1]) / (x[2] - x[1]);
    let a2 = (d2 - d1) / (x[2] - x[0]);
    let a1 = d1 - a2 * (x[0] + x[1]);
    let xm = -a1 / (2.0 * a2);
    let smin = sc[1] + a1 * (xm - x[1]) + a2 * (xm * xm - x[1] * x[1]);
    let lam = th.wavelength * xm.exp();
    eprintln!(
        "mu_f/mu_s = {r}, transition {} : s_c(lambda/lambda_th = 0,8 ; 1 ; 1,25) = {:.4} ; {:.4} ; {:.4} -> lambda* = {:.2} H (théorie {:.2}, {:+.1} %), eps_c = {:.4} (théorie {:.4}, {:+.1} %)",
        if sharp { "nette" } else { "du code" },
        sc[0],
        sc[1],
        sc[2],
        lam,
        th.wavelength,
        100.0 * (lam / th.wavelength - 1.0),
        smin / (1.0 + smin),
        th.strain,
        100.0 * ((smin / (1.0 + smin)) / th.strain - 1.0)
    );
    (lam, smin)
}

/// Maillage grossier (3 éléments par H) : voir docs/validation.md pour les limites. Ignoré par défaut (environ 3 min) :
/// `cargo test --release -p gyroxide-sim --test bilayer -- --ignored --nocapture`.
#[test]
#[ignore = "long"]
fn flambage_bicouche() {
    let th = theory(50.0, 5.0, 1.0);
    let (lam, _) = scan(50.0, true);
    assert!(
        (lam / th.wavelength - 1.0).abs() < 0.15,
        "longueur d'onde (transition nette) : {lam:.2} contre {:.2}",
        th.wavelength
    );
    // Transition de la loi du code : mesure rapportée, pas de seuil (la théorie est celle du bicouche net).
    let _ = scan(50.0, false);
}

/// Balayage creux : s_c pour cinq longueurs d'onde (0,8 à 1,25 lambda_th), ajustement par les moindres carrés d'une
/// parabole en ln(lambda), minimum -> (lambda*, s_c min).
fn scan_sparse(r: f64, sharp: bool, nh: usize, hx: f64) -> (f64, f64, Vec<f64>) {
    let th = theory(r, 5.0, 1.0);
    let fs = [0.8, 0.9, 1.0, 1.12, 1.25];
    let sc: Vec<f64> = fs
        .iter()
        .map(|&f| measure_sparse(r, 0.5 * f * th.wavelength, hx, nh, 0.8 * th.wavelength, sharp, 16).s_crit)
        .collect();
    let x: Vec<f64> = fs.iter().map(|f: &f64| f.ln()).collect();
    // normales : [1 x x^2] c = s
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for (xi, si) in x.iter().zip(&sc) {
        let p = [1.0, *xi, xi * xi];
        for i in 0..3 {
            b[i] += p[i] * si;
            for j in 0..3 {
                a[i][j] += p[i] * p[j];
            }
        }
    }
    // résolution 3x3 (Cramer)
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    let coef: Vec<f64> = (0..3)
        .map(|k| {
            let mut m = a;
            for i in 0..3 {
                m[i][k] = b[i];
            }
            det(m) / d
        })
        .collect();
    let xm = -coef[1] / (2.0 * coef[2]);
    let smin = coef[0] + coef[1] * xm + coef[2] * xm * xm;
    (th.wavelength * xm.exp(), smin, sc)
}

fn report(r: f64, sharp: bool, nh: usize, hx: f64) {
    let th = theory(r, 5.0, 1.0);
    let t = std::time::Instant::now();
    let (lam, smin, sc) = scan_sparse(r, sharp, nh, hx);
    let eps = smin / (1.0 + smin);
    eprintln!(
        "mu_f/mu_s = {r}, transition {}, {nh} par H (h_x = {hx} H) : s_c = {:.5} ; {:.5} ; {:.5} ; {:.5} ; {:.5} -> lambda* = {:.2} H (théorie {:.2}, {:+.1} %), eps_c = {:.4} (théorie {:.4}, {:+.1} %) [{:.0}s]",
        if sharp { "nette" } else { "du code" },
        sc[0], sc[1], sc[2], sc[3], sc[4],
        lam, th.wavelength, 100.0 * (lam / th.wavelength - 1.0),
        eps, th.strain, 100.0 * (eps / th.strain - 1.0),
        t.elapsed().as_secs_f64()
    );
}

/// Balayage de la tâche 2 refaite (E36/E37) : `cargo test --release -p gyroxide-sim --test bilayer balayage_creux -- --ignored --nocapture`.
#[test]
#[ignore = "long"]
fn balayage_creux() {
    for &(nh, hx) in &[(3usize, 0.5f64), (6, 0.25), (12, 0.125)] {
        for r in [20.0, 50.0, 100.0] {
            for sharp in [true, false] {
                report(r, sharp, nh, hx);
            }
        }
    }
}

/// Une finesse de plus pour la transition nette (E37) : `... fine_nette -- --ignored --nocapture`.
#[test]
#[ignore = "long"]
fn fine_nette() {
    report(50.0, true, 24, 0.0625);
}

/// Test de CI (4 s) du solveur creux : K assemblée par sondes colorées égale le produit par différences finies, et la
/// croissance critique du maillage grossier (transition nette, mu_f / mu_s = 50, lambda_th, 3 éléments par H) retrouve
/// la valeur de l'ancienne mesure par Lanczos (0,0265 ; 0,0266 avec la bissection creuse).
#[test]
fn assemblage_creux_et_croissance_critique() {
    let th = theory(50.0, 5.0, 1.0);
    let (length, hx, nh, depth) = (0.5 * th.wavelength, 0.5, 3usize, 0.8 * th.wavelength);
    let nx = (length / hx).round() as usize;
    let xs: Vec<f64> = (0..=nx).map(|i| length * i as f64 / nx as f64).collect();
    let zs = z_levels(1.0, nh, depth, 1.0);
    let col = build(50.0, &[0.0, length / nx as f64], &zs, 1.0, true);
    let long = build(50.0, &xs, &zs, 1.0, true);
    let nz = zs.len();
    let s = 0.02;
    let x_col = col.equilibrium(col.mesh.nodes.clone(), s);
    let uz: Vec<f64> = (0..nz).map(|k| x_col[k][2] - col.mesh.nodes[k][2]).collect();
    let x: Vec<[f64; 3]> = long
        .mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(idx, p)| [p[0], p[1], p[2] + uz[idx % nz]])
        .collect();
    let (k, free) = long.assemble(&x, s, nz);
    let mut v = vec![[0.0; 3]; x.len()];
    for (i, r) in v.iter_mut().enumerate() {
        for c in 0..3 {
            r[c] = (((i * 7 + c * 13) % 17) as f64 / 8.0) - 1.0;
        }
    }
    long.mask(&mut v);
    let mut kv = long.kmul(&x, s, &v);
    long.mask(&mut kv);
    let mut vf = Mat::<f64>::zeros(k.nrows(), 1);
    for (i, f) in free.iter().enumerate() {
        for c in 0..3 {
            if f[c] != usize::MAX {
                vf[(f[c], 0)] = v[i][c];
            }
        }
    }
    let w = &k * &vf;
    let (mut err, mut nrm) = (0.0f64, 0.0f64);
    for (i, f) in free.iter().enumerate() {
        for c in 0..3 {
            if f[c] != usize::MAX {
                err = err.max((w[(f[c], 0)] - kv[i][c]).abs());
                nrm = nrm.max(kv[i][c].abs());
            }
        }
    }
    assert!(
        err < 1e-8 * nrm,
        "K creuse contre K v par différences finies : {err:e} (échelle {nrm:e})"
    );
    assert!(k.sp_cholesky(Side::Lower).is_ok(), "K définie positive à s = {s}");
    let r = measure_sparse(50.0, length, hx, nh, depth, true, 12);
    assert!(
        (r.s_crit / 0.0266 - 1.0).abs() < 0.02,
        "croissance critique {} (Lanczos : 0,0265)",
        r.s_crit
    );
}
