//! Vérification numérique (P03, tâche 1) : le code résout-il bien ses équations ? Tests en f64 sur CPU.
//!
//! 1. Patch test : déformation homogène F et croissance G connues sur un cube de tétraèdres ; contraintes, énergie
//!    et forces des nœuds intérieurs comparées à leur valeur analytique (forces nulles à l'arrondi).
//! 2. Cohérence force / énergie : la force est le gradient de l'énergie (différences finies centrées).
//! 3. Solution manufacturée : déplacement analytique imposé au bord d'un cube, force volumique déduite de la
//!    divergence de P, équilibre résolu par Newton-Krylov sur trois maillages ; ordre de convergence mesuré.
//!
//! Chaque test tient en moins de 30 s en release (`cargo test --release`).

#![allow(clippy::needless_range_loop)] // indices de matrices 3x3 : plus lisible ainsi

use burn::tensor::{Device, FloatDType};
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

/// Cube [0, 1]^3 en `n`^3 cellules, six tétraèdres par cellule (Kuhn, les diagonales se raccordent).
fn cube(n: usize) -> TetMesh {
    let id = |i: usize, j: usize, k: usize| ((i * (n + 1) + j) * (n + 1) + k) as u32;
    let h = 1.0 / n as f64;
    let mut m = TetMesh::default();
    for i in 0..=n {
        for j in 0..=n {
            for k in 0..=n {
                m.nodes.push([i as f64 * h, j as f64 * h, k as f64 * h]);
            }
        }
    }
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for p in perms {
                    let mut c = [i, j, k];
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
    }
    m.faces = extract_boundary(&m);
    m
}

/// Matériau homogène (mu_gray = mu_white = 1, K = 5), sans contact : la loi ne dépend plus de la profondeur.
fn params() -> Params {
    let mut p = Preset::Tallinen2016.params();
    p.growth_mask = GrowthMask::Uniform;
    p.contact.enabled = false;
    p.contact.midplane = None;
    p.material.mu_gray = 1.0;
    p.material.mu_white = 1.0;
    p.material.bulk = 5.0;
    p
}

type Mat3 = [[f64; 3]; 3];

fn mm(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn tr(a: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn det(a: &Mat3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}
fn apply(a: &Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum())
}
/// Transposée de l'adjugée, J F^-T.
fn cof(a: &Mat3) -> Mat3 {
    let c = |i: usize, j: usize| {
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]
    };
    std::array::from_fn(|i| std::array::from_fn(|j| c(i, j)))
}

/// Contrainte de Cauchy de la branche régulière, mu J^(-5/3) dev(B) + K (Ja - 1) I (Ja = J si le champ est homogène).
fn cauchy(f: &Mat3, mu: f64, bulk: f64, ja: f64) -> Mat3 {
    let j = det(f);
    let b = mm(f, &tr(f));
    let t = (b[0][0] + b[1][1] + b[2][2]) / 3.0;
    std::array::from_fn(|i| {
        std::array::from_fn(|k| {
            mu * j.powf(-5.0 / 3.0) * (b[i][k] - if i == k { t } else { 0.0 })
                + if i == k { bulk * (ja - 1.0) } else { 0.0 }
        })
    })
}

/// Tenseur de croissance constant sur tous les tétraèdres : G = I + g T.
fn const_tensor(t: &Mat3, nt: usize) -> M3 {
    M3 {
        m: std::array::from_fn(|i| std::array::from_fn(|j| floats(&vec![t[i][j]; nt], &device()))),
    }
}

fn max_abs(v: &[[f64; 3]]) -> f64 {
    v.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn is_boundary(p: [f64; 3]) -> bool {
    p.iter().any(|&c| c < 1e-12 || c > 1.0 - 1e-12)
}

const FEL: Mat3 = [[1.10, 0.20, -0.10], [0.05, 0.90, 0.15], [-0.20, 0.10, 1.20]];
const T_GROWTH: Mat3 = [[0.5, 0.1, -0.2], [0.1, 0.2, 0.3], [-0.2, 0.3, 0.8]];
const G_AMP: f64 = 0.6;

#[test]
fn patch_test_homogeneous_deformation_and_growth() {
    let mesh = cube(4);
    let nt = mesh.n_tets();
    let sim = Simulation::new(mesh.clone(), params(), device()).unwrap();
    // G = I + g T, uniforme ; positions x = F_el G X : F = F_el sur chaque tétraèdre.
    let g: Mat3 = std::array::from_fn(|i| std::array::from_fn(|j| (i == j) as u8 as f64 + G_AMP * T_GROWTH[i][j]));
    let total = mm(&FEL, &g);
    let x_rows: Vec<[f64; 3]> = mesh.nodes.iter().map(|&p| apply(&total, p)).collect();
    let x = V3::from_rows(&x_rows, &device());
    let growth = floats(&vec![G_AMP; nt], &device());
    let tensor = const_tensor(&T_GROWTH, nt);
    let out = elastic::forces_tensor(
        sim.precomputed(),
        &sim.params.material,
        &x,
        &growth,
        &tensor,
        &Thickness::Uniform(1.0),
        true,
    );

    // Forces : nulles aux nœuds intérieurs (à l'arrondi), de somme nulle partout.
    let f = out.force.to_rows();
    let interior: Vec<usize> = (0..mesh.n_nodes()).filter(|&i| !is_boundary(mesh.nodes[i])).collect();
    let fi: Vec<[f64; 3]> = interior.iter().map(|&i| f[i]).collect();
    let scale = max_abs(&f); // les forces de bord sont l'échelle naturelle
    assert!(scale > 1e-3, "forces de bord triviales : {scale}");
    assert!(
        max_abs(&fi) < 1e-13 * scale.max(1.0),
        "forces intérieures : {} (échelle {scale})",
        max_abs(&fi)
    );
    let sum: [f64; 3] = std::array::from_fn(|k| f.iter().map(|r| r[k]).sum());
    assert!(max_abs(&[sum]) < 1e-12 * scale.max(1.0), "somme des forces : {sum:?}");

    // Contrainte : la même dans tous les tétraèdres, égale à la valeur analytique.
    let j = det(&FEL);
    let want = cauchy(&FEL, 1.0, 5.0, j);
    let (sigma, jt) = out.stress.expect("contrainte demandée");
    let jt = to_f64(jt);
    for r in 0..3 {
        for c in 0..3 {
            let s = to_f64(sigma.get(r, c));
            let worst = s.iter().fold(0.0f64, |m, v| m.max((v - want[r][c]).abs()));
            assert!(
                worst < 1e-13 * want[r][c].abs().max(1.0),
                "sigma[{r}][{c}] : écart {worst:e} (attendu {})",
                want[r][c]
            );
        }
    }
    assert!(jt.iter().all(|v| (v - j).abs() < 1e-13), "J : attendu {j}");

    // Énergie : somme sur les tétraèdres de W V0, V0 = volume du cube grandi, W = mu/2 (tr B J^-2/3 - 3) + K/2 (J - 1)^2.
    let b = mm(&FEL, &tr(&FEL));
    let w = 0.5 * ((b[0][0] + b[1][1] + b[2][2]) * j.powf(-2.0 / 3.0) - 3.0) + 2.5 * (j - 1.0) * (j - 1.0);
    let v0 = det(&g); // cube de côté 1 grandi par G
    let (ue, ninv) = out.energy.expect("énergie demandée");
    let (ue, ninv) = (to_f64(ue)[0], to_f64(ninv)[0]);
    assert!(
        (ue - w * v0).abs() < 1e-13 * (w * v0).abs(),
        "énergie {ue} contre {}",
        w * v0
    );
    assert_eq!(ninv, 0.0);
}

#[test]
fn force_is_energy_gradient() {
    // Maillage perturbé (aucun tétraèdre inversé) : la force est l'opposé du gradient de l'énergie.
    let mesh = cube(3);
    let nt = mesh.n_tets();
    let sim = Simulation::new(mesh.clone(), params(), device()).unwrap();
    let growth = floats(&vec![0.2; nt], &device());
    let tensor = const_tensor(&T_GROWTH, nt);
    let mut s = 11u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let x0: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|&p| {
            let q = apply(&FEL, p);
            std::array::from_fn(|k| q[k] + 0.03 * rnd())
        })
        .collect();
    let energy_of = |rows: &[[f64; 3]]| {
        let x = V3::from_rows(rows, &device());
        let o = elastic::forces_tensor(
            sim.precomputed(),
            &sim.params.material,
            &x,
            &growth,
            &tensor,
            &Thickness::Uniform(1.0),
            true,
        );
        to_f64(o.energy.unwrap().0)[0]
    };
    let x = V3::from_rows(&x0, &device());
    let f = elastic::forces_tensor(
        sim.precomputed(),
        &sim.params.material,
        &x,
        &growth,
        &tensor,
        &Thickness::Uniform(1.0),
        false,
    )
    .force
    .to_rows();
    let eps = 1e-6;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for &i in &[0usize, 5, 17, 30, 41, 55] {
        for k in 0..3 {
            let (mut p, mut m) = (x0.clone(), x0.clone());
            p[i][k] += eps;
            m[i][k] -= eps;
            let g = -(energy_of(&p) - energy_of(&m)) / (2.0 * eps);
            num += (g - f[i][k]).powi(2);
            den += g * g;
        }
    }
    let rel = (num / den).sqrt();
    eprintln!("force contre -grad(énergie) : écart relatif {rel:.3e}");
    // Mesuré : 1,8e-10 (limité par les différences finies) ; 3,4e-8 avant l'exposant 2/3 exact en f64.
    assert!(rel < 1e-8, "écart relatif force / gradient : {rel:e}");
}

// ---- Solution manufacturée -------------------------------------------------------------------------------------

const PI: f64 = std::f64::consts::PI;
const AMP: f64 = 0.05;

fn u_exact(p: [f64; 3]) -> [f64; 3] {
    let s = |v: f64| (PI * v).sin();
    [
        AMP * s(p[0]) * s(p[1]),
        AMP * s(p[1]) * s(p[2]),
        AMP * s(p[2]) * s(p[0]),
    ]
}

/// F = I + grad u, dérivées analytiques.
fn f_exact(p: [f64; 3]) -> Mat3 {
    let s = |v: f64| (PI * v).sin();
    let c = |v: f64| PI * (PI * v).cos();
    let (x, y, z) = (p[0], p[1], p[2]);
    let g = [
        [AMP * c(x) * s(y), AMP * s(x) * c(y), 0.0],
        [0.0, AMP * c(y) * s(z), AMP * s(y) * c(z)],
        [AMP * s(z) * c(x), 0.0, AMP * c(z) * s(x)],
    ];
    std::array::from_fn(|i| std::array::from_fn(|j| g[i][j] + (i == j) as u8 as f64))
}

/// Premier tenseur de Piola-Kirchhoff exact P = J sigma F^-T (J_a = J : limite continue).
fn p_exact(p: [f64; 3]) -> Mat3 {
    let f = f_exact(p);
    let s = cauchy(&f, 1.0, 5.0, det(&f));
    mm(&s, &cof(&f))
}

/// Force volumique de référence b = - Div P, divergence par différences finies centrées de P (fonction exacte).
fn body_force(p: [f64; 3]) -> [f64; 3] {
    let e = 1e-5;
    std::array::from_fn(|i| {
        -(0..3)
            .map(|j| {
                let (mut a, mut b) = (p, p);
                a[j] += e;
                b[j] -= e;
                (p_exact(a)[i][j] - p_exact(b)[i][j]) / (2.0 * e)
            })
            .sum::<f64>()
    })
}

struct Solved {
    h: f64,
    max_err: f64,
    l2_err: f64,
    newton: usize,
    cg: usize,
}

/// Équilibre du cube avec u imposé au bord et la force volumique manufacturée : Newton-Krylov (CG sur la tangente exacte).
fn solve(n: usize) -> Solved {
    let mesh = cube(n);
    let nt = mesh.n_tets();
    let nn = mesh.n_nodes();
    let sim = Simulation::new(mesh.clone(), params(), device()).unwrap();
    let pre = sim.precomputed();
    let mat = &sim.params.material;
    let zero = floats(&vec![0.0; nt], &device());
    let th = Thickness::Uniform(1.0);
    let free: Vec<bool> = mesh.nodes.iter().map(|&p| !is_boundary(p)).collect();
    let exact: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|&p| std::array::from_fn(|k| p[k] + u_exact(p)[k]))
        .collect();
    // Départ : la solution exacte au bord, la configuration de référence ailleurs.
    let mut x: Vec<[f64; 3]> = (0..nn)
        .map(|i| if free[i] { mesh.nodes[i] } else { exact[i] })
        .collect();
    let force = |rows: &[[f64; 3]]| {
        let xv = V3::from_rows(rows, &device());
        elastic::forces_with(pre, mat, &xv, &zero, &th, false)
    };
    let vn0 = force(&x).vn0;
    let vn0 = to_f64(vn0);
    let ext: Vec<[f64; 3]> = (0..nn).map(|i| body_force(mesh.nodes[i]).map(|b| b * vn0[i])).collect();
    let mask = |r: &mut Vec<[f64; 3]>| {
        for i in 0..nn {
            if !free[i] {
                r[i] = [0.0; 3];
            }
        }
    };
    let dot = |a: &[[f64; 3]], b: &[[f64; 3]]| {
        a.iter()
            .zip(b)
            .map(|(p, q)| (0..3).map(|k| p[k] * q[k]).sum::<f64>())
            .sum::<f64>()
    };
    let (mut newton, mut cg_total) = (0, 0);
    for _ in 0..12 {
        let fi = force(&x).force.to_rows();
        let mut r: Vec<[f64; 3]> = (0..nn).map(|i| std::array::from_fn(|k| fi[i][k] + ext[i][k])).collect();
        mask(&mut r);
        let rn = dot(&r, &r).sqrt();
        if rn < 1e-12 {
            break;
        }
        newton += 1;
        // K d = r, K = - dF/dx (symétrique, définie positive près de l'état de référence), CG sans préconditionneur.
        let kmul = |v: &[[f64; 3]]| {
            let xv = V3::from_rows(&x, &device());
            let vv = V3::from_rows(v, &device());
            let mut out: Vec<[f64; 3]> = elastic::force_tangent(pre, mat, &xv, &vv, &zero, &th, None, None)
                .to_rows()
                .into_iter()
                .map(|r| r.map(|c| -c))
                .collect();
            mask(&mut out);
            out
        };
        let mut d = vec![[0.0; 3]; nn];
        let (mut res, mut p) = (r.clone(), r.clone());
        let mut rr = dot(&res, &res);
        for _ in 0..2000 {
            cg_total += 1;
            let kp = kmul(&p);
            let a = rr / dot(&p, &kp);
            for i in 0..nn {
                for k in 0..3 {
                    d[i][k] += a * p[i][k];
                    res[i][k] -= a * kp[i][k];
                }
            }
            let rr2 = dot(&res, &res);
            if rr2.sqrt() < 1e-10 * rn.max(1e-300) {
                break;
            }
            let b = rr2 / rr;
            rr = rr2;
            for i in 0..nn {
                for k in 0..3 {
                    p[i][k] = res[i][k] + b * p[i][k];
                }
            }
        }
        for i in 0..nn {
            for k in 0..3 {
                x[i][k] += d[i][k];
            }
        }
    }
    let fi = force(&x).force.to_rows();
    let rn = (0..nn)
        .filter(|&i| free[i])
        .map(|i| (0..3).map(|k| (fi[i][k] + ext[i][k]).powi(2)).sum::<f64>())
        .sum::<f64>()
        .sqrt();
    assert!(rn < 1e-9, "Newton-Krylov n = {n} : résidu final {rn:e}");
    // Erreur de déplacement aux nœuds intérieurs, poids = volume nodal.
    let (mut max_err, mut l2) = (0.0f64, 0.0f64);
    let vtot: f64 = (0..nn).filter(|&i| free[i]).map(|i| vn0[i]).sum();
    for i in (0..nn).filter(|&i| free[i]) {
        let e = (0..3).map(|k| (x[i][k] - exact[i][k]).powi(2)).sum::<f64>();
        max_err = max_err.max(e.sqrt());
        l2 += e * vn0[i];
    }
    Solved {
        h: 1.0 / n as f64,
        max_err,
        l2_err: (l2 / vtot).sqrt(),
        newton,
        cg: cg_total,
    }
}

#[test]
fn manufactured_solution_converges_at_second_order() {
    let runs: Vec<Solved> = [3, 6, 12].iter().map(|&n| solve(n)).collect();
    for r in &runs {
        eprintln!(
            "h = 1/{:<2}  erreur max {:.3e}  L2 {:.3e}  ({} Newton, {} CG)",
            (1.0 / r.h).round(),
            r.max_err,
            r.l2_err,
            r.newton,
            r.cg
        );
    }
    let order = |a: f64, b: f64| (a / b).log2();
    let l2 = [
        order(runs[0].l2_err, runs[1].l2_err),
        order(runs[1].l2_err, runs[2].l2_err),
    ];
    let mx = [
        order(runs[0].max_err, runs[1].max_err),
        order(runs[1].max_err, runs[2].max_err),
    ];
    eprintln!(
        "ordre L2 : {:.2}, {:.2} ; ordre max : {:.2}, {:.2}",
        l2[0], l2[1], mx[0], mx[1]
    );
    // Éléments P1, solution lisse : ordre 2 en norme L2 du déplacement attendu ; seuil bas pour l'ordre mesuré.
    assert!(l2[1] > 1.8, "ordre L2 entre les deux maillages fins : {:.2}", l2[1]);
    assert!(mx[1] > 1.5, "ordre en norme max : {:.2}", mx[1]);
}
