//! Unit tests of the reference tensor implementation, in f64 on the CPU.

use burn::backend::NdArray;
use gyroxide_mesh::{generate::voxel_ball, topology::extract_boundary, TetMesh};
use gyroxide_model::{GrowthMask, Preset};
use gyroxide_sim::algebra::{floats, to_f64, M3, V3};
use gyroxide_sim::elastic::{self, sym_eigen};
use gyroxide_sim::trace::NoTrace;
use gyroxide_sim::Simulation;

type B = NdArray<f64>;

fn device() -> burn::backend::ndarray::NdArrayDevice {
    Default::default()
}

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn single_tet() -> TetMesh {
    let mut m = TetMesh {
        nodes: vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
        tets: vec![[0, 1, 2, 3]],
        faces: vec![],
    };
    m.faces = extract_boundary(&m);
    m
}

fn params_no_contact() -> gyroxide_model::Params {
    let mut p = Preset::Tallinen2016.params();
    p.growth_mask = GrowthMask::Uniform;
    p.contact.enabled = false;
    p.contact.midplane = None;
    p
}

fn max_abs(rows: &[[f64; 3]]) -> f64 {
    rows.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()))
}

#[test]
fn undeformed_mesh_has_zero_force() {
    let mesh = voxel_ball(5, 1.0);
    let sim = Simulation::<B>::new(mesh, params_no_contact(), device()).unwrap();
    let g = floats::<B>(&vec![0.0; sim.mesh.n_tets()], &device());
    let out = elastic::forces(sim.precomputed(), &sim.params.material, &sim.x, &g, 0.05, true);
    let f = out.force.to_rows();
    assert!(max_abs(&f) < 1e-12, "max |f| = {}", max_abs(&f));
    let (ue, ninv) = out.energy.unwrap();
    assert!(to_f64(ue)[0].abs() < 1e-14);
    assert_eq!(to_f64(ninv)[0], 0.0);
}

#[test]
fn free_growth_is_stress_free() {
    // One tetrahedron: every vertex has the same growth normal, so placing
    // the nodes at G X must cancel the force exactly (F = I, J = Ja = 1).
    let mesh = single_tet();
    let sim = Simulation::<B>::new(mesh.clone(), params_no_contact(), device()).unwrap();
    let n = sim.precomputed().growth_normal.to_rows()[0];
    let s = 0.7;
    let grown: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|x| {
            let d = n[0] * x[0] + n[1] * x[1] + n[2] * x[2];
            std::array::from_fn(|k| x[k] + s * (x[k] - n[k] * d))
        })
        .collect();
    let x = V3::<B>::from_rows(&grown, &device());
    let g = floats::<B>(&[s], &device());
    let out = elastic::forces(sim.precomputed(), &sim.params.material, &x, &g, 0.05, false);
    assert!(max_abs(&out.force.to_rows()) < 1e-12);
    // ... while the same growth without deformation pushes outwards.
    let out0 = elastic::forces(sim.precomputed(), &sim.params.material, &sim.x, &g, 0.05, false);
    assert!(max_abs(&out0.force.to_rows()) > 1e-6);
}

#[test]
fn inverted_tet_has_finite_restoring_force() {
    let mesh = single_tet();
    let sim = Simulation::<B>::new(mesh.clone(), params_no_contact(), device()).unwrap();
    let mut x = mesh.nodes.clone();
    x[3][2] = -0.005; // apex pushed through the opposite face
    let xv = V3::<B>::from_rows(&x, &device());
    let g = floats::<B>(&[0.0], &device());
    let out = elastic::forces(sim.precomputed(), &sim.params.material, &xv, &g, 0.05, true);
    let f = out.force.to_rows();
    assert!(f.iter().flatten().all(|v| v.is_finite()));
    assert!(f[3][2] > 0.0, "apex force should point back up, got {:?}", f[3]);
    assert_eq!(to_f64(out.energy.unwrap().1)[0], 1.0);
    // Forces are internal: they sum to zero.
    let sum: [f64; 3] = std::array::from_fn(|k| f.iter().map(|r| r[k]).sum());
    assert!(max_abs(&[sum]) < 1e-12);
}

#[test]
fn symmetric_eigen_decomposition() {
    let mut r = rng(7);
    let n = 200;
    let mut rows = vec![[[0.0; 3]; 3]; n];
    for (i, m) in rows.iter_mut().enumerate() {
        let g: [[f64; 3]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| r()));
        for a in 0..3 {
            for b in 0..3 {
                m[a][b] = (0..3).map(|k| g[k][a] * g[k][b]).sum();
            }
        }
        if i % 10 == 0 {
            // Degenerate cases: multiples of identity and double eigenvalues.
            *m = [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, if i % 20 == 0 { 2.0 } else { 0.5 }],
            ];
        }
    }
    let comp = |a: usize, b: usize| floats::<B>(&rows.iter().map(|m| m[a][b]).collect::<Vec<_>>(), &device());
    let c = M3::<B> {
        m: std::array::from_fn(|a| std::array::from_fn(|b| comp(a, b))),
    };
    let e = sym_eigen(&c);
    let vals: Vec<Vec<f64>> = e.values.iter().map(|v| to_f64(v.clone())).collect();
    let cols: Vec<Vec<[f64; 3]>> = (0..3).map(|k| e.vectors.col(k).to_rows()).collect();
    let lmin = to_f64(e.lambda_min.clone());
    for i in 0..n {
        assert!(
            vals[0][i] <= vals[1][i] + 1e-12 && vals[1][i] <= vals[2][i] + 1e-12,
            "not sorted at {i}"
        );
        assert!((lmin[i] - vals[0][i]).abs() < 1e-9 * (1.0 + vals[2][i]));
        for k in 0..3 {
            let v = cols[k][i];
            let cv: [f64; 3] = std::array::from_fn(|a| (0..3).map(|b| rows[i][a][b] * v[b]).sum());
            let res = (0..3).map(|a| (cv[a] - vals[k][i] * v[a]).powi(2)).sum::<f64>().sqrt();
            assert!(res < 1e-9 * (1.0 + vals[2][i]), "residual {res} at {i},{k}");
            assert!(((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn branches_agree_at_the_stretch_floor() {
    // At l1 = eps the diagonalised branch is the first-order expansion of the
    // regular one around that point: both must give the same P.
    let mesh = single_tet();
    let p = params_no_contact();
    let sim = Simulation::<B>::new(mesh.clone(), p.clone(), device()).unwrap();
    let eps = p.material.eps;
    // F = R diag(eps, 1.1, 0.9) with a rotation R about (1,1,1).
    let (c, s) = (0.6f64.cos(), 0.6f64.sin());
    let k = [1.0 / 3f64.sqrt(); 3];
    let r: [[f64; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let kron = if i == j { 1.0 } else { 0.0 };
            let cross = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
            c * kron + s * cross[i][j] + (1.0 - c) * k[i] * k[j]
        })
    });
    let d = [eps, 1.1, 0.9];
    let f: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| r[i][j] * d[j]));
    let x: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|x0| std::array::from_fn(|i| (0..3).map(|j| f[i][j] * x0[j]).sum()))
        .collect();
    let xv = V3::<B>::from_rows(&x, &device());
    let g = floats::<B>(&[0.0], &device());
    // Just above and just below the floor: forces must be continuous.
    let out = elastic::forces(sim.precomputed(), &p.material, &xv, &g, 0.05, false)
        .force
        .to_rows();
    let mut x2 = x.clone();
    let shrink = 1.0 - 1e-7;
    for p in x2.iter_mut() {
        // scale along R e_x by (1 - 1e-7): crosses lambda_min = eps^2
        let ax: [f64; 3] = std::array::from_fn(|i| r[i][0]);
        let dd = p[0] * ax[0] + p[1] * ax[1] + p[2] * ax[2];
        for i in 0..3 {
            p[i] -= (1.0 - shrink) * dd * ax[i];
        }
    }
    let x2v = V3::<B>::from_rows(&x2, &device());
    let out2 = elastic::forces(sim.precomputed(), &p.material, &x2v, &g, 0.05, false)
        .force
        .to_rows();
    let scale = max_abs(&out);
    let diff = out
        .iter()
        .zip(&out2)
        .map(|(a, b)| max_abs(&[[a[0] - b[0], a[1] - b[1], a[2] - b[2]]]))
        .fold(0.0, f64::max);
    assert!(diff < 1e-4 * scale, "jump {diff} vs scale {scale}");
}

#[test]
fn contact_closest_point_matches_scalar() {
    use gyroxide_sim::contact::closest_point_triangle;
    // Indirectly through penalty forces: compare against a CPU evaluation.
    let mut r = rng(3);
    let a = 0.01;
    let mesh = voxel_ball(4, 1.0);
    let x: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| std::array::from_fn(|k| p[k] + 0.01 * r()))
        .collect();
    let surf: Vec<u32> = gyroxide_mesh::topology::Surface::new(&mesh).nodes;
    let pairs = gyroxide_sim::contact::build_pairs(&mesh, &surf, &x, 1.0, 10.0);
    assert!(!pairs.is_empty());
    let gr = vec![1.0; mesh.n_nodes()];
    let dev = device();
    let gp = gyroxide_sim::contact::upload_pairs::<B>(&mesh, &gr, &pairs, &dev).unwrap();
    let hc = 0.3; // large skin so that most pairs are active
    let kc = 50.0;
    let got = gyroxide_sim::contact::penalty_forces(&gp, &V3::from_rows(&x, &dev), mesh.n_nodes(), hc, kc, a).to_rows();
    let mut want = vec![[0.0; 3]; mesh.n_nodes()];
    for &(pt, tri) in &pairs {
        let f = mesh.faces[tri as usize];
        let p = x[pt as usize];
        let (q, uvw) = closest_point_triangle(p, x[f[0] as usize], x[f[1] as usize], x[f[2] as usize]);
        let cc = gyroxide_mesh::sub(q, p);
        let rc = gyroxide_mesh::norm(cc);
        if rc >= hc {
            continue;
        }
        let nt = gyroxide_mesh::normalize(gyroxide_mesh::cross(
            gyroxide_mesh::sub(x[f[1] as usize], x[f[0] as usize]),
            gyroxide_mesh::sub(x[f[2] as usize], x[f[0] as usize]),
        ));
        let mut fnv = gyroxide_mesh::scale(gyroxide_mesh::normalize(cc), (rc - hc) / hc * kc * a * a);
        let d = gyroxide_mesh::dot(fnv, nt);
        if d < 0.0 {
            fnv = gyroxide_mesh::sub(fnv, gyroxide_mesh::scale(nt, 2.0 * d));
        }
        for k in 0..3 {
            for (j, &n) in f.iter().enumerate() {
                want[n as usize][k] -= fnv[k] * uvw[j];
            }
            want[pt as usize][k] += fnv[k];
        }
    }
    let diff = got
        .iter()
        .zip(&want)
        .map(|(g, w)| max_abs(&[[g[0] - w[0], g[1] - w[1], g[2] - w[2]]]))
        .fold(0.0, f64::max);
    assert!(diff < 1e-12 * (1.0 + max_abs(&want)), "diff {diff}");
}

#[test]
fn growing_ball_smoke_run() {
    let mesh = voxel_ball(6, 1.0);
    let mut p = Preset::BrainGrowth.params();
    p.growth_mask = GrowthMask::Uniform;
    p.dynamics.mesh_spacing = 2.0 / 6.0; // coarse mesh: a sets dt and the contact skins
    let mut sim = Simulation::<B>::new(mesh, p, device()).unwrap();
    sim.run(Some(200), &mut NoTrace).unwrap();
    let snap = sim.step_recorded();
    let x = &snap.x;
    assert!(x.iter().flatten().all(|v| v.is_finite()));
    assert!(snap.diagnostics.elastic > 0.0);
    assert!(sim.n_rebuilds >= 1);
}

#[test]
fn force_tangent_matches_finite_differences() {
    let mesh = voxel_ball(5, 1.0);
    let mut params = params_no_contact();
    params.material.wm_growth = 0.3;
    let sim = Simulation::<B>::new(mesh, params, device()).unwrap();
    let mut r = rng(7);
    let x0 = sim.x.to_rows();
    let x: Vec<[f64; 3]> = x0.iter().map(|p| [0, 1, 2].map(|d| p[d] + 0.01 * r())).collect();
    let v: Vec<[f64; 3]> = x0.iter().map(|_| [r(), r(), r()]).collect();
    let g: Vec<f64> = (0..sim.mesh.n_tets()).map(|_| 0.15 + 0.15 * r()).collect();
    let (pre, mat) = (sim.precomputed(), &sim.params.material);
    let gt = floats::<B>(&g, &device());
    let th = elastic::Thickness::Uniform(0.05);
    let nt = sim.mesh.n_tets();
    let comp = |d: f64, r: &mut dyn FnMut() -> f64| floats::<B>(&(0..nt).map(|_| d + 0.1 * r()).collect::<Vec<_>>(), &device());
    let (xx, yy, zz) = (comp(1.3, &mut r), comp(1.2, &mut r), comp(1.4, &mut r));
    let (xy, xz, yz) = (comp(0.0, &mut r), comp(0.0, &mut r), comp(0.0, &mut r));
    let core = M3 { m: [[xx, xy.clone(), xz.clone()], [xy, yy, yz.clone()], [xz, yz, zz]] };
    let rad = comp(-0.2, &mut r);
    let f = |y: &[[f64; 3]]| elastic::forces_regular(pre, mat, &V3::from_rows(y, &device()), &gt, &th, Some(&core), Some(&rad)).to_rows();
    let eps = 1e-6;
    let shift = |s: f64| -> Vec<[f64; 3]> { x.iter().zip(&v).map(|(a, b)| [0, 1, 2].map(|d| a[d] + s * b[d])).collect() };
    let (fp, fm) = (f(&shift(eps)), f(&shift(-eps)));
    let fd: Vec<[f64; 3]> = fp.iter().zip(&fm).map(|(a, b)| [0, 1, 2].map(|d| (a[d] - b[d]) / (2.0 * eps))).collect();
    let an = elastic::force_tangent(pre, mat, &V3::from_rows(&x, &device()), &V3::from_rows(&v, &device()), &gt, &th, Some(&core), Some(&rad)).to_rows();
    let err = an.iter().zip(&fd).flat_map(|(a, b)| (0..3).map(move |d| (a[d] - b[d]).abs())).fold(0.0f64, f64::max);
    assert!(err < 1e-6 * max_abs(&fd), "max error {err:.3e}, max |fd| {:.3e}", max_abs(&fd));
}

#[test]
fn core_growth_is_stress_free() {
    // G = I + s (I - n n^T) + (1 - g_m) b I: nodes placed at G X carry no force.
    let mesh = single_tet();
    let mut params = params_no_contact();
    params.material.wm_growth = 0.4;
    let sim = Simulation::<B>::new(mesh.clone(), params, device()).unwrap();
    let n = sim.precomputed().growth_normal.to_rows()[0];
    let (s, h) = (0.3, 0.0004);
    let g = floats::<B>(&[s], &device());
    let mat = &sim.params.material;
    let gm = to_f64(elastic::forces(sim.precomputed(), mat, &sim.x, &g, h, false).gm)[0];
    assert!(gm > 0.0 && gm < 1.0, "g_m = {gm}");
    let b = (1.0 - gm) * mat.wm_growth;
    let grown: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|x| {
            let d = n[0] * x[0] + n[1] * x[1] + n[2] * x[2];
            std::array::from_fn(|k| x[k] + s * (x[k] - n[k] * d) + b * x[k])
        })
        .collect();
    let x = V3::<B>::from_rows(&grown, &device());
    let out = elastic::forces(sim.precomputed(), mat, &x, &g, h, false);
    assert!(max_abs(&out.force.to_rows()) < 1e-12, "{:?}", out.force.to_rows());
}

#[test]
fn thickening_is_stress_free() {
    // G = I + s (I - n n^T) + g_m c n n^T: nodes placed at G X carry no force.
    let mesh = single_tet();
    let sim = Simulation::<B>::new(mesh.clone(), params_no_contact(), device()).unwrap();
    let n = sim.precomputed().growth_normal.to_rows()[0];
    let (s, c, h) = (0.3, -0.25, 0.05);
    let g = floats::<B>(&[s], &device());
    let mat = &sim.params.material;
    let th = elastic::Thickness::Uniform(h);
    let gm = to_f64(elastic::forces(sim.precomputed(), mat, &sim.x, &g, h, false).gm)[0];
    let grown: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|x| {
            let d = n[0] * x[0] + n[1] * x[1] + n[2] * x[2];
            std::array::from_fn(|k| x[k] + s * (x[k] - n[k] * d) + gm * c * n[k] * d)
        })
        .collect();
    let x = V3::<B>::from_rows(&grown, &device());
    let rad = floats::<B>(&[c], &device());
    let out = elastic::forces_full(sim.precomputed(), mat, &x, &g, None, None, Some(&rad), &th, false);
    assert!(max_abs(&out.force.to_rows()) < 1e-12, "{:?}", out.force.to_rows());
}
