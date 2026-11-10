//! Gradients de `examples/adjoint_week.rs` contre des différences finies en f64 (P03, tâche 5).
//!
//! Boule synthétique (maillage en voxels, 5 cubes par diamètre), CPU `ndarray` en f64 (`--backend ndarray`) : l'adjoint
//! donne le gradient de la perte totale par rapport aux paramètres (croissance `a`, correction d'épaississement `c`,
//! correction isotrope `k` et radiale `r` de la croissance du cœur) à l'équilibre quasi statique ; `--check` le compare
//! à une différence finie centrée de la perte, équilibre recalculé à chaque point. Le rapport adjoint / différence finie
//! doit valoir 1 : on exige 1 ± 2e-4 au plus petit pas et 1 ± 2e-3 aux autres (mesuré : 1 ± 1e-5 à 1e-4 ; une
//! régression de l'adjoint donne un écart de l'ordre de 1e-2 ou plus), pour chaque terme de perte pris seul (distance des sommets, normales, laplacien, aires, indice de plissement, retour vers
//! la croissance mesurée, régularisation des arêtes) et pour l'ensemble des paramètres.
//!
//! Lancé seulement avec `--features autodiff` (l'exemple `adjoint_week` l'exige) : la CI le fait.
#![cfg(feature = "autodiff")]

mod common;

use common::gyroxide;
use gyroxide_mesh::{generate::voxel_ball, netgen, topology::Surface, TetMesh};
use gyroxide_validate::npy;
use std::path::PathBuf;
use std::process::Command;

const KNOT: f64 = 0.05; // instant du point de contrôle = second nœud de la table de croissance

fn example() -> PathBuf {
    let exe = std::env::current_exe().unwrap(); // target/<profil>/deps/adjoint_gradients-<hash>
    exe.parent().unwrap().parent().unwrap().join("examples/adjoint_week")
}

struct Case {
    dir: PathBuf,
}

impl Case {
    fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }
}

/// Fichiers d'entrée de l'exemple pour une boule : maillage (nœuds de surface d'abord), configuration, tables de
/// croissance, point de contrôle (run de `gyroxide`), cible et dossier `prep/`.
fn build_case(tag: &str) -> Case {
    let dir = std::env::temp_dir().join(format!("gyroxide-adjoint-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("prep")).unwrap();
    let case = Case { dir };

    // Maillage : les nœuds de surface en tête (convention des maillages de l'atlas : les NS premiers nœuds).
    let ball = voxel_ball(5, 1.0);
    let surf = Surface::new(&ball);
    let ns = surf.len();
    let mut order: Vec<u32> = surf.nodes.clone();
    order.extend((0..ball.n_nodes() as u32).filter(|&i| surf.index_of[i as usize] == u32::MAX));
    let mut new_of = vec![0u32; ball.n_nodes()];
    for (new, &old) in order.iter().enumerate() {
        new_of[old as usize] = new as u32;
    }
    let mesh = TetMesh {
        nodes: order.iter().map(|&o| ball.nodes[o as usize]).collect(),
        tets: ball.tets.iter().map(|t| t.map(|n| new_of[n as usize])).collect(),
        faces: ball.faces.iter().map(|f| f.map(|n| new_of[n as usize])).collect(),
    };
    netgen::write(&mesh, case.dir.join("ball.mesh")).unwrap();
    let n = mesh.n_nodes();

    // Tables : a (croissance mesurée, 0 au premier nœud) et h (épaisseur), [2 nœuds, n nœuds].
    let a_row: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|p| 0.25 + 0.08 * (3.0 * p[0] + 2.0 * p[1]).sin() + 0.05 * p[2])
        .collect();
    let a: Vec<f64> = vec![0.0; n].into_iter().chain(a_row.iter().copied()).collect();
    npy::write_f64(case.dir.join("a.npy"), &[2, n], &a).unwrap();
    npy::write_f64(case.dir.join("h.npy"), &[2, n], &vec![0.3; 2 * n]).unwrap();
    // Tenseur de croissance du cœur U [2 nœuds, n nœuds, 6] (xx, yy, zz, xy, xz, yz) : identité, puis un étirement
    // anisotrope lisse ; sans lui, les champs k et r n'ont rien à corriger.
    let core: Vec<f64> = [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
        .iter()
        .cycle()
        .take(n)
        .flatten()
        .copied()
        .chain(mesh.nodes.iter().flat_map(|p| {
            [
                1.04 + 0.01 * p[0],
                1.03 + 0.01 * p[1],
                1.05,
                0.01 * p[2],
                0.0,
                0.005 * p[0],
            ]
        }))
        .collect();
    npy::write_f64(case.dir.join("core.npy"), &[2, n, 6], &core).unwrap();

    let (spacing, density, bulk, step_factor) = (0.2f64, 0.0025f64, 5.0f64, 0.025f64);
    let dt = step_factor * (density * spacing * spacing / bulk).sqrt();
    let steps = (KNOT / dt).round() as usize;
    let config = format!(
        r#"name = "boule synthétique (test des gradients)"
normalization = "none"

[material]
mu_gray = 1.0
mu_white = 0.5
bulk = {bulk}
eps = 0.1
k_inverted = 0.5

[dynamics]
mesh_spacing = {spacing}
density = {density}
damping = 0.5
step_factor = {step_factor}
t_start = 0.0
t_end = {KNOT}

[schedule]
law = "nodal"
t = [0.0, {t1}]
a = "{a}"
h = "{h}"
a_scale = 1.0
h_scale = 1
thickening = true
core = "{core}"

[contact]
enabled = false
stiffness_factor = 10.0
repulsive_skin = 0.2
proximity_skin = 0.6
check_every = 10

[growth_mask]
kind = "uniform"
"#,
        t1 = steps as f64 * dt,
        a = case.path("a.npy"),
        h = case.path("h.npy"),
        core = case.path("core.npy"),
    );
    std::fs::write(case.dir.join("params.toml"), config).unwrap();

    // Point de contrôle : le run explicite de `gyroxide` (CPU f64) jusqu'au second nœud.
    gyroxide(&[
        "run",
        "--config",
        &case.path("params.toml"),
        "--mesh",
        &case.path("ball.mesh"),
        "--backend",
        "ndarray",
        "--steps",
        &steps.to_string(),
        "--checkpoint-every",
        &steps.to_string(),
        "--out",
        &case.path("run"),
    ]);

    // prep/ : cible (positions de surface déformées autrement que le modèle), arêtes, faces, cortex, propriétaire, theta0.
    let target: Vec<[f64; 3]> = (0..ns)
        .map(|i| {
            let p = mesh.nodes[i];
            let s = 1.18 + 0.05 * (4.0 * p[0] - 3.0 * p[2]).cos();
            [s * p[0] + 0.02, s * p[1] - 0.01, s * p[2] + 0.015 * (5.0 * p[1]).sin()]
        })
        .collect();
    npy::write_rows3(case.dir.join("prep/target.npy"), &target).unwrap();
    let mut edges: Vec<[f64; 2]> = Vec::new();
    for f in &mesh.faces {
        for (x, y) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            edges.push([x.min(y) as f64, x.max(y) as f64]);
        }
    }
    edges.sort_by(|p, q| p.partial_cmp(q).unwrap());
    edges.dedup();
    let flat: Vec<f64> = edges.iter().flatten().copied().collect();
    npy::write_f64(case.dir.join("prep/edges.npy"), &[edges.len(), 2], &flat).unwrap();
    let faces: Vec<f64> = mesh.faces.iter().flatten().map(|&v| v as f64).collect();
    npy::write_f64(case.dir.join("prep/faces.npy"), &[mesh.n_faces(), 3], &faces).unwrap();
    npy::write_f64(case.dir.join("prep/cortex.npy"), &[ns], &vec![1.0; ns]).unwrap();
    let owner: Vec<f64> = (0..n)
        .map(|i| {
            if i < ns {
                i as f64
            } else {
                (0..ns)
                    .min_by(|&p, &q| {
                        let d = |j: usize| {
                            (0..3)
                                .map(|k| (mesh.nodes[j][k] - mesh.nodes[i][k]).powi(2))
                                .sum::<f64>()
                        };
                        d(p).partial_cmp(&d(q)).unwrap()
                    })
                    .unwrap() as f64
            }
        })
        .collect();
    npy::write_f64(case.dir.join("prep/owner.npy"), &[n], &owner).unwrap();
    npy::write_f64(case.dir.join("prep/theta0.npy"), &[ns], &a_row[..ns]).unwrap();
    // Table « mesurée » décalée de la croissance de départ (terme de retour vers la mesure), et theta de départ
    // perturbé (croissance et champs c, k, r : les termes de régularisation ont alors un gradient non nul).
    let measured: Vec<f64> = vec![0.0; n]
        .into_iter()
        .chain(a_row.iter().map(|v| 0.9 * v + 0.01))
        .collect();
    npy::write_f64(case.dir.join("measured.npy"), &[2, n], &measured).unwrap();
    let theta_init: Vec<f64> = (0..4 * ns)
        .map(|i| {
            let (field, v) = (i / ns, i % ns);
            let w = (0.7 * v as f64).sin();
            match field {
                0 => a_row[v] + 0.02 * w,
                1 => 0.03 * w,
                2 => 0.02 * (0.3 * v as f64).cos(),
                _ => 0.02 * w * (0.5 * v as f64).cos(),
            }
        })
        .collect();
    npy::write_f64(case.dir.join("theta_init.npy"), &[4 * ns], &theta_init).unwrap();
    case
}

/// Lance `adjoint_week --check` avec les options données ; rend les trois rapports adjoint / différence finie.
fn check(tag: &str, extra: &[&str], steps: usize) -> Vec<f64> {
    let case = build_case(tag);
    let out = Command::new(example())
        .args([
            &case.path("params.toml"),
            &case.path("ball.mesh"),
            &case.path("run/checkpoint"),
            &case.path("prep"),
            &case.path("out"),
            "--backend",
            "ndarray",
            "--theta0-table",
            "--relax",
            "100",
            "--newton",
            "30",
            "--force-tol",
            "1e-11",
            "--newton-rel",
            "1e-12",
            "--newton-cg",
            "400",
            "--cg",
            "600",
            "--cg-tol",
            "1e-10",
            "--shift",
            "1e-12",
            "--check",
            "--check-steps",
            &steps.to_string(),
        ])
        .args(extra.iter().map(|a| match *a {
            "{measured}" => case.path("measured.npy"),
            "{theta_init}" => case.path("theta_init.npy"),
            a => a.to_owned(),
        }))
        .output()
        .expect("exemple adjoint_week introuvable (cargo test --features autodiff)");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "adjoint_week {extra:?}\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ratios: Vec<f64> = stdout
        .lines()
        .filter(|l| l.starts_with("check h = "))
        .map(|l| {
            let r = l.split("ratio ").nth(1).unwrap();
            r.split(';').next().unwrap().trim().parse().unwrap()
        })
        .collect();
    if std::env::var_os("GYROXIDE_VERBOSE").is_some() {
        eprintln!("{stdout}");
    }
    eprintln!("{tag} {extra:?} : rapports {ratios:?}");
    let _ = std::fs::remove_dir_all(&case.dir);
    assert_eq!(ratios.len(), steps, "{steps} différences finies attendues :\n{stdout}");
    ratios
}

fn assert_ratios(tag: &str, extra: &[&str]) {
    assert_ratios_within(tag, extra, 2e-4, 1);
}

/// `strict` : écart permis au plus petit pas. Les termes de normales et de laplacien ne sont pas invariants par rotation
/// et la boule libre a un mode de rotation mou : l'équilibre y dérive (bruit d'arrondi amplifié, de l'ordre de 1e-3
/// relatif sur la perte), ce que l'adjoint ne voit pas ; leur tolérance est de 5e-3 à tous les pas.
fn assert_ratios_within(tag: &str, extra: &[&str], strict: f64, steps: usize) {
    // Trois pas de différence finie (h = 0,5 ; 1 ; 2) : le plus petit est le plus juste (erreur d'ordre 2 en h), les
    // autres bornent la non-linéarité de la perte le long de la direction.
    let r = check(tag, extra, steps);
    assert!(
        (r[0] - 1.0).abs() < strict,
        "{tag} {extra:?} : rapport adjoint / différence finie {r:?} (h = 0,5 : 1 ± {strict} attendu)"
    );
    assert!(
        r.iter().all(|v| (v - 1.0).abs() < strict.max(2e-3)),
        "{tag} {extra:?} : rapport adjoint / différence finie {r:?} (tolérance {})",
        strict.max(2e-3)
    );
}

// Une exécution par terme de perte (les termes sont isolés : un poids nul pour les autres) et une pour l'ensemble des
// paramètres (a, c, k, r). `--rigid` : la perte de sommets invariante par mouvement rigide (voir la note ci-dessous).

#[test]
fn gradient_growth_vertex_loss() {
    // Les trois pas de différence finie, pour voir l'erreur d'ordre 2 en h ; les autres tests n'en font qu'un (h = 0,5).
    assert_ratios_within("vertex", &["--rigid"], 2e-4, 3);
}

#[test]
fn gradient_growth_normal_loss() {
    assert_ratios_within("normal", &["--rigid", "--w-vertex", "0", "--w-normal", "1"], 5e-3, 1);
}

#[test]
fn gradient_growth_laplacian_loss() {
    assert_ratios_within("lap", &["--rigid", "--w-vertex", "0", "--w-lap", "1"], 5e-3, 1);
}

#[test]
fn gradient_growth_area_loss() {
    assert_ratios("area", &["--rigid", "--w-vertex", "0", "--w-area", "1"]);
}

#[test]
fn gradient_growth_folding_index_loss() {
    assert_ratios("fi", &["--rigid", "--w-vertex", "0", "--fi-weight", "1"]);
}

#[test]
fn gradient_growth_pull_to_measured() {
    assert_ratios("alpha", &["--rigid", "--alpha", "1", "--measured", "{measured}"]);
}

#[test]
fn gradient_fields_c_k_r_with_edge_regularisation() {
    assert_ratios(
        "fields",
        &[
            "--rigid",
            "--fit-fields",
            "c,k,r",
            "--theta-init",
            "{theta_init}",
            "--beta",
            "1e-2",
        ],
    );
}

#[test]
fn gradient_each_field_separately() {
    // Champs seuls (départ à 0) : la perte de sommets donne un gradient non nul pour chacun ; correction d'épaisseur
    // multiplicative (`--thick-mult`, celle de la chaîne de l'article) pour c.
    for fields in ["c", "k", "r"] {
        assert_ratios(&format!("champ-{fields}"), &["--rigid", "--fit-fields", fields]);
    }
    assert_ratios("champ-c-mult", &["--rigid", "--fit-fields", "c", "--thick-mult"]);
}
