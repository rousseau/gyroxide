//! Diagnostic de l'étape E40 : les forces de contact sont-elles reproductibles au bit près d'une exécution à l'autre ?
//!
//! Usage : `contact_determinisme [n=24] [reps=10]`, à lancer **dans deux processus** et à comparer les lignes `HASH`.
//! Une maille de boule voxelisée (`n` voxels par diamètre) est perturbée par un bruit déterministe ; les paires de contact sont construites
//! et envoyées sur le GPU (wgpu), puis, pour chaque étage :
//!   - `gather_sum_dim` : la table ELL appliquée à des contributions fixes avec `sum_dim` (le chemin historique) ;
//!   - `gather_ordered` : la même, avec la somme colonne par colonne (`ordered_sum`) ;
//!   - `forces_sum_dim` / `forces_ordered` : `penalty_forces` complète, des deux façons ;
//!   - `tangent_sum_dim` / `tangent_ordered` : `penalty_tangent` (produit K·v de la partie contact), des deux façons.
//! Chaque ligne donne le nombre de valeurs distinctes de la somme de contrôle sur les `reps` répétitions du processus, et la somme de
//! contrôle (FNV-1a des bits en `f64` des valeurs lues) de la première répétition.

use burn::tensor::{Device, DeviceKind};
use gyroxide_mesh::{generate::voxel_ball, topology::Surface};
use gyroxide_sim::algebra::{floats, to_f64, T1, V3};
use gyroxide_sim::contact::{build_pairs, gather, penalty_forces, penalty_tangent, upload_pairs};
use std::collections::BTreeSet;

fn fnv(values: &[f64]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(24);
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(10);
    let device = Device::wgpu(DeviceKind::DefaultDevice);
    let mesh = voxel_ball(n, 1.0);
    let mut s = 12345u64;
    let mut noise = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let h = 2.0 / n as f64;
    let x: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| std::array::from_fn(|k| p[k] + 0.2 * h * noise()))
        .collect();
    let dx: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|_| std::array::from_fn(|_| 0.1 * h * noise()))
        .collect();
    let surface = Surface::new(&mesh).nodes;
    let pairs = build_pairs(&mesh, &surface, &x, 8.0 * h, 3.0 * h);
    let gr = vec![1.0; mesh.n_nodes()];
    let mut gp = upload_pairs(&mesh, &gr, &pairs, &device).expect("des paires de contact");
    println!(
        "n = {n} : {} nœuds, {} paires, largeur ELL {}",
        mesh.n_nodes(),
        gp.n_pairs,
        gp.ell_width
    );
    let (hc, kc, a) = (0.6 * h, 50.0, h);
    let xv = V3::from_rows(&x, &device);
    let dxv = V3::from_rows(&dx, &device);
    // contributions fixes, de l'ordre de grandeur de forces : pas de noyau avant la somme, seule la somme est en cause
    let cols_fixes = |k: usize| -> Vec<T1> {
        (0..4)
            .map(|b| {
                let v: Vec<f64> = (0..gp.n_pairs)
                    .map(|i| ((i * 7 + b * 13 + k * 3) % 1009) as f64 * 1e-3 - 0.5 + 1e-7 * (i as f64))
                    .collect();
                floats(&v, &device)
            })
            .collect()
    };
    let n_nodes = mesh.n_nodes();
    for ordered in [false, true] {
        gp.ordered_sum = ordered;
        let nom = if ordered { "ordered" } else { "sum_dim" };
        let mut h_gather = BTreeSet::new();
        let mut h_forces = BTreeSet::new();
        let mut h_tangent = BTreeSet::new();
        let (mut p_gather, mut p_forces, mut p_tangent) = (0, 0, 0);
        for r in 0..reps {
            let g: Vec<f64> = (0..3)
                .flat_map(|k| to_f64(gather(cols_fixes(k), &gp, n_nodes)))
                .collect();
            let hg = fnv(&g);
            let f = penalty_forces(&gp, &xv, n_nodes, hc, kc, a).to_rows();
            let hf = fnv(&f.iter().flatten().copied().collect::<Vec<_>>());
            let t = penalty_tangent(&gp, &xv, &dxv, n_nodes, hc, kc, a).to_rows();
            let ht = fnv(&t.iter().flatten().copied().collect::<Vec<_>>());
            if r == 0 {
                (p_gather, p_forces, p_tangent) = (hg, hf, ht);
            }
            h_gather.insert(hg);
            h_forces.insert(hf);
            h_tangent.insert(ht);
        }
        println!("HASH gather_{nom} distinct={} premier={p_gather:016x}", h_gather.len());
        println!("HASH forces_{nom} distinct={} premier={p_forces:016x}", h_forces.len());
        println!(
            "HASH tangent_{nom} distinct={} premier={p_tangent:016x}",
            h_tangent.len()
        );
    }
}
