//! Small synthetic meshes for tests and smoke runs.

use crate::topology::extract_boundary;
use crate::TetMesh;
use std::collections::HashMap;

/// A "voxel ball": the cubes of a regular grid of spacing `2r / n` whose
/// centre lies inside the sphere of radius `r`, each split into 6
/// tetrahedra along the same main diagonal (Kuhn subdivision, conforming).
pub fn voxel_ball(n: usize, r: f64) -> TetMesh {
    let h = 2.0 * r / n as f64;
    let coord = |i: usize| -r + i as f64 * h;
    let mut id: HashMap<[usize; 3], u32> = HashMap::new();
    let mut nodes = Vec::new();
    let mut tets = Vec::new();
    // Six paths from corner 000 to 111, one per permutation of the axes.
    const PERMS: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let c = [coord(i) + h / 2.0, coord(j) + h / 2.0, coord(k) + h / 2.0];
                if c[0] * c[0] + c[1] * c[1] + c[2] * c[2] > r * r {
                    continue;
                }
                for p in PERMS {
                    let mut v = [i, j, k];
                    let mut ids = [0u32; 4];
                    for (s, slot) in ids.iter_mut().enumerate() {
                        if s > 0 {
                            v[p[s - 1]] += 1;
                        }
                        *slot = *id.entry(v).or_insert_with(|| {
                            nodes.push([coord(v[0]), coord(v[1]), coord(v[2])]);
                            (nodes.len() - 1) as u32
                        });
                    }
                    tets.push(ids);
                }
            }
        }
    }
    let mut mesh = TetMesh {
        nodes,
        tets,
        faces: Vec::new(),
    };
    mesh.fix_tet_orientation();
    mesh.faces = extract_boundary(&mesh);
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::outward_fraction;

    #[test]
    fn voxel_ball_is_valid_and_closed() {
        let m = voxel_ball(6, 1.0);
        m.validate().unwrap();
        assert!(m.n_tets() > 0 && m.n_faces() > 0);
        assert!(outward_fraction(&m) > 0.95);
        // Closed surface: every edge shared by exactly two triangles.
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        for f in &m.faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2));
        // Total volume equals the number of kept cubes times h^3.
        let vol: f64 = (0..m.n_tets()).map(|i| m.tet_volume(i)).sum();
        let h: f64 = 2.0 / 6.0;
        let cubes = m.n_tets() / 6;
        assert!((vol - cubes as f64 * h.powi(3)).abs() < 1e-12);
    }
}
