//! Connectivity derived from a [`TetMesh`]: surface node maps, reference
//! surface normals and the node -> (tetrahedron, local vertex) incidence in
//! ELL format used by the GPU gather assembly.

use crate::{cross, dot, normalize, sub, TetMesh, Vec3};
use std::collections::HashMap;

/// Surface node index maps (the C code's `SN` and `SNb`).
#[derive(Debug, Clone)]
pub struct Surface {
    /// Surface index -> mesh node index (`SN`).
    pub nodes: Vec<u32>,
    /// Mesh node index -> surface index, `u32::MAX` for interior nodes (`SNb`).
    pub index_of: Vec<u32>,
}

impl Surface {
    pub fn new(mesh: &TetMesh) -> Self {
        let mut on = vec![false; mesh.n_nodes()];
        for f in &mesh.faces {
            for &n in f {
                on[n as usize] = true;
            }
        }
        let mut nodes = Vec::new();
        let mut index_of = vec![u32::MAX; mesh.n_nodes()];
        for (i, &s) in on.iter().enumerate() {
            if s {
                index_of[i] = nodes.len() as u32;
                nodes.push(i as u32);
            }
        }
        Self { nodes, index_of }
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Area-weighted vertex normals of the surface, indexed by surface node,
/// computed exactly like `N0` in Brains.cpp (sum of un-normalised face
/// normals, then normalised).
pub fn surface_normals(mesh: &TetMesh, surf: &Surface) -> Vec<Vec3> {
    let mut n0 = vec![[0.0; 3]; surf.len()];
    for f in &mesh.faces {
        let [a, b, c] = f.map(|i| mesh.nodes[i as usize]);
        let nt = cross(sub(b, a), sub(c, a));
        for &v in f {
            let s = &mut n0[surf.index_of[v as usize] as usize];
            for k in 0..3 {
                s[k] += nt[k];
            }
        }
    }
    n0.into_iter().map(normalize).collect()
}

/// Boundary triangles of a tetrahedral mesh (faces owned by exactly one
/// tetrahedron), oriented outwards.
pub fn extract_boundary(mesh: &TetMesh) -> Vec<[u32; 3]> {
    const LOCAL: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
    let mut count: HashMap<[u32; 3], (u32, [u32; 3], u32)> = HashMap::new();
    for t in &mesh.tets {
        for (opp, l) in LOCAL.iter().enumerate() {
            let f = [t[l[0]], t[l[1]], t[l[2]]];
            let mut key = f;
            key.sort_unstable();
            count.entry(key).and_modify(|e| e.0 += 1).or_insert((1, f, t[opp]));
        }
    }
    let mut faces: Vec<[u32; 3]> = count
        .into_values()
        .filter(|e| e.0 == 1)
        .map(|(_, f, opp)| {
            let [a, b, c] = f.map(|i| mesh.nodes[i as usize]);
            let d = mesh.nodes[opp as usize];
            if dot(cross(sub(b, a), sub(c, a)), sub(d, a)) > 0.0 {
                [f[0], f[2], f[1]]
            } else {
                f
            }
        })
        .collect();
    faces.sort_unstable();
    faces
}

/// Fraction of boundary triangles whose normal `(x1-x0) x (x2-x0)` points
/// away from the mesh centroid. Close to 1 for an outward, star-shaped
/// surface; close to 0 means the faces are inward and must be flipped.
pub fn outward_fraction(mesh: &TetMesh) -> f64 {
    let n = mesh.nodes.len() as f64;
    let mut c = [0.0; 3];
    for p in &mesh.nodes {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    let ok = mesh
        .faces
        .iter()
        .filter(|f| {
            let [a, b, cc] = f.map(|i| mesh.nodes[i as usize]);
            dot(cross(sub(b, a), sub(cc, a)), sub(a, c)) > 0.0
        })
        .count();
    ok as f64 / mesh.faces.len().max(1) as f64
}

/// Node -> contributions incidence in ELL (padded rows) layout.
///
/// Row `i` lists the slots `local_vertex * n_tets + tet` touching node `i`
/// (vertex-major, so the four per-vertex force arrays are simply concatenated),
/// padded with `pad` (an index of an all-zero row appended to the
/// contribution array) up to `width` entries. Summing `contrib[ell[i, :]]`
/// assembles the nodal value deterministically, without atomics.
#[derive(Debug, Clone)]
pub struct Ell {
    pub width: usize,
    pub pad: u32,
    /// Row-major `[n_rows, width]`.
    pub idx: Vec<u32>,
}

impl Ell {
    /// Builds the ELL table for `n_rows` rows from `(row, slot)` pairs.
    pub fn from_pairs(n_rows: usize, n_slots: usize, pairs: impl Iterator<Item = (u32, u32)> + Clone) -> Self {
        let mut deg = vec![0usize; n_rows];
        for (r, _) in pairs.clone() {
            deg[r as usize] += 1;
        }
        let width = deg.iter().copied().max().unwrap_or(0).max(1);
        let pad = n_slots as u32;
        let mut idx = vec![pad; n_rows * width];
        let mut fill = vec![0usize; n_rows];
        for (r, s) in pairs {
            let r = r as usize;
            idx[r * width + fill[r]] = s;
            fill[r] += 1;
        }
        Self { width, pad, idx }
    }

    /// Incidence of the tetrahedra's vertices: slot `k * n_tets + t`.
    pub fn tet_incidence(mesh: &TetMesh) -> Self {
        let nt = mesh.n_tets();
        let pairs = mesh
            .tets
            .iter()
            .enumerate()
            .flat_map(move |(t, n)| (0..4).map(move |k| (n[k], (k * nt + t) as u32)));
        Self::from_pairs(mesh.n_nodes(), 4 * mesh.n_tets(), pairs)
    }
}
