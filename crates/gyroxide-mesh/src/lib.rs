//! Tetrahedral meshes for gyroxide.
//!
//! Everything here runs once, on the CPU, before the simulation starts (or
//! when the contact lists are rebuilt): reading and writing meshes,
//! normalisation, surface extraction, node -> tetrahedron incidence in ELL
//! format, and the nearest-surface-node query that sets up cortical growth.

pub mod distance;
pub mod generate;
pub mod inside;
pub mod netgen;
pub mod normalize;
pub mod tetgen;
pub mod topology;
pub mod vtk;
pub mod vtu;

use thiserror::Error;

pub type Vec3 = [f64; 3];

#[derive(Debug, Error)]
pub enum MeshError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("invalid mesh: {0}")]
    Invalid(String),
}

/// A volume mesh of linear tetrahedra plus its boundary triangles.
///
/// Conventions (identical to Tallinen's `Brains.cpp`):
/// - tetrahedra are right-handed: `det[x1-x0, x2-x0, x3-x0] > 0`;
/// - boundary triangles are oriented so that `(x1-x0) x (x2-x0)` points
///   outwards (the contact force relies on it).
#[derive(Debug, Clone, Default)]
pub struct TetMesh {
    pub nodes: Vec<Vec3>,
    pub tets: Vec<[u32; 4]>,
    pub faces: Vec<[u32; 3]>,
}

impl TetMesh {
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn n_tets(&self) -> usize {
        self.tets.len()
    }
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Signed volume of tetrahedron `i`.
    pub fn tet_volume(&self, i: usize) -> f64 {
        let [a, b, c, d] = self.tets[i].map(|n| self.nodes[n as usize]);
        det3(sub(b, a), sub(c, a), sub(d, a)) / 6.0
    }

    /// Minimum, mean and maximum edge length of the boundary triangles
    /// (the C code prints the same statistics; `a` should be close to the mean).
    pub fn surface_edge_stats(&self) -> (f64, f64, f64) {
        let (mut mn, mut sum, mut mx) = (f64::INFINITY, 0.0, 0.0f64);
        for f in &self.faces {
            let p = f.map(|n| self.nodes[n as usize]);
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                let l = norm(sub(p[j], p[i]));
                mn = mn.min(l);
                mx = mx.max(l);
                sum += l;
            }
        }
        (mn, sum / (3.0 * self.faces.len().max(1) as f64), mx)
    }

    /// Makes every tetrahedron right-handed by swapping its last two nodes
    /// when needed. Returns how many were flipped.
    pub fn fix_tet_orientation(&mut self) -> usize {
        let mut flipped = 0;
        for i in 0..self.tets.len() {
            if self.tet_volume(i) < 0.0 {
                self.tets[i].swap(2, 3);
                flipped += 1;
            }
        }
        flipped
    }

    /// Smallest and largest dihedral angles over all tetrahedra, in degrees (quality check).
    pub fn dihedral_range_deg(&self) -> (f64, f64) {
        const EDGES: [(usize, usize, usize, usize); 6] = [
            (0, 1, 2, 3),
            (0, 2, 1, 3),
            (0, 3, 1, 2),
            (1, 2, 0, 3),
            (1, 3, 0, 2),
            (2, 3, 0, 1),
        ];
        let (mut lo, mut hi) = (180.0f64, 0.0f64);
        for t in &self.tets {
            let p = t.map(|n| self.nodes[n as usize]);
            for (a, b, c, d) in EDGES {
                // angle between the faces (a, b, c) and (a, b, d) along the edge ab
                let e = sub(p[b], p[a]);
                let n1 = cross(e, sub(p[c], p[a]));
                let n2 = cross(e, sub(p[d], p[a]));
                let cos = dot(n1, n2) / (norm(n1) * norm(n2));
                let angle = cos.clamp(-1.0, 1.0).acos().to_degrees();
                lo = lo.min(angle);
                hi = hi.max(angle);
            }
        }
        (lo, hi)
    }

    pub fn validate(&self) -> Result<(), MeshError> {
        let n = self.nodes.len() as u32;
        if self
            .tets
            .iter()
            .flatten()
            .chain(self.faces.iter().flatten())
            .any(|&i| i >= n)
        {
            return Err(MeshError::Invalid("node index out of range".into()));
        }
        if let Some(i) = (0..self.tets.len()).find(|&i| self.tet_volume(i) <= 0.0) {
            return Err(MeshError::Invalid(format!(
                "tetrahedron {i} has non-positive volume {:e}",
                self.tet_volume(i)
            )));
        }
        Ok(())
    }
}

#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
pub fn normalize(a: Vec3) -> Vec3 {
    let l = norm(a);
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}
/// Determinant of the matrix whose columns are `a`, `b`, `c`.
#[inline]
pub fn det3(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    dot(a, cross(b, c))
}
