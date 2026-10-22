//! Simulated surface written back in the frame of the source image (gyroxide#2).
//!
//! The simulation works on the normalised mesh (`normalize::apply`, brain half-size ~ 1); the
//! surface is written after `Transform::inverse`, i.e. in RAS mm of the image the initial state came
//! from, as a GIfTI file (surfoxide-io): only the surface nodes, faces oriented outwards,
//! `AnatomicalStructurePrimary` = `CortexLeft` / `CortexRight`, `Pial`.
//!
//! With `normalization = "none"` (meshes already normalised, as Tallinen's meshes or the w21 atlas
//! mesh) there is no transform to invert: the coordinates written are then those of the mesh, in
//! normalised units, **not in mm**.

use anyhow::{Context, Result};
use gyroxide_mesh::normalize::Transform;
use gyroxide_mesh::TetMesh;
use std::path::Path;
use surfoxide_core::TriMesh;
use surfoxide_io::gifti::{write_surface, SurfaceMeta};

/// Surface nodes of the tetrahedral mesh, renumbered, and the way back to image space.
pub struct SurfaceOut {
    /// Mesh node of each surface vertex.
    nodes: Vec<u32>,
    /// Faces on the surface vertices, oriented outwards.
    faces: Vec<[u32; 3]>,
    transform: Transform,
    left: bool,
}

impl SurfaceOut {
    pub fn new(mesh: &TetMesh, transform: Transform, left: bool) -> Self {
        let mut index = vec![u32::MAX; mesh.n_nodes()];
        let mut nodes = Vec::new();
        for f in &mesh.faces {
            for &n in f {
                if index[n as usize] == u32::MAX {
                    index[n as usize] = nodes.len() as u32;
                    nodes.push(n);
                }
            }
        }
        let mut faces: Vec<[u32; 3]> = mesh.faces.iter().map(|f| f.map(|n| index[n as usize])).collect();
        // Outward orientation: positive enclosed volume in the frame where the file is written
        // (the inverse transform may reverse handedness).
        let points: Vec<[f64; 3]> = nodes
            .iter()
            .map(|&n| transform.inverse(mesh.nodes[n as usize]))
            .collect();
        if signed_volume(&points, &faces) < 0.0 {
            for f in &mut faces {
                f.swap(1, 2);
            }
        }
        Self {
            nodes,
            faces,
            transform,
            left,
        }
    }

    /// Surface vertices in image space for the node positions `x` of the simulation.
    pub fn vertices(&self, x: &[[f64; 3]]) -> Vec<[f64; 3]> {
        self.nodes
            .iter()
            .map(|&n| self.transform.inverse(x[n as usize]))
            .collect()
    }

    pub fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }

    /// Writes the surface as GIfTI and returns its vertices (image space).
    pub fn write_gifti(&self, path: &Path, x: &[[f64; 3]]) -> Result<Vec<[f64; 3]>> {
        let v = self.vertices(x);
        let mesh = TriMesh::new(v.clone(), self.faces.clone()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        write_surface(path, &mesh, &SurfaceMeta::hemisphere(self.left, "Pial"))
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(v)
    }
}

fn signed_volume(p: &[[f64; 3]], faces: &[[u32; 3]]) -> f64 {
    faces
        .iter()
        .map(|f| {
            let [a, b, c] = f.map(|i| p[i as usize]);
            a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
        })
        .sum::<f64>()
        / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use gyroxide_mesh::{generate, normalize};

    #[test]
    fn surface_written_back_in_image_space_and_outwards() {
        // A ball in "mm", normalised as for a real mesh, written back: same points as before.
        let mut mesh = generate::voxel_ball(4, 1.0);
        for p in &mut mesh.nodes {
            *p = [10.0 * p[0] - 30.0, 10.0 * p[1] + 5.0, 10.0 * p[2] + 40.0];
        }
        let original = mesh.nodes.clone();
        let tr = normalize::apply(&mut mesh, normalize::Normalization::BrainGrowthWhole);
        let out = SurfaceOut::new(&mesh, tr, true);
        let v = out.vertices(&mesh.nodes);
        for (k, &n) in out.nodes.iter().enumerate() {
            let o = original[n as usize];
            assert!((0..3).all(|d| (v[k][d] - o[d]).abs() < 1e-9), "{:?} vs {:?}", v[k], o);
        }
        assert!(signed_volume(&v, &out.faces) > 0.0, "faces oriented outwards");
        let path = std::env::temp_dir().join(format!("gyroxide-surface-{}.surf.gii", std::process::id()));
        out.write_gifti(&path, &mesh.nodes).unwrap();
        let (read, meta) = surfoxide_io::gifti::read_surface(&path).unwrap();
        assert_eq!(read.vertices().len(), v.len());
        assert_eq!(meta.primary.as_deref(), Some("CortexLeft"));
        let err = read
            .vertices()
            .iter()
            .zip(&v)
            .map(|(a, b)| (0..3).map(|d| (a[d] - b[d]).abs()).fold(0.0, f64::max))
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "float32 round trip: {err}");
        std::fs::remove_file(&path).unwrap();
    }
}
