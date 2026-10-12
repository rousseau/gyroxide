//! Mesh normalisation: the model is written in normalised units (brain
//! half-size ~ 1, mesh spacing a ~ 0.01). The transform is kept so that
//! results can be written back in image space (mm).

use crate::TetMesh;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// Mesh is already normalised (Tallinen's week23 meshes).
    None,
    /// BrainGrowth `normalise_coord(..., "whole")`:
    /// x' = -(x - cx)/maxd, y' = (y - cy)/maxd, z' = -(z - cz)/maxd.
    BrainGrowthWhole,
    /// BrainGrowth `normalise_coord(..., "half")`: y measured from min y.
    BrainGrowthHalf,
}

/// The affine map applied by [`apply`]: `x' = sign * (x - origin) / maxd`.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub origin: [f64; 3],
    pub maxd: f64,
    pub sign: [f64; 3],
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            origin: [0.0; 3],
            maxd: 1.0,
            sign: [1.0; 3],
        }
    }
    pub fn forward(&self, p: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|k| self.sign[k] * (p[k] - self.origin[k]) / self.maxd)
    }
    pub fn inverse(&self, q: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|k| self.sign[k] * q[k] * self.maxd + self.origin[k])
    }
}

pub fn apply(mesh: &mut TetMesh, mode: Normalization) -> Transform {
    if mode == Normalization::None || mesh.nodes.is_empty() {
        return Transform::identity();
    }
    let n = mesh.nodes.len() as f64;
    let mut cog = [0.0; 3];
    let mut mn = [f64::INFINITY; 3];
    let mut mx = [f64::NEG_INFINITY; 3];
    for p in &mesh.nodes {
        for k in 0..3 {
            cog[k] += p[k] / n;
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    let half = |k: usize| (mx[k] - cog[k]).abs().max((mn[k] - cog[k]).abs());
    let (origin, maxd) = match mode {
        Normalization::BrainGrowthHalf => ([cog[0], mn[1], cog[2]], half(0).max((mx[1] - mn[1]).abs()).max(half(2))),
        _ => (cog, half(0).max(half(1)).max(half(2))),
    };
    // Two sign flips = a rotation by pi around y: handedness is preserved.
    let tr = Transform {
        origin,
        maxd,
        sign: [-1.0, 1.0, -1.0],
    };
    for p in &mut mesh.nodes {
        *p = tr.forward(*p);
    }
    tr
}
