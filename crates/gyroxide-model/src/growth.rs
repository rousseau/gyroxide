//! Growth laws. Version 1 implements Tallinen's total tangential growth,
//! recomputed from t at every step (no incremental accumulation):
//!
//!   G = I + (I - n n^T) g_m a_t,   g_m = gr / (1 + exp(10 (d / H(t) - 1)))
//!
//! with n the reference surface normal carried by the tetrahedron, d its
//! mean distance to the reference surface and gr the growth mask.

use serde::{Deserialize, Serialize};

/// Time laws for the relative tangential expansion a_t and the thickness
/// H(t) of the growing layer. New laws (regional Gompertz, MRI-derived
/// alpha_TAN, incremental growth with Mandel relaxation) are added as new
/// variants; the simulator only calls [`Schedule::expansion`] and
/// [`Schedule::thickness`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "law", rename_all = "snake_case")]
pub enum Schedule {
    /// a_t = a0 + a1 t,  H(t) = h0 + h1 t.
    Linear { a0: f64, a1: f64, h0: f64, h1: f64 },
}

impl Schedule {
    /// Relative tangential expansion a_t (global, same for every tetrahedron).
    pub fn expansion(&self, t: f64) -> f64 {
        match *self {
            Schedule::Linear { a0, a1, .. } => a0 + a1 * t,
        }
    }
    /// Thickness H(t) of the growing cortical layer.
    pub fn thickness(&self, t: f64) -> f64 {
        match *self {
            Schedule::Linear { h0, h1, .. } => h0 + h1 * t,
        }
    }
}

/// Smooth cortical indicator g_m for a mean distance `d` to the surface.
pub fn cortical_indicator(d: f64, thickness: f64, gr: f64) -> f64 {
    gr / (1.0 + (10.0 * (d / thickness - 1.0)).exp())
}

/// Per-node growth mask `gr` in [0, 1] (0 = no growth: medial wall, stem).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GrowthMask {
    /// gr = 1 everywhere.
    Uniform,
    /// The hard-coded ellipsoid of Brains.cpp / BrainGrowth `mark_nogrowth`,
    /// in normalised coordinates. Only meaningful for their brain meshes;
    /// kept to reproduce the reference traces.
    TallinenEllipsoid,
    /// One value per node, whitespace-separated text file.
    File { path: String },
}

impl GrowthMask {
    pub fn evaluate(&self, nodes: &[[f64; 3]]) -> std::io::Result<Vec<f64>> {
        match self {
            GrowthMask::Uniform => Ok(vec![1.0; nodes.len()]),
            GrowthMask::TallinenEllipsoid => Ok(nodes.iter().map(|p| tallinen_ellipsoid(*p)).collect()),
            GrowthMask::File { path } => {
                let text = std::fs::read_to_string(path)?;
                let v: Vec<f64> = text
                    .split_ascii_whitespace()
                    .map(|s| s.parse::<f64>())
                    .collect::<Result<_, _>>()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                if v.len() != nodes.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("growth mask has {} values for {} nodes", v.len(), nodes.len()),
                    ));
                }
                Ok(v)
            }
        }
    }
}

fn tallinen_ellipsoid(p: [f64; 3]) -> f64 {
    let q = [(p[0] + 0.1) * 0.714, p[1], p[2] - 0.05];
    let r = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
    if r < 0.6 {
        (1.0 - 10.0 * (0.6 - r)).max(0.0)
    } else {
        1.0
    }
}
