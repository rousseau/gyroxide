//! Parameters of the Tallinen et al. (2016) cortical folding model.
//!
//! All quantities are in the normalised units of the reference codes:
//! lengths in units of the brain half-size (mesh normalised to ~[-1, 1]),
//! stiffnesses relative to the grey-matter shear modulus, and a
//! dimensionless time t. Physical units (Pa, mm, gestational weeks) are a
//! planned extension (see docs/model.md).

pub mod growth;
pub mod presets;

pub use growth::{GrowthMask, Schedule};
pub use presets::Preset;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("TOML write error: {0}")]
    TomlSer(#[from] toml::ser::Error),
    #[error("invalid parameters: {0}")]
    Invalid(String),
}

/// How the input mesh is brought to normalised coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshNormalization {
    /// The mesh is already normalised (Tallinen's week23 meshes).
    None,
    /// BrainGrowth `normalise_coord(..., "whole")`.
    BrainGrowthWhole,
    /// BrainGrowth `normalise_coord(..., "half")`.
    BrainGrowthHalf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Shear modulus of the cortex (grey matter), `mug`.
    pub mu_gray: f64,
    /// Shear modulus of the white matter, `muw`.
    pub mu_white: f64,
    /// Bulk modulus `K`.
    pub bulk: f64,
    /// Floor stretch for the inverted-element branch, `eps`.
    pub eps: f64,
    /// Extra stiffness on the floored stretch in the inverted branch, `k` (0 in both codes).
    pub k_inverted: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dynamics {
    /// Mean mesh spacing `a`, set by hand in both codes; drives dt and contact.
    pub mesh_spacing: f64,
    /// Mass density `rho`.
    pub density: f64,
    /// Damping coefficient `gamma`.
    pub damping: f64,
    /// Time-step factor `s` in dt = s * sqrt(rho a^2 / K).
    pub step_factor: f64,
    pub t_start: f64,
    pub t_end: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contact {
    pub enabled: bool,
    /// Contact stiffness kc = factor * K.
    pub stiffness_factor: f64,
    /// Repulsive skin hc = factor * a.
    pub repulsive_skin: f64,
    /// Proximity skin hs = factor * a (radius used to build candidate lists).
    pub proximity_skin: f64,
    /// Steps between two checks of the surface displacement that triggers a
    /// rebuild of the proximity lists (the C code checks every step; each
    /// check costs one GPU -> CPU scalar read).
    pub check_every: usize,
    /// Repulsive midplane at y = midplane (whole-brain meshes), or none.
    pub midplane: Option<f64>,
    /// Brains.cpp and BrainGrowth apply `Ft.y -= (mpy - y)/hc a^2 K` to a
    /// node that crossed the plane, which pushes it *further* across (the
    /// force grows with the distance: runaway). `false` reproduces the
    /// reference codes; `true` uses the restoring sign. See docs/model.md.
    #[serde(default)]
    pub midplane_restoring: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Params {
    pub name: String,
    pub material: Material,
    pub dynamics: Dynamics,
    pub schedule: Schedule,
    pub contact: Contact,
    pub growth_mask: GrowthMask,
    pub normalization: MeshNormalization,
}

impl Params {
    /// dt = s * sqrt(rho a^2 / K).
    pub fn dt(&self) -> f64 {
        let d = &self.dynamics;
        d.step_factor * (d.density * d.mesh_spacing * d.mesh_spacing / self.material.bulk).sqrt()
    }
    pub fn n_steps(&self) -> usize {
        // Same loop as the C code: `while (t < t_end) { ...; t += dt; }`.
        ((self.dynamics.t_end - self.dynamics.t_start) / self.dt()).ceil() as usize
    }
    pub fn contact_stiffness(&self) -> f64 {
        self.contact.stiffness_factor * self.material.bulk
    }
    pub fn repulsive_skin(&self) -> f64 {
        self.contact.repulsive_skin * self.dynamics.mesh_spacing
    }
    pub fn proximity_skin(&self) -> f64 {
        self.contact.proximity_skin * self.dynamics.mesh_spacing
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        let pos = [
            ("mu_gray", self.material.mu_gray),
            ("mu_white", self.material.mu_white),
            ("bulk", self.material.bulk),
            ("eps", self.material.eps),
            ("mesh_spacing", self.dynamics.mesh_spacing),
            ("density", self.dynamics.density),
            ("step_factor", self.dynamics.step_factor),
        ];
        if let Some((n, v)) = pos.iter().find(|(_, v)| v.is_nan() || *v <= 0.0) {
            return Err(ModelError::Invalid(format!("{n} must be > 0 (got {v})")));
        }
        if self.dynamics.t_end <= self.dynamics.t_start {
            return Err(ModelError::Invalid("t_end must be > t_start".into()));
        }
        if self.contact.proximity_skin <= self.contact.repulsive_skin {
            return Err(ModelError::Invalid("proximity_skin must exceed repulsive_skin".into()));
        }
        Ok(())
    }

    pub fn from_toml_str(s: &str) -> Result<Self, ModelError> {
        let p: Self = toml::from_str(s)?;
        p.validate()?;
        Ok(p)
    }
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ModelError> {
        Self::from_toml_str(&std::fs::read_to_string(path)?)
    }
    pub fn to_toml(&self) -> Result<String, ModelError> {
        Ok(toml::to_string_pretty(self)?)
    }
}
