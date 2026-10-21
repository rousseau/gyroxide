//! The two parameter sets of version 1, each to be validated against its
//! own reference code.

use crate::growth::{GrowthMask, Schedule};
use crate::{Contact, Dynamics, Material, MeshNormalization, Params};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// Tallinen's C code of the 2016 paper (`Brain_code_and_meshes/Brains.cpp`),
    /// folding-only scenario (commented out in the file, the active one
    /// folds then unfolds): a_t = 1.85 t, H = 0.084 - 0.042 t, rho = 0.0025,
    /// dt = 0.025 sqrt(rho a^2 / K).
    Tallinen2016,
    /// BrainGrowth (Python/Numba rewrite) as described in the design note:
    /// a_t = 1.829 t, H = 0.042 + 0.01 t, rho = 0.01, dt = 0.05 sqrt(rho a^2 / K).
    BrainGrowth,
}

impl Preset {
    pub const ALL: [Preset; 2] = [Preset::Tallinen2016, Preset::BrainGrowth];

    pub fn key(self) -> &'static str {
        match self {
            Preset::Tallinen2016 => "tallinen2016",
            Preset::BrainGrowth => "braingrowth",
        }
    }

    pub fn from_key(s: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|p| p.key() == s.to_ascii_lowercase().replace(['-', '_', ' '], ""))
    }

    pub fn params(self) -> Params {
        let material = Material {
            mu_gray: 1.0,
            mu_white: 1.167,
            bulk: 5.0,
            eps: 0.1,
            k_inverted: 0.0,
            wm_growth: 0.0,
            vol_stab: 0.0,
        };
        let contact = Contact {
            enabled: true,
            stiffness_factor: 10.0,
            repulsive_skin: 0.2,
            proximity_skin: 0.6,
            check_every: 10,
            midplane: Some(-0.004),
            midplane_restoring: false,
        };
        match self {
            Preset::Tallinen2016 => Params {
                name: "Tallinen 2016".into(),
                material,
                dynamics: Dynamics {
                    mesh_spacing: 0.01,
                    density: 0.0025,
                    damping: 0.5,
                    step_factor: 0.025,
                    t_start: 0.0,
                    t_end: 1.0,
                },
                schedule: Schedule::Linear {
                    a0: 0.0,
                    a1: 1.85,
                    h0: 0.084,
                    h1: -0.042,
                },
                contact,
                growth_mask: GrowthMask::TallinenEllipsoid,
                normalization: MeshNormalization::None,
            },
            Preset::BrainGrowth => Params {
                name: "BrainGrowth".into(),
                material,
                dynamics: Dynamics {
                    mesh_spacing: 0.01,
                    density: 0.01,
                    damping: 0.5,
                    step_factor: 0.05,
                    t_start: 0.0,
                    t_end: 1.0,
                },
                schedule: Schedule::Linear {
                    a0: 0.0,
                    a1: 1.829,
                    h0: 0.042,
                    h1: 0.01,
                },
                contact,
                growth_mask: GrowthMask::TallinenEllipsoid,
                normalization: MeshNormalization::BrainGrowthWhole,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_steps_match_design_note() {
        let t = Preset::Tallinen2016.params();
        assert!((t.dt() - 5.590169943749474e-6).abs() < 1e-15);
        assert_eq!(t.n_steps(), 178_886);
        let b = Preset::BrainGrowth.params();
        assert!((b.dt() - 2.23606797749979e-5).abs() < 1e-15);
        assert_eq!(b.n_steps(), 44_722);
    }

    #[test]
    fn toml_round_trip() {
        for p in Preset::ALL {
            let params = p.params();
            let s = params.to_toml().unwrap();
            assert_eq!(crate::Params::from_toml_str(&s).unwrap(), params);
        }
        assert_eq!(Preset::from_key("Tallinen-2016"), Some(Preset::Tallinen2016));
        assert_eq!(Preset::from_key("BrainGrowth"), Some(Preset::BrainGrowth));
    }
}
