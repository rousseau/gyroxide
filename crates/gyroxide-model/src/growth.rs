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
/// variants; the simulator only calls [`Schedule::expansion`],
/// [`Schedule::thickness`] and [`Schedule::shear_moduli`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "law", rename_all = "snake_case")]
pub enum Schedule {
    /// a_t = a0 + a1 t,  H(t) = h0 + h1 t.
    Linear { a0: f64, a1: f64, h0: f64, h1: f64 },
    /// Piecewise-linear interpolation between knots `t` (strictly
    /// increasing), held constant outside `[t[0], t[n-1]]`. Optional
    /// `mu_gray` / `mu_white` knots make the shear moduli time-dependent
    /// (they replace `[material]` values when given).
    Tabulated {
        t: Vec<f64>,
        a: Vec<f64>,
        h: Vec<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mu_gray: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mu_white: Option<Vec<f64>>,
    },
    /// Per-node a_t and H, tabulated at knots `t` and interpolated linearly
    /// in time (held constant outside the knots). `a` and `h` are NPY files
    /// of shape [n_knots, n_nodes] (read by the caller, see [`NodalTable`]);
    /// a tetrahedron uses the mean of its 4 nodes. `a_scale` / `h_scale`
    /// multiply the tables (global knobs for fitting). Optional `mu_gray` /
    /// `mu_white` knots (one value per knot, uniform in space) make the shear
    /// moduli time-dependent, as for the tabulated law.
    Nodal {
        t: Vec<f64>,
        a: String,
        h: String,
        #[serde(default = "one")]
        a_scale: f64,
        #[serde(default = "one")]
        h_scale: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mu_gray: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mu_white: Option<Vec<f64>>,
        /// Optional anisotropic tangential growth: NPY [n_knots, n_nodes, 6] of the
        /// symmetric tensor T (xx, yy, zz, xy, xz, yz); G = I + g_m T replaces
        /// I + g_m a_t (I - n n^T), and `a` then only serves the diagnostics.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tensor: Option<String>,
    },
}

fn one() -> f64 {
    1.0
}

impl Schedule {
    /// Relative tangential expansion a_t (global, same for every
    /// tetrahedron); NaN for the per-node law (see [`NodalTable`]).
    pub fn expansion(&self, t: f64) -> f64 {
        match self {
            Schedule::Linear { a0, a1, .. } => a0 + a1 * t,
            Schedule::Tabulated { t: ts, a, .. } => interpolate(ts, a, t),
            Schedule::Nodal { .. } => f64::NAN,
        }
    }
    /// Thickness H(t) of the growing cortical layer; NaN for the per-node law.
    pub fn thickness(&self, t: f64) -> f64 {
        match self {
            Schedule::Linear { h0, h1, .. } => h0 + h1 * t,
            Schedule::Tabulated { t: ts, h, .. } => interpolate(ts, h, t),
            Schedule::Nodal { .. } => f64::NAN,
        }
    }
    /// True when a_t and H are per node (the simulator needs a [`NodalTable`]).
    pub fn is_nodal(&self) -> bool {
        matches!(self, Schedule::Nodal { .. })
    }
    /// Time-dependent shear moduli (mu_gray, mu_white); `None` keeps the
    /// `[material]` value.
    pub fn shear_moduli(&self, t: f64) -> (Option<f64>, Option<f64>) {
        match self {
            Schedule::Linear { .. } => (None, None),
            Schedule::Tabulated {
                t: ts,
                mu_gray,
                mu_white,
                ..
            }
            | Schedule::Nodal {
                t: ts,
                mu_gray,
                mu_white,
                ..
            } => (
                mu_gray.as_ref().map(|v| interpolate(ts, v, t)),
                mu_white.as_ref().map(|v| interpolate(ts, v, t)),
            ),
        }
    }

    /// Checks the knots of a tabulated law (the linear law is not checked,
    /// to keep the reference presets untouched).
    pub fn validate(&self) -> Result<(), String> {
        if let Schedule::Nodal {
            t,
            a_scale,
            h_scale,
            mu_gray,
            mu_white,
            ..
        } = self
        {
            check_knots(t)?;
            for (name, v) in [("a_scale", a_scale), ("h_scale", h_scale)] {
                if !v.is_finite() || *v <= 0.0 {
                    return Err(format!("nodal schedule: {name} must be > 0 (got {v})"));
                }
            }
            for (name, v) in [("mu_gray", mu_gray), ("mu_white", mu_white)] {
                let Some(v) = v else { continue };
                if v.len() != t.len() {
                    return Err(format!(
                        "nodal schedule: {name} has {} values for {} knots",
                        v.len(),
                        t.len()
                    ));
                }
                if let Some((i, x)) = v.iter().enumerate().find(|(_, x)| !x.is_finite() || **x <= 0.0) {
                    return Err(format!("nodal schedule: {name}[{i}] must be > 0 (got {x})"));
                }
            }
            return Ok(());
        }
        let Schedule::Tabulated {
            t,
            a,
            h,
            mu_gray,
            mu_white,
        } = self
        else {
            return Ok(());
        };
        if t.len() < 2 {
            return Err(format!("tabulated schedule needs at least 2 knots (got {})", t.len()));
        }
        let mut columns = vec![("a", a), ("h", h)];
        columns.extend(mu_gray.iter().map(|v| ("mu_gray", v)));
        columns.extend(mu_white.iter().map(|v| ("mu_white", v)));
        for (name, v) in &columns {
            if v.len() != t.len() {
                return Err(format!(
                    "tabulated schedule: {name} has {} values for {} knots",
                    v.len(),
                    t.len()
                ));
            }
            if let Some(x) = v.iter().find(|x| !x.is_finite()) {
                return Err(format!("tabulated schedule: {name} contains {x}"));
            }
        }
        check_knots(t).map_err(|e| format!("tabulated {e}"))?;
        for (name, v) in columns.iter().skip(1) {
            if let Some((i, x)) = v.iter().enumerate().find(|(_, x)| **x <= 0.0) {
                return Err(format!("tabulated schedule: {name}[{i}] must be > 0 (got {x})"));
            }
        }
        Ok(())
    }
}

/// Knots: finite and strictly increasing.
fn check_knots(t: &[f64]) -> Result<(), String> {
    if let Some(x) = t.iter().find(|x| !x.is_finite()) {
        return Err(format!("schedule: t contains {x}"));
    }
    if let Some(i) = (1..t.len()).find(|&i| t[i] <= t[i - 1]) {
        return Err(format!(
            "schedule: t must be strictly increasing (t[{}] = {} <= t[{}] = {})",
            i,
            t[i],
            i - 1,
            t[i - 1]
        ));
    }
    if t.len() < 2 {
        return Err(format!("schedule needs at least 2 knots (got {})", t.len()));
    }
    Ok(())
}

/// Per-node values of a [`Schedule::Nodal`] law, already scaled: `a[k][i]`
/// and `h[k][i]` at knot `t[k]` and node `i`.
#[derive(Debug, Clone, PartialEq)]
pub struct NodalTable {
    pub t: Vec<f64>,
    pub a: Vec<Vec<f64>>,
    pub h: Vec<Vec<f64>>,
    /// Anisotropic growth tensors per knot (6 values per node), empty if none.
    pub tensor: Vec<Vec<f64>>,
}

impl NodalTable {
    /// Builds the table from row-major [n_knots, n_nodes] arrays and checks it
    /// against the schedule (knots, scales) and the node count.
    pub fn new(schedule: &Schedule, a: &[f64], h: &[f64], n_nodes: usize) -> Result<Self, String> {
        let Schedule::Nodal {
            t, a_scale, h_scale, ..
        } = schedule
        else {
            return Err("NodalTable needs a nodal schedule".into());
        };
        let n = t.len() * n_nodes;
        for (name, v) in [("a", a), ("h", h)] {
            if v.len() != n {
                return Err(format!(
                    "nodal schedule: {name} has {} values, expected {} knots x {n_nodes} nodes",
                    v.len(),
                    t.len()
                ));
            }
            if let Some(x) = v.iter().find(|x| !x.is_finite()) {
                return Err(format!("nodal schedule: {name} contains {x}"));
            }
        }
        if let Some(i) = h.iter().position(|&x| x <= 0.0) {
            return Err(format!(
                "nodal schedule: h must be > 0 (knot {}, node {})",
                i / n_nodes,
                i % n_nodes
            ));
        }
        let rows = |v: &[f64], s: f64| {
            v.chunks_exact(n_nodes)
                .map(|r| r.iter().map(|x| x * s).collect())
                .collect()
        };
        Ok(Self {
            t: t.clone(),
            a: rows(a, *a_scale),
            h: rows(h, *h_scale),
            tensor: Vec::new(),
        })
    }

    /// Adds the anisotropic growth tensors, row-major [n_knots, n_nodes, 6],
    /// scaled by the schedule's `a_scale`.
    pub fn with_tensor(mut self, schedule: &Schedule, data: &[f64]) -> Result<Self, String> {
        let Schedule::Nodal { a_scale, .. } = schedule else {
            return Err("tensor growth needs a nodal schedule".into());
        };
        let n = self.a.first().map_or(0, |r| r.len());
        if data.len() != self.t.len() * n * 6 {
            return Err(format!(
                "nodal schedule: tensor has {} values, expected {} knots x {n} nodes x 6",
                data.len(),
                self.t.len()
            ));
        }
        if let Some(x) = data.iter().find(|x| !x.is_finite()) {
            return Err(format!("nodal schedule: tensor contains {x}"));
        }
        self.tensor = data
            .chunks_exact(n * 6)
            .map(|r| r.iter().map(|x| x * a_scale).collect())
            .collect();
        Ok(self)
    }

    /// Knots bracketing t and the weight of the upper one, clamped outside.
    pub fn bracket(&self, t: f64) -> (usize, usize, f64) {
        bracket(&self.t, t)
    }
}

fn bracket(ts: &[f64], t: f64) -> (usize, usize, f64) {
    let n = ts.len();
    if t <= ts[0] {
        return (0, 0, 0.0);
    }
    if t >= ts[n - 1] {
        return (n - 1, n - 1, 0.0);
    }
    let i = ts.partition_point(|&x| x <= t);
    (i - 1, i, (t - ts[i - 1]) / (ts[i] - ts[i - 1]))
}

/// Piecewise-linear interpolation of (ts, ys) at t, clamped to the end
/// values outside [ts[0], ts[n-1]]. `ts` strictly increasing, n >= 1.
fn interpolate(ts: &[f64], ys: &[f64], t: f64) -> f64 {
    let n = ts.len();
    if t <= ts[0] {
        return ys[0];
    }
    if t >= ts[n - 1] {
        return ys[n - 1];
    }
    // First knot strictly above t; 1 <= i <= n-1 here.
    let i = ts.partition_point(|&x| x <= t);
    let w = (t - ts[i - 1]) / (ts[i] - ts[i - 1]);
    ys[i - 1] + w * (ys[i] - ys[i - 1])
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tab(t: &[f64], a: &[f64], h: &[f64]) -> Schedule {
        Schedule::Tabulated {
            t: t.to_vec(),
            a: a.to_vec(),
            h: h.to_vec(),
            mu_gray: None,
            mu_white: None,
        }
    }

    #[test]
    fn tabulated_interpolates_between_knots() {
        let s = tab(&[0.0, 0.5, 1.0], &[0.0, 1.0, 1.5], &[0.04, 0.06, 0.02]);
        assert_eq!(s.expansion(0.5), 1.0);
        assert!((s.expansion(0.25) - 0.5).abs() < 1e-15);
        assert!((s.expansion(0.75) - 1.25).abs() < 1e-15);
        assert!((s.thickness(0.75) - 0.04).abs() < 1e-15);
    }

    #[test]
    fn tabulated_is_clamped_outside_the_knots() {
        let s = tab(&[0.0, 1.0], &[0.2, 1.0], &[0.05, 0.03]);
        assert_eq!(s.expansion(-0.3), 0.2);
        assert_eq!(s.expansion(0.0), 0.2);
        assert_eq!(s.expansion(1.0), 1.0);
        assert_eq!(s.expansion(7.0), 1.0);
        assert_eq!(s.thickness(-1.0), 0.05);
        assert_eq!(s.thickness(2.0), 0.03);
    }

    #[test]
    fn two_knots_match_the_linear_law() {
        let lin = Schedule::Linear {
            a0: 0.1,
            a1: 1.85,
            h0: 0.084,
            h1: -0.042,
        };
        let s = tab(&[0.0, 1.0], &[0.1, 1.95], &[0.084, 0.042]);
        for k in 0..=1000 {
            let t = k as f64 / 1000.0;
            assert!((s.expansion(t) - lin.expansion(t)).abs() < 1e-14, "a at t = {t}");
            assert!((s.thickness(t) - lin.thickness(t)).abs() < 1e-14, "H at t = {t}");
        }
        assert_eq!(s.shear_moduli(0.5), (None, None));
    }

    #[test]
    fn tabulated_shear_moduli() {
        let s = Schedule::Tabulated {
            t: vec![0.0, 1.0],
            a: vec![0.0, 1.0],
            h: vec![0.04, 0.04],
            mu_gray: Some(vec![1.0, 3.0]),
            mu_white: None,
        };
        assert_eq!(s.shear_moduli(0.5), (Some(2.0), None));
        assert_eq!(s.shear_moduli(4.0), (Some(3.0), None));
    }

    #[test]
    fn tabulated_validation() {
        assert!(tab(&[0.0, 1.0], &[0.0, 1.0], &[0.04, 0.02]).validate().is_ok());
        let err = |s: Schedule| s.validate().unwrap_err();
        assert!(err(tab(&[0.0], &[0.0], &[0.04])).contains("at least 2 knots"));
        assert!(err(tab(&[0.0, 1.0], &[0.0], &[0.04, 0.02])).contains("a has 1 values"));
        assert!(err(tab(&[0.0, 1.0], &[0.0, 1.0], &[0.04])).contains("h has 1 values"));
        assert!(err(tab(&[0.0, 0.5, 0.5], &[0.0; 3], &[0.04; 3])).contains("strictly increasing"));
        assert!(err(tab(&[1.0, 0.0], &[0.0; 2], &[0.04; 2])).contains("strictly increasing"));
        assert!(err(tab(&[0.0, 1.0], &[0.0, 1.0], &[0.04, 0.0])).contains("h[1] must be > 0"));
        assert!(err(tab(&[0.0, f64::NAN], &[0.0; 2], &[0.04; 2])).contains("t contains NaN"));
        let mu = Schedule::Tabulated {
            t: vec![0.0, 1.0],
            a: vec![0.0, 1.0],
            h: vec![0.04, 0.04],
            mu_gray: Some(vec![1.0]),
            mu_white: Some(vec![1.0, -1.0]),
        };
        assert!(err(mu).contains("mu_gray has 1 values"));
    }

    #[test]
    fn nodal_table_interpolates_and_checks() {
        let s = Schedule::Nodal {
            t: vec![0.0, 1.0],
            a: "a.npy".into(),
            h: "h.npy".into(),
            a_scale: 2.0,
            h_scale: 1.0,
            mu_gray: None,
            mu_white: None,
            tensor: None,
        };
        assert!(s.validate().is_ok() && s.is_nodal() && s.expansion(0.3).is_nan());
        // 2 knots x 3 nodes, row-major.
        let tab = NodalTable::new(&s, &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2], &[0.04; 6], 3).unwrap();
        assert_eq!(tab.a[1], vec![2.0, 2.2, 2.4]);
        assert_eq!(tab.bracket(0.25), (0, 1, 0.25));
        assert_eq!(tab.bracket(-1.0), (0, 0, 0.0));
        assert_eq!(tab.bracket(3.0), (1, 1, 0.0));
        assert!(NodalTable::new(&s, &[0.0; 5], &[0.04; 6], 3)
            .unwrap_err()
            .contains("a has 5 values"));
        assert!(NodalTable::new(&s, &[0.0; 6], &[0.04, 0.0, 0.04, 0.04, 0.04, 0.04], 3)
            .unwrap_err()
            .contains("knot 0, node 1"));
        let bad = Schedule::Nodal {
            t: vec![0.0, 0.0],
            a: "a".into(),
            h: "h".into(),
            a_scale: 1.0,
            h_scale: 1.0,
            mu_gray: None,
            mu_white: None,
            tensor: None,
        };
        assert!(bad.validate().unwrap_err().contains("strictly increasing"));
        let mu = Schedule::Nodal {
            t: vec![0.0, 1.0],
            a: "a".into(),
            h: "h".into(),
            a_scale: 1.0,
            h_scale: 1.0,
            mu_gray: None,
            mu_white: Some(vec![0.5, 0.25]),
            tensor: None,
        };
        assert!(mu.validate().is_ok());
        assert_eq!(mu.shear_moduli(0.5), (None, Some(0.375)));
        let bad_mu = Schedule::Nodal {
            t: vec![0.0, 1.0],
            a: "a".into(),
            h: "h".into(),
            a_scale: 1.0,
            h_scale: 1.0,
            mu_gray: None,
            mu_white: Some(vec![0.5, 0.0]),
            tensor: None,
        };
        assert!(bad_mu.validate().unwrap_err().contains("mu_white[1] must be > 0"));
        let src = "law = \"nodal\"\nt = [0.0, 1.0]\na = \"a.npy\"\nh = \"h.npy\"\n";
        let parsed: Schedule = toml::from_str(src).unwrap();
        assert!(matches!(parsed, Schedule::Nodal { a_scale, .. } if a_scale == 1.0));
    }

    #[test]
    fn tabulated_toml_round_trip() {
        let src = "law = \"tabulated\"\nt = [0.0, 0.5, 1.0]\na = [0.0, 0.9, 1.85]\nh = [0.044, 0.04, 0.03]\nmu_gray = [1.0, 1.0, 1.2]\n";
        let s: Schedule = toml::from_str(src).unwrap();
        assert_eq!(s.shear_moduli(1.0), (Some(1.2), None));
        let back: Schedule = toml::from_str(&toml::to_string(&s).unwrap()).unwrap();
        assert_eq!(s, back);
    }
}
