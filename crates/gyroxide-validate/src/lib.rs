//! Validation against reference oracles.
//!
//! **Trace layout** (written by [`NpyTraceWriter`], and the layout the
//! oracle traces are converted to): one directory per run,
//!
//! ```text
//! meta.json                    run description (free-form, see TraceMeta)
//! diagnostics.jsonl            one JSON line per recorded step
//! step_0000500/x.npy           positions        [N, 3] f64
//! step_0000500/v.npy           velocities       [N, 3] f64
//! step_0000500/f.npy           total force      [N, 3] f64 (before damping)
//! ```
//!
//! The state is the one at the *beginning* of the step, as in the C code's
//! output block (forces assembled, Newton update not yet applied).
//! Reference traces from another layout only need a [`TraceLayout`] mapping
//! a step to its file names (`{step}` is substituted, zero-padding allowed
//! as `{step:07}`).

pub mod npy;

use gyroxide_sim::{Snapshot, TraceSink};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TraceMeta {
    pub code: String,
    pub preset: String,
    pub mesh: String,
    pub n_nodes: usize,
    pub n_tets: usize,
    pub dt: f64,
    pub precision: String,
    #[serde(default)]
    pub notes: String,
}

/// Records snapshots every `every` steps (and at `extra` steps) as NPY files.
pub struct NpyTraceWriter {
    pub dir: PathBuf,
    pub every: usize,
    pub extra: Vec<usize>,
    diag: std::io::BufWriter<std::fs::File>,
}

impl NpyTraceWriter {
    pub fn create(dir: impl AsRef<Path>, every: usize, meta: &TraceMeta) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(meta).unwrap())?;
        let diag = std::io::BufWriter::new(std::fs::File::create(dir.join("diagnostics.jsonl"))?);
        Ok(Self {
            dir,
            every: every.max(1),
            extra: vec![],
            diag,
        })
    }
}

pub fn step_dir(step: usize) -> String {
    format!("step_{step:07}")
}

impl TraceSink for NpyTraceWriter {
    fn wants(&self, step: usize) -> bool {
        step % self.every == 0 || self.extra.contains(&step)
    }
    fn record(&mut self, s: &Snapshot) -> std::io::Result<()> {
        let d = self.dir.join(step_dir(s.diagnostics.step));
        std::fs::create_dir_all(&d)?;
        npy::write_rows3(d.join("x.npy"), &s.x)?;
        npy::write_rows3(d.join("v.npy"), &s.v)?;
        npy::write_rows3(d.join("f.npy"), &s.force)?;
        let g = &s.diagnostics;
        let line = serde_json::json!({
            "step": g.step, "t": g.t, "kinetic": g.kinetic, "elastic": g.elastic,
            "area": g.area, "volume": g.volume, "span_x": g.span_x, "span_y": g.span_y,
            "n_inverted": g.n_inverted,
        });
        writeln!(self.diag, "{line}")?;
        self.diag.flush()
    }
}

/// Maps a step to the files of a trace directory.
#[derive(Debug, Clone)]
pub struct TraceLayout {
    pub positions: String,
    pub forces: Option<String>,
}

impl Default for TraceLayout {
    fn default() -> Self {
        Self {
            positions: "step_{step:07}/x.npy".into(),
            forces: Some("step_{step:07}/f.npy".into()),
        }
    }
}

impl TraceLayout {
    pub fn resolve(pattern: &str, step: usize) -> String {
        if let Some(i) = pattern.find("{step:0") {
            let j = i + pattern[i..].find('}').unwrap();
            let width: usize = pattern[i + 7..j].parse().unwrap_or(0);
            format!("{}{:0width$}{}", &pattern[..i], step, &pattern[j + 1..], width = width)
        } else {
            pattern.replace("{step}", &step.to_string())
        }
    }
}

/// Error statistics between two [N, 3] fields.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct FieldError {
    /// max_i |a_i - b_i|
    pub max_abs: f64,
    /// sqrt(mean_i |a_i - b_i|^2)
    pub rms: f64,
    /// max_i |a_i - b_i| / max_i |b_i|
    pub max_rel: f64,
}

pub fn compare(a: &[[f64; 3]], b: &[[f64; 3]]) -> FieldError {
    assert_eq!(a.len(), b.len(), "field sizes differ");
    let mut max_abs = 0.0f64;
    let mut sq = 0.0;
    let mut scale = 0.0f64;
    for (p, q) in a.iter().zip(b) {
        let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
        max_abs = max_abs.max(d);
        sq += d * d;
        scale = scale.max((q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt());
    }
    FieldError {
        max_abs,
        rms: (sq / a.len().max(1) as f64).sqrt(),
        max_rel: max_abs / scale.max(f64::MIN_POSITIVE),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StepReport {
    pub step: usize,
    pub positions: FieldError,
    pub forces: Option<FieldError>,
}

/// Compares the steps present in both trace directories.
pub fn compare_dirs(
    ours: &Path,
    ours_layout: &TraceLayout,
    reference: &Path,
    ref_layout: &TraceLayout,
    steps: &[usize],
) -> std::io::Result<Vec<StepReport>> {
    let load = |dir: &Path, pat: &str, step: usize| -> std::io::Result<Option<Vec<[f64; 3]>>> {
        let p = dir.join(TraceLayout::resolve(pat, step));
        if !p.exists() {
            return Ok(None);
        }
        Ok(npy::read(&p)?.rows3())
    };
    let mut out = Vec::new();
    for &step in steps {
        let (Some(xa), Some(xb)) = (
            load(ours, &ours_layout.positions, step)?,
            load(reference, &ref_layout.positions, step)?,
        ) else {
            continue;
        };
        let forces = match (&ours_layout.forces, &ref_layout.forces) {
            (Some(pa), Some(pb)) => match (load(ours, pa, step)?, load(reference, pb, step)?) {
                (Some(fa), Some(fb)) => Some(compare(&fa, &fb)),
                _ => None,
            },
            _ => None,
        };
        out.push(StepReport {
            step,
            positions: compare(&xa, &xb),
            forces,
        });
    }
    Ok(out)
}

/// Mesh of an oracle trace, from `static/X0.npy`, `static/tets.npy` and
/// `static/faces.npy` (0-based, tetrahedra already right-handed, faces
/// outward), so that node and element orders match the oracle exactly.
pub fn load_static_mesh(dir: &Path) -> std::io::Result<gyroxide_mesh::TetMesh> {
    let st = dir.join("static");
    let x = npy::read(st.join("X0.npy"))?;
    let t = npy::read(st.join("tets.npy"))?;
    let f = npy::read(st.join("faces.npy"))?;
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    let nodes = x.rows3().ok_or_else(|| bad("X0 must be [N,3]"))?;
    if t.shape.len() != 2 || t.shape[1] != 4 || f.shape.len() != 2 || f.shape[1] != 3 {
        return Err(bad("tets must be [T,4] and faces [F,3]"));
    }
    let tets = t.data.chunks_exact(4).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32]).collect();
    let faces = f.data.chunks_exact(3).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32]).collect();
    Ok(gyroxide_mesh::TetMesh { nodes, tets, faces })
}

/// Steps recorded in a trace directory written with the default layout.
pub fn list_steps(dir: &Path) -> std::io::Result<Vec<usize>> {
    let mut steps: Vec<usize> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str()?.strip_prefix("step_")?.parse().ok())
        .collect();
    steps.sort_unstable();
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_patterns() {
        assert_eq!(TraceLayout::resolve("step_{step:07}/x.npy", 500), "step_0000500/x.npy");
        assert_eq!(TraceLayout::resolve("pos_{step}.npy", 42), "pos_42.npy");
    }
}
