//! Hook for reference traces.
//!
//! The simulator hands a [`Snapshot`] to a [`TraceSink`] at the steps the
//! sink asks for. `gyroxide-validate` provides an NPY writer (same layout
//! as the instrumented C code is expected to produce) and the comparison
//! against an oracle; any other consumer (live viewer, metrics) can
//! implement the trait too.

use crate::Diagnostics;

/// State at the beginning of a step, with the total force (elastic +
/// contact + midplane, before damping) acting on it.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub diagnostics: Diagnostics,
    pub x: Vec<[f64; 3]>,
    pub v: Vec<[f64; 3]>,
    pub force: Vec<[f64; 3]>,
}

pub trait TraceSink {
    /// Whether a snapshot is wanted at `step` (each one costs a readback).
    fn wants(&self, step: usize) -> bool;
    fn record(&mut self, snapshot: &Snapshot) -> std::io::Result<()>;
    fn finish(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A sink that records nothing.
pub struct NoTrace;

impl TraceSink for NoTrace {
    fn wants(&self, _: usize) -> bool {
        false
    }
    fn record(&mut self, _: &Snapshot) -> std::io::Result<()> {
        Ok(())
    }
}
