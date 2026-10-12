//! gyroxide simulation core: the Tallinen et al. (2016) model of cortical
//! folding, with the whole state resident on the device.
//!
//! This crate is generic over any Burn [`Backend`]. Version 1 is the
//! *reference* implementation written with Burn tensor operations (see
//! docs/architecture.md); fused CubeCL kernels will replace the hot path
//! later and be checked against it.
//!
//! Between two outputs the CPU reads nothing back except one scalar every
//! `contact.check_every` steps (the maximum surface displacement deciding
//! whether the contact lists must be rebuilt).

pub mod algebra;
pub mod contact;
pub mod elastic;
pub mod setup;
pub mod trace;

use algebra::{floats, scalar, to_f64, T1, V3};
use burn::tensor::backend::Backend;
use contact::ContactPairs;
use gyroxide_mesh::TetMesh;
use gyroxide_model::Params;
use setup::{HostFields, Precomputed};
pub use trace::{Snapshot, TraceSink};

/// Global quantities printed by the C code at each output step.
#[derive(Debug, Clone, Copy, Default)]
pub struct Diagnostics {
    pub step: usize,
    pub t: f64,
    /// Kinetic energy 0.5 sum(Vn rho |v|^2).
    pub kinetic: f64,
    /// Elastic energy.
    pub elastic: f64,
    /// Area of the growing surface (weighted by gr).
    pub area: f64,
    /// Volume (sum of deformed nodal volumes).
    pub volume: f64,
    /// Extent of the surface along x and y.
    pub span_x: f64,
    pub span_y: f64,
    /// Number of inverted tetrahedra.
    pub n_inverted: usize,
}

pub struct Simulation<B: Backend> {
    pub params: Params,
    pub mesh: TetMesh,
    pub host: HostFields,
    pub device: B::Device,
    pre: Precomputed<B>,
    /// Positions and velocities [N].
    pub x: V3<B>,
    pub v: V3<B>,
    /// g_m a_t of the previous step, per tetrahedron (defines G).
    growth_prev: T1<B>,
    pairs: Option<ContactPairs<B>>,
    /// Surface positions at the last contact rebuild; None forces a rebuild.
    x_surf_old: Option<V3<B>>,
    pub n_rebuilds: usize,
    pub t: f64,
    pub step: usize,
}

impl<B: Backend> Simulation<B> {
    /// `mesh` must already be normalised (see `gyroxide_mesh::normalize`).
    pub fn new(mesh: TetMesh, params: Params, device: B::Device) -> std::io::Result<Self> {
        let host = setup::host_fields(&mesh, &params)?;
        let pre = setup::upload::<B>(&mesh, &host, &params, &device);
        let x = V3::from_rows(&mesh.nodes, &device);
        let v = V3::zeros(mesh.n_nodes(), &device);
        let growth_prev = T1::<B>::zeros([mesh.n_tets()], &device);
        let t = params.dynamics.t_start;
        Ok(Self {
            params,
            mesh,
            host,
            device,
            pre,
            x,
            v,
            growth_prev,
            pairs: None,
            x_surf_old: None,
            n_rebuilds: 0,
            t,
            step: 0,
        })
    }

    pub fn dt(&self) -> f64 {
        self.params.dt()
    }

    pub fn done(&self) -> bool {
        self.t >= self.params.dynamics.t_end
    }

    /// Advances one time step.
    pub fn step(&mut self) {
        self.advance(false);
    }

    /// Advances one step and returns the state *before* the update, with the
    /// total force acting on it (what the reference traces record).
    pub fn step_recorded(&mut self) -> Snapshot {
        self.advance(true).expect("snapshot requested")
    }

    /// Runs until `t_end` or `max_steps`, feeding `sink` at the steps it asks for.
    pub fn run(&mut self, max_steps: Option<usize>, sink: &mut dyn TraceSink) -> std::io::Result<()> {
        let stop = max_steps.map(|m| self.step + m);
        while !self.done() && stop.is_none_or(|s| self.step < s) {
            if sink.wants(self.step) {
                let snap = self.step_recorded();
                sink.record(&snap)?;
            } else {
                self.step();
            }
        }
        sink.finish()
    }

    fn maybe_rebuild_contact(&mut self) {
        let c = &self.params.contact;
        if !c.enabled {
            return;
        }
        let check = self.x_surf_old.is_none() || self.step.is_multiple_of(c.check_every.max(1));
        if !check {
            return;
        }
        let threshold = 0.5 * (self.params.proximity_skin() - self.params.repulsive_skin());
        let need = match &self.x_surf_old {
            None => true,
            Some(old) => contact::max_surface_displacement(&self.x, &self.pre.surface_nodes, old) > threshold,
        };
        if !need {
            return;
        }
        let x = self.x.to_rows();
        let a = self.params.dynamics.mesh_spacing;
        let pairs = contact::build_pairs(
            &self.mesh,
            &self.host.surface.nodes,
            &x,
            8.0 * a,
            self.params.proximity_skin(),
        );
        self.pairs = contact::upload_pairs(&self.mesh, &self.host.gr, &pairs, &self.device);
        self.x_surf_old = Some(self.x.select(&self.pre.surface_nodes));
        self.n_rebuilds += 1;
    }

    fn advance(&mut self, record: bool) -> Option<Snapshot> {
        let p = &self.params;
        let (t, dt) = (self.t, p.dt());
        let thickness = p.schedule.thickness(t);
        let expansion = p.schedule.expansion(t);

        self.maybe_rebuild_contact();
        let p = &self.params;
        let (a, k) = (p.dynamics.mesh_spacing, p.material.bulk);
        let hc = p.repulsive_skin();

        let mut force = V3::zeros(self.pre.n_nodes, &self.device);
        if let Some(pairs) = &self.pairs {
            force = force.add(&contact::penalty_forces(
                pairs,
                &self.x,
                self.pre.n_nodes,
                hc,
                p.contact_stiffness(),
                a,
            ));
        }
        let el = elastic::forces(&self.pre, &p.material, &self.x, &self.growth_prev, thickness, record);
        force = force.add(&el.force);
        if let Some(mpy) = p.contact.midplane {
            let (lo, hi) = (&self.pre.midplane_low, &self.pre.midplane_high);
            let fy = contact::midplane_force(&self.x.y, lo, hi, mpy, hc, k, a, p.contact.midplane_restoring);
            force.y = force.y + fy;
        }

        let snapshot = record.then(|| self.snapshot(&force, &el));

        // Growth tensor for the next step (end of the C loop).
        self.growth_prev = el.gm * expansion;

        // Damped semi-implicit Euler, lumped mass rho Vn0.
        let damped = force.sub(&self.v.mul(&(el.vn0.clone() * p.dynamics.damping)));
        let inv_m = (el.vn0 * p.dynamics.density).recip() * dt;
        self.v = self.v.add(&damped.mul(&inv_m));
        self.x = self.x.add(&self.v.mul_scalar(dt));

        self.t += dt;
        self.step += 1;
        snapshot
    }

    fn snapshot(&self, force: &V3<B>, el: &elastic::ElasticOutput<B>) -> Snapshot {
        let (ue, ninv) = el
            .energy
            .clone()
            .map(|(u, n)| (scalar(u), scalar(n)))
            .unwrap_or_default();
        let vn = to_f64(el.vn.clone());
        let v = self.v.to_rows();
        let x = self.x.to_rows();
        let rho = self.params.dynamics.density;
        let kinetic = 0.5
            * rho
            * vn.iter()
                .zip(&v)
                .map(|(m, w)| m * (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]))
                .sum::<f64>();
        let gr = &self.host.gr;
        let area = self
            .mesh
            .faces
            .iter()
            .map(|f| {
                let [p0, p1, p2] = f.map(|i| x[i as usize]);
                let n = gyroxide_mesh::cross(gyroxide_mesh::sub(p1, p0), gyroxide_mesh::sub(p2, p0));
                0.5 * gyroxide_mesh::norm(n) * f.iter().map(|&i| gr[i as usize]).sum::<f64>() / 3.0
            })
            .sum();
        let span = |k: usize| {
            let it = self.host.surface.nodes.iter().map(|&i| x[i as usize][k]);
            it.clone().fold(f64::NEG_INFINITY, f64::max) - it.fold(f64::INFINITY, f64::min)
        };
        Snapshot {
            diagnostics: Diagnostics {
                step: self.step,
                t: self.t,
                kinetic,
                elastic: ue,
                area,
                volume: vn.iter().sum(),
                span_x: span(0),
                span_y: span(1),
                n_inverted: ninv as usize,
            },
            x,
            v,
            force: force.to_rows(),
        }
    }

    /// Device constants (for tests and alternative kernels).
    pub fn precomputed(&self) -> &Precomputed<B> {
        &self.pre
    }

    /// Current positions (one GPU -> CPU read).
    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.x.to_rows()
    }

    /// Number of candidate contact pairs currently on the device.
    pub fn n_contact_pairs(&self) -> usize {
        self.pairs.as_ref().map_or(0, |p| p.n_pairs)
    }

    /// Restarts from given positions and velocities (e.g. to replay an oracle
    /// state); the growth state is reset to G from `growth_prev`.
    pub fn set_state(&mut self, x: &[[f64; 3]], v: &[[f64; 3]], growth_prev: &[f64]) {
        self.x = V3::from_rows(x, &self.device);
        self.v = V3::from_rows(v, &self.device);
        self.growth_prev = floats(growth_prev, &self.device);
        self.x_surf_old = None;
    }
}
