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

use algebra::{floats, scalar, to_f64, M3, T1, V3};
use burn::tensor::backend::Backend;
use contact::ContactPairs;
use gyroxide_mesh::TetMesh;
use gyroxide_model::{NodalTable, Params};
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
    /// Per-tetrahedron a_t and H at each knot of a nodal schedule.
    nodal: Option<NodalTets<B>>,
    /// Growth g_m a_t held fixed (relaxation at constant growth, adjoint
    /// studies): when set, steps use it and do not advance t.
    growth_fixed: Option<T1<B>>,
    /// Optional spring w (target - x) on the first nodes (target positions, per-node weight).
    pull: Option<(V3<B>, T1<B>)>,
    /// Optional correction added to the cortical thickening c per tetrahedron (inverse studies).
    radial_delta: Option<T1<B>>,
    /// Optional isotropic correction delta I added to the core growth tensor U per tetrahedron.
    core_delta: Option<T1<B>>,
    /// Unit normals of the pull targets: if set, springs act along them only (tangential sliding).
    pull_normals: Option<V3<B>>,
    /// Optional core growth tensor per tetrahedron replacing the scheduled one (remodelling).
    core_override: Option<[T1<B>; 6]>,
    /// Dirichlet nodes: (free mask, 1 free / 0 fixed; prescribed positions of the fixed nodes).
    dirichlet: Option<(T1<B>, V3<B>)>,
}

/// Expansion a_t: one value, or one per tetrahedron.
enum Expansion<B: Backend> {
    Uniform(f64),
    PerTet(T1<B>),
}

/// A [`NodalTable`] averaged over the 4 nodes of each tetrahedron, on the device.
struct NodalTets<B: Backend> {
    table_t: NodalTable,
    a: Vec<T1<B>>,
    h: Vec<T1<B>>,
    /// Per knot: the 6 components (xx, yy, zz, xy, xz, yz) of the growth tensor
    /// per tetrahedron; empty without anisotropic growth.
    tensor: Vec<[T1<B>; 6]>,
    /// Per knot: the growth tensor U of the core per tetrahedron; empty if none.
    core: Vec<[T1<B>; 6]>,
}

impl<B: Backend> NodalTets<B> {
    fn at(&self, t: f64) -> (T1<B>, T1<B>) {
        let (i0, i1, w) = self.table_t.bracket(t);
        let lerp = |v: &[T1<B>]| {
            if i0 == i1 {
                v[i0].clone()
            } else {
                v[i0].clone() * (1.0 - w) + v[i1].clone() * w
            }
        };
        (lerp(&self.a), lerp(&self.h))
    }

    /// Thickness H of the cortical layer: H(t), or H at the first knot with thickening.
    fn thickness(&self, t: f64) -> T1<B> {
        if self.table_t.thickening {
            self.h[0].clone()
        } else {
            self.at(t).1
        }
    }

    /// Thickening c = H(t) / H(t_0) - 1 of the cortex, if enabled.
    fn radial_at(&self, t: f64) -> Option<T1<B>> {
        self.table_t
            .thickening
            .then(|| self.at(t).1 / self.h[0].clone() - 1.0)
    }

    fn tensor_at(&self, t: f64) -> Option<M3<B>> {
        self.sym_at(&self.tensor, t)
    }

    fn core_at(&self, t: f64) -> Option<M3<B>> {
        self.sym_at(&self.core, t)
    }

    fn sym_at(&self, rows: &[[T1<B>; 6]], t: f64) -> Option<M3<B>> {
        if rows.is_empty() {
            return None;
        }
        let (i0, i1, w) = self.table_t.bracket(t);
        let c: [T1<B>; 6] = std::array::from_fn(|k| {
            if i0 == i1 {
                rows[i0][k].clone()
            } else {
                rows[i0][k].clone() * (1.0 - w) + rows[i1][k].clone() * w
            }
        });
        let [xx, yy, zz, xy, xz, yz] = c;
        Some(M3 {
            m: [[xx, xy.clone(), xz.clone()], [xy, yy, yz.clone()], [xz, yz, zz]],
        })
    }
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
            nodal: None,
            growth_fixed: None,
            pull: None,
            radial_delta: None,
            core_delta: None,
            pull_normals: None,
            core_override: None,
            dirichlet: None,
        })
    }

    /// Installs the per-node a_t and H of a nodal schedule (required before
    /// the first step when `params.schedule` is nodal). Each tetrahedron takes
    /// the mean of its 4 nodes.
    pub fn set_nodal_table(&mut self, table: NodalTable) {
        let per_tet = |rows: &[Vec<f64>]| -> Vec<T1<B>> {
            rows.iter()
                .map(|r| {
                    let v: Vec<f64> = self
                        .mesh
                        .tets
                        .iter()
                        .map(|t| 0.25 * t.iter().map(|&i| r[i as usize]).sum::<f64>())
                        .collect();
                    floats(&v, &self.device)
                })
                .collect()
        };
        let a = per_tet(&table.a);
        let h = per_tet(&table.h);
        let sym = |rows: &[Vec<f64>]| -> Vec<[T1<B>; 6]> {
            rows.iter()
                .map(|row| {
                    std::array::from_fn(|k| {
                        let v: Vec<f64> = self
                            .mesh
                            .tets
                            .iter()
                            .map(|t| 0.25 * t.iter().map(|&i| row[6 * i as usize + k]).sum::<f64>())
                            .collect();
                        floats(&v, &self.device)
                    })
                })
                .collect()
        };
        let tensor = sym(&table.tensor);
        let core = sym(&table.core);
        self.nodal = Some(NodalTets {
            table_t: table,
            a,
            h,
            tensor,
            core,
        });
    }

    /// Holds the per-tetrahedron growth g_m a_t fixed (`None` resumes the
    /// schedule). While fixed, steps relax towards the equilibrium of that
    /// growth and `t` does not advance.
    pub fn set_fixed_growth(&mut self, growth: Option<&[f64]>) {
        self.growth_fixed = growth.map(|g| floats(g, &self.device));
    }

    /// Per-tetrahedron thickness H at time t (host copy).
    pub fn tet_thickness(&self, t: f64) -> Vec<f64> {
        match self.thickness_at(t) {
            elastic::Thickness::Uniform(h) => vec![h; self.mesh.n_tets()],
            elastic::Thickness::PerTet(h) => to_f64(h),
        }
    }

    /// Cortical indicator g_m per tetrahedron at time t (host copy).
    pub fn cortical_indicator(&self, t: f64) -> Vec<f64> {
        let el = elastic::forces_with(
            &self.pre,
            &self.params.material_at(t),
            &self.x,
            &self.growth_prev,
            &self.thickness_at(t),
            false,
        );
        to_f64(el.gm)
    }

    /// Total nodal force (elastic + contact + midplane) at positions `x`, with
    /// the current growth, contact lists and time; the state is unchanged.
    pub fn total_force(&self, x: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let xv = V3::from_rows(x, &self.device);
        let (force, _) = self.compute_force(&xv, &self.thickness_at(self.t), false);
        force.to_rows()
    }

    /// Rebuilds the contact proximity lists if the surface moved enough.
    pub fn refresh_contact(&mut self) {
        self.maybe_rebuild_contact();
    }

    fn thickness_at(&self, t: f64) -> elastic::Thickness<B> {
        match &self.nodal {
            None => elastic::Thickness::Uniform(self.params.schedule.thickness(t)),
            Some(nd) => elastic::Thickness::PerTet(nd.thickness(t)),
        }
    }

    fn compute_force(
        &self,
        x: &V3<B>,
        thickness: &elastic::Thickness<B>,
        record: bool,
    ) -> (V3<B>, elastic::ElasticOutput<B>) {
        let tensor = self.nodal.as_ref().and_then(|nd| nd.tensor_at(self.t));
        let material = self.params.material_at(self.t);
        let core = self.core_growth();
        let radial = self.radial_growth();
        let el = elastic::forces_full(
            &self.pre,
            &material,
            x,
            &self.growth_prev,
            tensor.as_ref(),
            core.as_ref(),
            radial.as_ref(),
            thickness,
            record,
        );
        let force = self.non_elastic(x).add(&el.force);
        let force = match &self.dirichlet {
            Some((free, _)) => force.mul(free),
            None => force,
        };
        (force, el)
    }

    /// Fixes the nodes where `fixed` is true at `positions` (Dirichlet boundary, e.g. the cut
    /// faces of a sub-mesh): zero force and velocity there, identity rows in `stiffness_product`.
    /// The positions are imposed now and by every `set_state`. `None` frees all nodes.
    pub fn set_dirichlet(&mut self, fixed: Option<(&[bool], &[[f64; 3]])>) {
        self.dirichlet = fixed.map(|(m, pos)| {
            let free: Vec<f64> = m.iter().map(|&f| if f { 0.0 } else { 1.0 }).collect();
            (floats(&free, &self.device), V3::from_rows(pos, &self.device))
        });
        self.impose_dirichlet();
    }

    fn impose_dirichlet(&mut self) {
        if let Some((free, pos)) = &self.dirichlet {
            let fixed = free.clone().neg().add_scalar(1.0);
            self.x = self.x.mul(free).add(&pos.mul(&fixed));
            self.v = self.v.mul(free);
        }
    }

    /// Growth tensor U of the core per tetrahedron at the current time (`core` table).
    pub fn core_growth(&self) -> Option<M3<B>> {
        let mut u = match &self.core_override {
            Some(c) => {
                let [xx, yy, zz, xy, xz, yz] = c.clone();
                Some(M3 { m: [[xx, xy.clone(), xz.clone()], [xy, yy, yz.clone()], [xz, yz, zz]] })
            }
            None => self.nodal.as_ref().and_then(|nd| nd.core_at(self.t)),
        };
        if let (Some(u), Some(d)) = (u.as_mut(), &self.core_delta) {
            for k in 0..3 {
                u.m[k][k] = u.m[k][k].clone() + d.clone();
            }
        }
        u
    }

    /// Core growth tensor U per tetrahedron, row-major [n_tets, 6] (xx, yy, zz, xy, xz, yz),
    /// replacing the scheduled one (e.g. after absorbing the elastic deformation); `None` restores it.
    pub fn set_core_override(&mut self, u: Option<&[f64]>) {
        self.core_override = u.map(|u| {
            std::array::from_fn(|k| floats(&u.iter().skip(k).step_by(6).copied().collect::<Vec<_>>(), &self.device))
        });
    }

    /// Isotropic correction delta I added to the core growth U per tetrahedron; `None` removes it.
    pub fn set_core_delta(&mut self, delta: Option<&[f64]>) {
        self.core_delta = delta.map(|d| floats(d, &self.device));
    }

    /// Thickening c of the cortex per tetrahedron at the current time (`thickening`).
    pub fn radial_growth(&self) -> Option<T1<B>> {
        let base = self.nodal.as_ref().and_then(|nd| nd.radial_at(self.t));
        match (base, &self.radial_delta) {
            (Some(c), Some(d)) => Some(c + d.clone()),
            (None, Some(d)) => Some(d.clone()),
            (c, None) => c,
        }
    }

    /// Correction added to the thickening c per tetrahedron; `None` removes it.
    pub fn set_radial_delta(&mut self, delta: Option<&[f64]>) {
        self.radial_delta = delta.map(|d| floats(d, &self.device));
    }

    /// Springs of stiffness `k` pulling the first `target.len()` nodes (the surface nodes) to
    /// `target`; `None` removes them. Added to every force evaluation and to `stiffness_product`.
    pub fn set_pull(&mut self, target: Option<(&[[f64; 3]], f64)>) {
        self.pull = target.map(|(t, k)| {
            let n = self.pre.n_nodes;
            let mut rows = vec![[0.0; 3]; n];
            rows[..t.len()].copy_from_slice(t);
            let w: Vec<f64> = (0..n).map(|i| if i < t.len() { k } else { 0.0 }).collect();
            (V3::from_rows(&rows, &self.device), floats(&w, &self.device))
        });
    }

    /// Like [`Self::set_pull`] with one stiffness per target node (0: node not pulled).
    pub fn set_pull_weighted(&mut self, target: &[[f64; 3]], k: &[f64]) {
        let n = self.pre.n_nodes;
        let mut rows = vec![[0.0; 3]; n];
        rows[..target.len()].copy_from_slice(target);
        let mut w = vec![0.0; n];
        w[..k.len()].copy_from_slice(k);
        self.pull = Some((V3::from_rows(&rows, &self.device), floats(&w, &self.device)));
    }

    /// Normals of the pull targets (first nodes): springs then act along them only, w n n.(t - x),
    /// so that the nodes slide freely in the tangent plane of their target. `None`: full springs.
    pub fn set_pull_normals(&mut self, normals: Option<&[[f64; 3]]>) {
        self.pull_normals = normals.map(|nr| {
            let mut rows = vec![[0.0; 3]; self.pre.n_nodes];
            rows[..nr.len()].copy_from_slice(nr);
            V3::from_rows(&rows, &self.device)
        });
    }

    /// Contact and mid-plane forces at `x` (current contact lists).
    fn non_elastic(&self, x: &V3<B>) -> V3<B> {
        let p = &self.params;
        let (a, k) = (p.dynamics.mesh_spacing, p.material.bulk);
        let hc = p.repulsive_skin();
        let mut force = V3::zeros(self.pre.n_nodes, &self.device);
        if let Some(pairs) = &self.pairs {
            force = force.add(&contact::penalty_forces(
                pairs,
                x,
                self.pre.n_nodes,
                hc,
                p.contact_stiffness(),
                a,
            ));
        }
        if let Some(mpy) = p.contact.midplane {
            let (lo, hi) = (&self.pre.midplane_low, &self.pre.midplane_high);
            let fy = contact::midplane_force(&x.y, lo, hi, mpy, hc, k, a, p.contact.midplane_restoring);
            force.y = force.y + fy;
        }
        if let Some((t, w)) = &self.pull {
            force = force.add(&match &self.pull_normals {
                None => t.sub(x).mul(w),
                Some(n) => n.mul(&(t.sub(x).dot(n) * w.clone())),
            });
        }
        force
    }

    /// Contact and mid-plane forces at `x` (host copy), without the elastic part.
    pub fn non_elastic_force(&self, x: &[[f64; 3]]) -> Vec<[f64; 3]> {
        self.non_elastic(&V3::from_rows(x, &self.device)).to_rows()
    }

    /// K v = -dF/dx . v at `x` with the current growth, contact lists and time: elastic part
    /// exact (forward mode, regular branch), contact and mid-plane by their linear, symmetric
    /// (Gauss-Newton) tangents.
    pub fn stiffness_product(&self, x: &[[f64; 3]], v: &[[f64; 3]]) -> Vec<[f64; 3]> {
        assert!(
            self.nodal.as_ref().and_then(|nd| nd.tensor_at(self.t)).is_none(),
            "stiffness_product: scalar growth only"
        );
        let xv = V3::from_rows(x, &self.device);
        let v_in = V3::from_rows(v, &self.device);
        let vv = match &self.dirichlet {
            Some((free, _)) => v_in.mul(free),
            None => v_in.clone(),
        };
        let material = self.params.material_at(self.t);
        let thickness = self.thickness_at(self.t);
        let df = elastic::force_tangent(&self.pre, &material, &xv, &vv, &self.growth_prev, &thickness, self.core_growth().as_ref(), self.radial_growth().as_ref());
        let p = &self.params;
        let (a, k) = (p.dynamics.mesh_spacing, p.material.bulk);
        let hc = p.repulsive_skin();
        let mut d = df;
        if let Some(pairs) = &self.pairs {
            d = d.add(&contact::penalty_tangent(pairs, &xv, &vv, self.pre.n_nodes, hc, p.contact_stiffness(), a));
        }
        if let Some(mpy) = p.contact.midplane {
            let (lo, hi) = (&self.pre.midplane_low, &self.pre.midplane_high);
            d.y = d.y + contact::midplane_tangent(&xv.y, &vv.y, lo, hi, mpy, hc, k, a, p.contact.midplane_restoring);
        }
        if let Some((_, w)) = &self.pull {
            d = d.sub(&match &self.pull_normals {
                None => vv.mul(w),
                Some(n) => n.mul(&(vv.dot(n) * w.clone())),
            });
        }
        let kv = d.neg();
        match &self.dirichlet {
            Some((free, _)) => kv.mul(free).add(&v_in.mul(&free.clone().neg().add_scalar(1.0))).to_rows(),
            None => kv.to_rows(),
        }
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
        let (thickness, expansion) = match &self.nodal {
            None => {
                assert!(
                    !p.schedule.is_nodal(),
                    "nodal schedule: call set_nodal_table before stepping"
                );
                (
                    elastic::Thickness::Uniform(p.schedule.thickness(t)),
                    Expansion::Uniform(p.schedule.expansion(t)),
                )
            }
            Some(nd) => (elastic::Thickness::PerTet(nd.thickness(t)), Expansion::PerTet(nd.at(t).0)),
        };

        self.maybe_rebuild_contact();
        if let Some(g) = &self.growth_fixed {
            self.growth_prev = g.clone();
        }
        let (force, el) = self.compute_force(&self.x, &thickness, record);
        let p = &self.params;

        let snapshot = record.then(|| self.snapshot(&force, &el));

        // Growth tensor for the next step (end of the C loop).
        let fixed = self.growth_fixed.is_some();
        if !fixed {
            let tensor_mode = self.nodal.as_ref().is_some_and(|nd| !nd.tensor.is_empty());
            self.growth_prev = match expansion {
                // Anisotropic growth: G = I + g_m T(t), so the state carries g_m.
                _ if tensor_mode => el.gm,
                Expansion::Uniform(a) => el.gm * a,
                Expansion::PerTet(a) => el.gm * a,
            };
        }

        // Damped semi-implicit Euler, lumped mass rho Vn0.
        let damped = force.sub(&self.v.mul(&(el.vn0.clone() * p.dynamics.damping)));
        let inv_m = (el.vn0 * p.dynamics.density).recip() * dt;
        self.v = self.v.add(&damped.mul(&inv_m));
        if let Some((free, _)) = &self.dirichlet {
            self.v = self.v.mul(free);
        }
        self.x = self.x.add(&self.v.mul_scalar(dt));

        if !fixed {
            self.t += dt;
        }
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
        self.impose_dirichlet();
    }

    /// g_m a_t of the previous step per tetrahedron (one readback), for checkpoints.
    pub fn growth_state(&self) -> Vec<f64> {
        to_f64(self.growth_prev.clone())
    }

    /// Restores a checkpoint written from `positions`, `v`, `growth_state`, `t` and `step`.
    pub fn restore(&mut self, x: &[[f64; 3]], v: &[[f64; 3]], growth_prev: &[f64], t: f64, step: usize) {
        self.set_state(x, v, growth_prev);
        self.t = t;
        self.step = step;
    }
}
