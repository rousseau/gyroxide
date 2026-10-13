//! `gyroxide` command-line driver.

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use clap::{Parser, Subcommand, ValueEnum};
use gyroxide_mesh::{generate, netgen, normalize, vtk, TetMesh};
use gyroxide_model::{GrowthMask, MeshNormalization, Params, Preset};
use gyroxide_sim::{Simulation, Snapshot, TraceSink};
use gyroxide_validate::{compare_dirs, list_steps, NpyTraceWriter, TraceLayout, TraceMeta};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "gyroxide",
    version,
    about = "Cortical folding by tangential growth (Tallinen et al. 2016) on Burn"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Copy, Clone, ValueEnum)]
enum BackendKind {
    /// GPU through wgpu (Metal / Vulkan / DX12), f32, with kernel fusion.
    Wgpu,
    /// CPU reference, f64.
    Ndarray,
}

#[derive(Subcommand)]
enum Cmd {
    /// Runs a simulation.
    Run(RunArgs),
    /// Prints the TOML parameters of a preset (tallinen2016, braingrowth).
    Preset { name: String },
    /// Writes a voxel-ball test mesh in Netgen .mesh format.
    Ball {
        /// Cubes per diameter.
        #[arg(long, default_value_t = 8)]
        n: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Compares two trace directories step by step.
    Compare {
        ours: PathBuf,
        reference: PathBuf,
        /// File pattern of reference positions, e.g. "pos_{step:07}.npy".
        #[arg(long)]
        ref_positions: Option<String>,
        /// File pattern of reference forces.
        #[arg(long)]
        ref_forces: Option<String>,
    },
}

#[derive(clap::Args)]
struct RunArgs {
    /// Parameter preset.
    #[arg(long, default_value = "tallinen2016")]
    preset: String,
    /// TOML parameter file (overrides the preset).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Netgen .mesh file. Without it, a voxel ball is generated (uniform growth, no midplane).
    #[arg(long)]
    mesh: Option<PathBuf>,
    /// Reference trace directory: read the mesh from its `static/` NPY files
    /// (X0, tets, faces; same node order, no normalisation applied).
    #[arg(long, conflicts_with = "mesh")]
    ref_static: Option<PathBuf>,
    /// Cubes per diameter of the generated voxel ball.
    #[arg(long, default_value_t = 8)]
    ball: usize,
    #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
    backend: BackendKind,
    /// Stop after this many steps (default: until t_end).
    #[arg(long)]
    steps: Option<usize>,
    /// Output directory.
    #[arg(long, default_value = "out")]
    out: PathBuf,
    /// Print diagnostics every N steps (the C code uses 500).
    #[arg(long, default_value_t = 500)]
    print_every: usize,
    /// Write NPY traces every N steps (0 = never).
    #[arg(long, default_value_t = 0)]
    trace_every: usize,
    /// Save a restart point every N steps in <out>/checkpoint (0 = never).
    #[arg(long, default_value_t = 0)]
    checkpoint_every: usize,
    /// Resume from a checkpoint directory written by --checkpoint-every.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// First step written to the NPY traces.
    #[arg(long, default_value_t = 0)]
    trace_from: usize,
    /// Write a VTK surface every N steps (0 = never).
    #[arg(long, default_value_t = 0)]
    vtk_every: usize,
    /// Use the restoring sign for the midplane force (the reference codes push
    /// crossing nodes further away; see docs/model.md).
    #[arg(long)]
    midplane_restoring: bool,
    /// Overrides the mesh spacing a (sets dt and the contact skins).
    #[arg(long)]
    mesh_spacing: Option<f64>,
    /// Growth mask gr = 1 everywhere.
    #[arg(long)]
    uniform_growth: bool,
    #[arg(long)]
    no_contact: bool,
    #[arg(long)]
    no_midplane: bool,
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Run(args) => run(args),
        Cmd::Preset { name } => {
            let p = Preset::from_key(&name).with_context(|| format!("unknown preset '{name}'"))?;
            print!("{}", p.params().to_toml()?);
            Ok(())
        }
        Cmd::Ball { n, out } => {
            let m = generate::voxel_ball(n, 1.0);
            netgen::write(&m, &out)?;
            println!(
                "{}: {} nodes, {} tets, {} faces",
                out.display(),
                m.n_nodes(),
                m.n_tets(),
                m.n_faces()
            );
            Ok(())
        }
        Cmd::Compare {
            ours,
            reference,
            ref_positions,
            ref_forces,
        } => {
            let ours_layout = TraceLayout::default();
            let ref_layout = match ref_positions {
                Some(p) => TraceLayout {
                    positions: p,
                    forces: ref_forces,
                },
                None => TraceLayout::default(),
            };
            let steps = list_steps(&ours)?;
            let rep = compare_dirs(&ours, &ours_layout, &reference, &ref_layout, &steps)?;
            if rep.is_empty() {
                bail!("no common step between the two traces");
            }
            for r in &rep {
                println!("{}", serde_json::to_string(r)?);
            }
            Ok(())
        }
    }
}

fn load_params(args: &RunArgs) -> Result<Params> {
    let mut p = match &args.config {
        Some(path) => Params::load(path).with_context(|| format!("reading {}", path.display()))?,
        None => Preset::from_key(&args.preset)
            .with_context(|| format!("unknown preset '{}' (tallinen2016, braingrowth)", args.preset))?
            .params(),
    };
    if let Some(a) = args.mesh_spacing {
        p.dynamics.mesh_spacing = a;
    }
    if args.uniform_growth {
        p.growth_mask = GrowthMask::Uniform;
    }
    if args.no_contact {
        p.contact.enabled = false;
    }
    if args.no_midplane {
        p.contact.midplane = None;
    }
    if args.midplane_restoring {
        p.contact.midplane_restoring = true;
    }
    p.validate()?;
    Ok(p)
}

fn load_mesh(args: &RunArgs, params: &mut Params) -> Result<(TetMesh, String)> {
    if let Some(dir) = &args.ref_static {
        let m = gyroxide_validate::load_static_mesh(dir)?;
        m.validate()?;
        params.normalization = MeshNormalization::None;
        return Ok((m, format!("{} (static)", dir.display())));
    }
    match &args.mesh {
        Some(path) => {
            let mut m = netgen::read(path).with_context(|| format!("reading {}", path.display()))?;
            let mode = match params.normalization {
                MeshNormalization::None => normalize::Normalization::None,
                MeshNormalization::BrainGrowthWhole => normalize::Normalization::BrainGrowthWhole,
                MeshNormalization::BrainGrowthHalf => normalize::Normalization::BrainGrowthHalf,
            };
            normalize::apply(&mut m, mode);
            Ok((m, path.display().to_string()))
        }
        None => {
            let m = generate::voxel_ball(args.ball, 1.0);
            // The brain-specific ellipsoid mask and midplane make no sense on a ball.
            params.growth_mask = GrowthMask::Uniform;
            params.contact.midplane = None;
            if args.mesh_spacing.is_none() {
                // Use the real spacing so that dt and the skins match the mesh.
                params.dynamics.mesh_spacing = m.surface_edge_stats().1;
            }
            Ok((m, format!("voxel_ball(n={})", args.ball)))
        }
    }
}

fn run(args: RunArgs) -> Result<()> {
    let mut params = load_params(&args)?;
    let (mesh, mesh_name) = load_mesh(&args, &mut params)?;
    params.validate()?;
    match args.backend {
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        BackendKind::Wgpu => {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            run_on::<Wgpu>(args, params, mesh, mesh_name, WgpuDevice::default(), "f32")
        }
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        BackendKind::Wgpu => bail!("built without the wgpu feature"),
        BackendKind::Ndarray => {
            use burn::backend::ndarray::{NdArray, NdArrayDevice};
            run_on::<NdArray<f64>>(args, params, mesh, mesh_name, NdArrayDevice::default(), "f64")
        }
    }
}

/// Prints diagnostics, writes VTK surfaces and forwards to the NPY writer.
struct Outputs<'a> {
    args: &'a RunArgs,
    mesh: &'a TetMesh,
    npy: Option<NpyTraceWriter>,
    start: Instant,
}

impl TraceSink for Outputs<'_> {
    fn wants(&self, step: usize) -> bool {
        let every = |n: usize| n > 0 && step.is_multiple_of(n);
        every(self.args.print_every) || every(self.args.vtk_every) || self.npy.as_ref().is_some_and(|w| w.wants(step))
    }
    fn record(&mut self, s: &Snapshot) -> std::io::Result<()> {
        let g = &s.diagnostics;
        let every = |n: usize| n > 0 && g.step.is_multiple_of(n);
        if every(self.args.print_every) {
            println!(
                "{:>9} {:>9.4} {:>11.4e} {:>11.4e} {:>9.4} {:>9.4} {:>8.4} {:>8.4} {:>6}   {:.1}s",
                g.step,
                g.t,
                g.kinetic,
                g.elastic,
                g.area,
                g.volume,
                g.span_x,
                g.span_y,
                g.n_inverted,
                self.start.elapsed().as_secs_f64()
            );
        }
        if every(self.args.vtk_every) {
            vtk::write_surface(
                self.args.out.join(format!("surface_{:07}.vtk", g.step)),
                self.mesh,
                &s.x,
                &[],
            )?;
        }
        if let Some(w) = &mut self.npy {
            if w.wants(g.step) {
                w.record(s)?;
            }
        }
        Ok(())
    }
    fn finish(&mut self) -> std::io::Result<()> {
        self.npy.as_mut().map_or(Ok(()), |w| w.finish())
    }
}

fn save_checkpoint<B: Backend>(sim: &Simulation<B>, dir: &std::path::Path) -> Result<()> {
    use gyroxide_validate::npy;
    let tmp = dir.with_extension("tmp");
    std::fs::create_dir_all(&tmp)?;
    npy::write_rows3(tmp.join("x.npy"), &sim.positions())?;
    npy::write_rows3(tmp.join("v.npy"), &sim.v.to_rows())?;
    let g = sim.growth_state();
    npy::write_f64(tmp.join("growth_prev.npy"), &[g.len()], &g)?;
    std::fs::write(
        tmp.join("state.json"),
        serde_json::json!({ "step": sim.step, "t": sim.t }).to_string(),
    )?;
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::rename(&tmp, dir)?;
    Ok(())
}

fn load_checkpoint<B: Backend>(sim: &mut Simulation<B>, dir: &std::path::Path) -> Result<(usize, f64)> {
    use gyroxide_validate::npy;
    let state: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("state.json"))?)?;
    let step = state["step"].as_u64().context("checkpoint step")? as usize;
    let t = state["t"].as_f64().context("checkpoint t")?;
    let x = npy::read(dir.join("x.npy"))?.rows3().context("x.npy")?;
    let v = npy::read(dir.join("v.npy"))?.rows3().context("v.npy")?;
    let g = npy::read(dir.join("growth_prev.npy"))?.data;
    if x.len() != sim.mesh.n_nodes() || g.len() != sim.mesh.n_tets() {
        bail!("checkpoint does not match the mesh");
    }
    sim.restore(&x, &v, &g, t, step);
    Ok((step, t))
}

fn run_on<B: Backend>(
    args: RunArgs,
    params: Params,
    mesh: TetMesh,
    mesh_name: String,
    device: B::Device,
    precision: &str,
) -> Result<()> {
    std::fs::create_dir_all(&args.out)?;
    std::fs::write(args.out.join("params.toml"), params.to_toml()?)?;
    let (emin, emean, emax) = mesh.surface_edge_stats();
    println!("gyroxide | preset '{}' | mesh {mesh_name}", params.name);
    println!(
        "  {} nodes, {} tets, {} faces | surface edges min {emin:.4} mean {emean:.4} max {emax:.4} | a = {}",
        mesh.n_nodes(),
        mesh.n_tets(),
        mesh.n_faces(),
        params.dynamics.mesh_spacing
    );
    println!(
        "  dt = {:.4e}, {} steps for t in [{}, {}] | backend {} ({precision})",
        params.dt(),
        params.n_steps(),
        params.dynamics.t_start,
        params.dynamics.t_end,
        std::any::type_name::<B>().split('<').next().unwrap_or("?")
    );

    let npy = if args.trace_every > 0 {
        let meta = TraceMeta {
            code: format!("gyroxide {}", env!("CARGO_PKG_VERSION")),
            preset: params.name.clone(),
            mesh: mesh_name.clone(),
            n_nodes: mesh.n_nodes(),
            n_tets: mesh.n_tets(),
            dt: params.dt(),
            precision: precision.into(),
            notes: String::new(),
        };
        let mut w = NpyTraceWriter::create(args.out.join("trace"), args.trace_every, &meta)?;
        w.from = args.trace_from;
        Some(w)
    } else {
        None
    };

    let t0 = Instant::now();
    let mut sim = Simulation::<B>::new(mesh.clone(), params, device)?;
    println!("  setup {:.2}s", t0.elapsed().as_secs_f64());
    println!(
        "{:>9} {:>9} {:>11} {:>11} {:>9} {:>9} {:>8} {:>8} {:>6}",
        "step", "t", "Uk", "Ue", "area", "volume", "dx", "dy", "ninv"
    );

    if let Some(dir) = &args.resume {
        let (step, t) = load_checkpoint(&mut sim, dir)?;
        println!("  resumed from {} at step {step}, t = {t:.4}", dir.display());
    }

    let start = Instant::now();
    let mut out = Outputs {
        args: &args,
        mesh: &mesh,
        npy,
        start,
    };
    let stop = args.steps.map(|n| sim.step + n);
    while !sim.done() && stop.is_none_or(|s| sim.step < s) {
        let mut chunk = stop.map_or(usize::MAX, |s| s - sim.step);
        if args.checkpoint_every > 0 {
            chunk = chunk.min(args.checkpoint_every - sim.step % args.checkpoint_every);
        }
        sim.run(Some(chunk), &mut out)?;
        if args.checkpoint_every > 0 && sim.step.is_multiple_of(args.checkpoint_every) {
            save_checkpoint(&sim, &args.out.join("checkpoint"))?;
        }
    }
    let x = sim.positions(); // also synchronises the device queue
    let el = start.elapsed().as_secs_f64();
    vtk::write_surface(args.out.join("surface_final.vtk"), &mesh, &x, &[])?;
    println!(
        "done: {} steps, t = {:.4}, {:.2}s ({:.3} ms/step), {} contact rebuilds, {} candidate pairs",
        sim.step,
        sim.t,
        el,
        1e3 * el / sim.step.max(1) as f64,
        sim.n_rebuilds,
        sim.n_contact_pairs()
    );
    if x.iter().flatten().any(|v| !v.is_finite()) {
        bail!("non-finite positions");
    }
    Ok(())
}
