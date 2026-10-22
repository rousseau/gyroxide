//! `gyroxide` command-line driver.

mod surface;

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use clap::{Parser, Subcommand, ValueEnum};
use gyroxide_io::{SequenceRow, SequenceWriter, State, Status};
use gyroxide_mesh::{generate, netgen, normalize, vtk, vtu, TetMesh};
use gyroxide_model::{GrowthMask, MeshNormalization, Params, Preset};
use gyroxide_sim::{Simulation, Snapshot, TraceSink};
use gyroxide_validate::{compare_dirs, list_steps, NpyTraceWriter, TraceLayout, TraceMeta};
use std::path::PathBuf;
use std::time::Instant;
use surface::SurfaceOut;

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

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Hemi {
    #[value(name = "L", alias = "l")]
    L,
    #[value(name = "R", alias = "r")]
    R,
}

impl Hemi {
    fn letter(self) -> char {
        match self {
            Hemi::L => 'L',
            Hemi::R => 'R',
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Runs a simulation.
    Run(Box<RunArgs>),
    /// Prints the TOML parameters of a preset (tallinen2016, braingrowth).
    Preset { name: String },
    /// Tetrahedral mesh of a session (gyroxide#5), initial state of `run --bids`: from the white and
    /// pial surfaces and the cortex mask written by `surfx run --bids`, with TetGen run as an external
    /// program (never linked). Writes derivatives/gyroxide/sub-X[/ses-Y]/anat/
    /// sub-X[_ses-Y]_hemi-H_desc-<desc>_tet.vtu, in mm in the frame of the image, with the per-node
    /// fields gray_matter (1 grey matter, 0 white matter: inside the white surface) and cortex (cortex
    /// mask of the nearest pial vertex).
    Mesh(MeshArgs),
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
struct MeshArgs {
    /// Root of the BIDS dataset (or its sourcedata/).
    #[arg(long)]
    bids: PathBuf,
    #[arg(long)]
    participant: String,
    #[arg(long)]
    session: Option<String>,
    /// Study root holding derivatives/ (default: as for `run --bids`).
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Hemi::L)]
    hemi: Hemi,
    /// TetGen executable (external program, AGPL: never linked to gyroxide).
    #[arg(long, default_value = "tetgen")]
    tetgen: PathBuf,
    /// TetGen switches (Y keeps the pial triangles exactly).
    #[arg(long, default_value = gyroxide_mesh::tetgen::DEFAULT_SWITCHES)]
    switches: String,
    /// desc- entity of the output (the method).
    #[arg(long, default_value = DEFAULT_MESH_DESC)]
    desc: String,
}

/// desc- entity of the session meshes written by `gyroxide mesh` (and read by default by `run --bids`).
const DEFAULT_MESH_DESC: &str = "tetgen";

#[derive(clap::Args)]
struct RunArgs {
    /// Parameter preset.
    #[arg(long, default_value = "tallinen2016")]
    preset: String,
    /// TOML parameter file (overrides the preset).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Initial state: Netgen .mesh, or .vtu of `gyroxide mesh` (in mm, normalised for the computation).
    /// Without it, a voxel ball is generated (uniform growth, no midplane); in BIDS mode the default is
    /// the session mesh derivatives/gyroxide/…/sub-X[_ses-Y]_hemi-H_desc-<mesh-desc>_tet.vtu.
    #[arg(long)]
    mesh: Option<PathBuf>,
    /// desc- entity of the default session mesh in BIDS mode.
    #[arg(long, default_value = DEFAULT_MESH_DESC)]
    mesh_desc: String,
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
    /// Write the surface (GIfTI, image space in mm: inverse of the mesh normalisation) every N
    /// steps (0 = only the final surface).
    #[arg(long, default_value_t = 0)]
    surface_every: usize,
    /// Hemisphere of the surface (GIfTI AnatomicalStructurePrimary, BIDS hemi- entity).
    #[arg(long, value_enum, default_value_t = Hemi::L)]
    hemi: Hemi,
    /// Debugging: also write a VTK legacy surface every N steps, in the normalised frame of the
    /// computation (0 = never).
    #[arg(long, default_value_t = 0)]
    vtk_every: usize,
    /// Use the restoring sign for the midplane force (the reference codes push
    /// crossing nodes further away; see docs/model.md).
    #[arg(long)]
    midplane_restoring: bool,
    /// Overrides the mesh spacing a (sets dt and the contact skins). For a .vtu session mesh the
    /// default is the shortest surface edge: the TetGen meshes hold flat tetrahedra against the pial
    /// surface and a = mean edge gives NaN within a few hundred steps (about 4x slower, until the
    /// isosurface stuffing of gyroxide#5).
    #[arg(long)]
    mesh_spacing: Option<f64>,
    /// Growth mask gr = 1 everywhere. Needed for a session hemisphere: the default ellipsoid mask of
    /// the presets is made for Tallinen's brain mesh.
    #[arg(long)]
    uniform_growth: bool,
    #[arg(long)]
    no_contact: bool,
    /// No midplane. Needed for a session hemisphere: the midplane of the presets (y' = -0.004 in
    /// tallinen2016) cuts a normalised hemisphere in its middle.
    #[arg(long)]
    no_midplane: bool,
    /// BIDS mode: root of the BIDS dataset (or its sourcedata/). The run is written to
    /// <study>/derivatives/gyroxide/sub-X[/ses-Y]/anat/run-<label>/ (config.toml, status.json,
    /// sequence.tsv, one GIfTI surface per output), instead of --out. For a session hemisphere
    /// (default initial state: the .vtu of `gyroxide mesh`), also give --uniform-growth and
    /// --no-midplane; the mesh spacing defaults to the shortest surface edge (see --mesh-spacing).
    #[arg(long, requires_all = ["participant", "label"])]
    bids: Option<PathBuf>,
    /// BIDS participant (without sub-).
    #[arg(long, requires = "bids")]
    participant: Option<String>,
    /// BIDS session (without ses-).
    #[arg(long, requires = "bids")]
    session: Option<String>,
    /// Study root receiving derivatives/ (default: the parent of the BIDS root if it is named
    /// sourcedata/, otherwise the root itself).
    #[arg(long, requires = "bids")]
    output_dir: Option<PathBuf>,
    /// Run label (letters and digits): run-<label>/.
    #[arg(long, requires = "bids")]
    label: Option<String>,
}

/// Surfaces written every 1000 steps in BIDS mode when --surface-every is not given.
const BIDS_SURFACE_EVERY: usize = 1000;

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Run(args) => run(*args),
        Cmd::Preset { name } => {
            let p = Preset::from_key(&name).with_context(|| format!("unknown preset '{name}'"))?;
            print!("{}", p.params().to_toml()?);
            Ok(())
        }
        Cmd::Mesh(args) => mesh_session(&args),
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

fn mesh_session(args: &MeshArgs) -> Result<()> {
    use surfoxide_io::bids;
    let session = args.session.as_deref();
    let h = args.hemi.letter();
    // surfx outputs: under --output-dir first (a chain run with the same --output-dir), then in the study.
    let mut src = None;
    for out in [args.output_dir.as_deref(), None] {
        let d = bids::derivative_dir(&args.bids, out, bids::PIPELINE_SURFACES, &args.participant, session)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if d.anat.join(format!("{}_hemi-{h}_pial.surf.gii", d.prefix)).is_file() {
            src = Some(d);
            break;
        }
        src.get_or_insert(d);
    }
    let src = src.expect("at least one candidate");
    let input = |name: &str| src.anat.join(format!("{}_hemi-{h}_{name}", src.prefix));
    let read = |name: &str| {
        let p = input(name);
        surfoxide_io::gifti::read_surface(&p).with_context(|| format!("reading {}", p.display()))
    };
    let (pial, _) = read("pial.surf.gii")?;
    let (white, _) = read("white.surf.gii")?;
    let cortex_path = input("cortex.shape.gii");
    let cortex = surfoxide_io::gifti::read_fields(&cortex_path)
        .with_context(|| format!("reading {}", cortex_path.display()))?
        .into_iter()
        .next()
        .with_context(|| format!("{}: no field", cortex_path.display()))?
        .values;
    let nv = pial.vertices().len();
    if cortex.len() != nv {
        bail!("cortex mask: {} values for {nv} pial vertices", cortex.len());
    }
    let work = std::env::temp_dir().join(format!("gyroxide-mesh-{}", std::process::id()));
    let t0 = Instant::now();
    let mesh =
        gyroxide_mesh::tetgen::tetrahedralise(&args.tetgen, pial.vertices(), pial.faces(), &args.switches, &work)?;
    let _ = std::fs::remove_dir_all(&work);
    let white_matter = gyroxide_mesh::inside::inside(&mesh.nodes, white.vertices(), white.faces());
    let near = gyroxide_mesh::distance::nearest(&mesh.nodes, &mesh.nodes[..nv]);
    let gray: Vec<f64> = white_matter.iter().map(|&w| if w { 0.0 } else { 1.0 }).collect();
    let cortex_nodes: Vec<f64> = near.index.iter().map(|&i| cortex[i as usize]).collect();
    let dst = bids::derivative_dir(
        &args.bids,
        args.output_dir.as_deref(),
        gyroxide_io::PIPELINE,
        &args.participant,
        session,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    std::fs::create_dir_all(&dst.anat)?;
    let out = dst
        .anat
        .join(format!("{}_hemi-{h}_desc-{}_tet.vtu", dst.prefix, args.desc));
    vtu::write(
        &out,
        &mesh,
        &[
            vtu::PointField {
                name: "gray_matter".into(),
                values: gray.clone(),
            },
            vtu::PointField {
                name: "cortex".into(),
                values: cortex_nodes,
            },
        ],
    )?;
    gyroxide_io::ecrire_dataset_description(
        &dst.pipeline,
        &gyroxide_io::Generator {
            name: "gyroxide mesh".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            commit: option_env!("GYROXIDE_COMMIT").map(str::to_string),
            command_line: Some(std::env::args().collect::<Vec<_>>().join(" ")),
        },
    )?;
    let vmin = (0..mesh.tets.len())
        .map(|i| mesh.tet_volume(i))
        .fold(f64::INFINITY, f64::min);
    let (dlo, dhi) = mesh.dihedral_range_deg();
    let n_wm = white_matter.iter().filter(|&&w| w).count();
    println!(
        "{}: {} nodes ({nv} pial), {} tets, {} white-matter nodes ({:.1} %) | min volume {vmin:.3e} mm3, dihedral {dlo:.1}-{dhi:.1} deg | {:.1}s",
        out.display(),
        mesh.n_nodes(),
        mesh.tets.len(),
        n_wm,
        100.0 * n_wm as f64 / mesh.n_nodes() as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
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

fn load_mesh(args: &RunArgs, params: &mut Params) -> Result<(TetMesh, String, normalize::Transform)> {
    if let Some(dir) = &args.ref_static {
        let m = gyroxide_validate::load_static_mesh(dir)?;
        m.validate()?;
        params.normalization = MeshNormalization::None;
        return Ok((
            m,
            format!("{} (static)", dir.display()),
            normalize::Transform::identity(),
        ));
    }
    match &args.mesh {
        Some(path) => {
            let is_vtu = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("vtu"));
            let mut m = if is_vtu {
                let (m, _) = vtu::read(path).with_context(|| format!("reading {}", path.display()))?;
                if params.normalization == MeshNormalization::None {
                    // A .vtu session mesh is in mm: it is always normalised for the computation.
                    params.normalization = MeshNormalization::BrainGrowthWhole;
                    println!("  {}: mesh in mm, normalization = brain_growth_whole", path.display());
                }
                m
            } else {
                netgen::read(path).with_context(|| format!("reading {}", path.display()))?
            };
            let mode = match params.normalization {
                MeshNormalization::None => normalize::Normalization::None,
                MeshNormalization::BrainGrowthWhole => normalize::Normalization::BrainGrowthWhole,
                MeshNormalization::BrainGrowthHalf => normalize::Normalization::BrainGrowthHalf,
            };
            // Kept so that the surfaces are written back in image space (identity for "none").
            let transform = normalize::apply(&mut m, mode);
            if is_vtu && args.mesh_spacing.is_none() {
                // Session meshes vary and hold short surface edges: the spacing a (explicit time step,
                // contact skins) is the shortest surface edge, for stability.
                params.dynamics.mesh_spacing = m.surface_edge_stats().0;
                println!(
                    "  mesh spacing a = {:.5} (shortest surface edge, normalised)",
                    params.dynamics.mesh_spacing
                );
            }
            Ok((m, path.display().to_string(), transform))
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
            Ok((
                m,
                format!("voxel_ball(n={})", args.ball),
                normalize::Transform::identity(),
            ))
        }
    }
}

fn run(mut args: RunArgs) -> Result<()> {
    if let (Some(root), None) = (&args.bids, &args.mesh) {
        // BIDS mode: the session mesh written by `gyroxide mesh` is the default initial state.
        let participant = args.participant.as_deref().context("--bids needs --participant")?;
        let d = surfoxide_io::bids::derivative_dir(
            root,
            args.output_dir.as_deref(),
            gyroxide_io::PIPELINE,
            participant,
            args.session.as_deref(),
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let path = d.anat.join(format!(
            "{}_hemi-{}_desc-{}_tet.vtu",
            d.prefix,
            args.hemi.letter(),
            args.mesh_desc
        ));
        if !path.is_file() {
            bail!(
                "session mesh {} not found: run `gyroxide mesh --bids …` first, or give --mesh",
                path.display()
            );
        }
        args.mesh = Some(path);
    }
    let mut params = load_params(&args)?;
    let (mesh, mesh_name, transform) = load_mesh(&args, &mut params)?;
    params.validate()?;
    let bids = match &args.bids {
        Some(_) => {
            let b = BidsRun::create(&args, &params)?;
            args.out = b.dir.clone();
            if args.surface_every == 0 {
                args.surface_every = BIDS_SURFACE_EVERY;
            }
            Some(b)
        }
        None => None,
    };
    let status = bids.as_ref().map(|b| (b.status_path.clone(), b.status.clone()));
    let result = dispatch(args, params, mesh, mesh_name, transform, bids);
    if let (Err(e), Some((path, mut st))) = (&result, status) {
        // status.json: failed, with the message (the last step reached is kept when known).
        if let Ok(last) = gyroxide_io::lire_status(&path) {
            st.step = last.step;
            st.total_steps = last.total_steps;
        }
        st.state = State::Failed;
        st.updated = Some(gyroxide_io::now_iso8601());
        st.message = Some(format!("{e:#}"));
        gyroxide_io::ecrire_status(&path, &st)?;
    }
    result
}

fn dispatch(
    args: RunArgs,
    params: Params,
    mesh: TetMesh,
    mesh_name: String,
    transform: normalize::Transform,
    bids: Option<BidsRun>,
) -> Result<()> {
    match args.backend {
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        BackendKind::Wgpu => {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            run_on::<Wgpu>(
                args,
                params,
                mesh,
                mesh_name,
                transform,
                bids,
                WgpuDevice::default(),
                "f32",
            )
        }
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        BackendKind::Wgpu => bail!("built without the wgpu feature"),
        BackendKind::Ndarray => {
            use burn::backend::ndarray::{NdArray, NdArrayDevice};
            run_on::<NdArray<f64>>(
                args,
                params,
                mesh,
                mesh_name,
                transform,
                bids,
                NdArrayDevice::default(),
                "f64",
            )
        }
    }
}

/// A run written as a BIDS derivative (gyroxide#3, guioxide `docs/donnees.md` section 4).
struct BidsRun {
    dir: PathBuf,
    /// `sub-X[_ses-Y]`.
    prefix: String,
    hemi: char,
    status_path: PathBuf,
    status: Status,
    sequence: SequenceWriter,
    /// Last step whose surface is in sequence.tsv.
    last_written: Option<usize>,
}

impl BidsRun {
    fn create(args: &RunArgs, params: &Params) -> Result<Self> {
        let root = args.bids.as_deref().expect("--bids");
        let participant = args.participant.as_deref().context("--bids needs --participant")?;
        let label = args.label.as_deref().context("--bids needs --label")?;
        surfoxide_io::bids::check_label(label).map_err(|e| anyhow::anyhow!("--label: {e}"))?;
        let mesh = args
            .mesh
            .as_deref()
            .context("--bids needs --mesh (initial state; the default .vtu of surfoxide awaits surfoxide#3)")?;
        let d = surfoxide_io::bids::derivative_dir(
            root,
            args.output_dir.as_deref(),
            gyroxide_io::PIPELINE,
            participant,
            args.session.as_deref(),
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let dir = d.anat.join(format!("run-{label}"));
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        gyroxide_io::ecrire_dataset_description(
            &d.pipeline,
            &gyroxide_io::Generator {
                name: "gyroxide run".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                commit: option_env!("GYROXIDE_COMMIT").map(str::to_string),
                command_line: Some(std::env::args().collect::<Vec<_>>().join(" ")),
            },
        )?;
        // config.toml: the complete parameters (preset + overrides) and the initial state.
        let initial = relative_path(&dir, mesh)?;
        let config = format!(
            "# gyroxide run --label {label}: complete parameters (preset + overrides).\n\
             # initial_state: initial mesh, relative to this run directory.\n\
             initial_state = {}\n\n{}",
            toml_string(&initial),
            params.to_toml()?
        );
        std::fs::write(dir.join("config.toml"), config)?;
        let status_path = dir.join("status.json");
        let status = Status {
            state: State::Running,
            step: Some(0),
            total_steps: None,
            started: Some(gyroxide_io::now_iso8601()),
            updated: Some(gyroxide_io::now_iso8601()),
            message: None,
        };
        gyroxide_io::ecrire_status(&status_path, &status)?;
        let sequence = SequenceWriter::create(&dir.join("sequence.tsv"))?;
        Ok(Self {
            dir,
            prefix: d.prefix,
            hemi: args.hemi.letter(),
            status_path,
            status,
            sequence,
            last_written: None,
        })
    }

    fn update_status(&mut self, state: State, step: usize) -> Result<()> {
        self.status.state = state;
        self.status.step = Some(step as u64);
        self.status.updated = Some(gyroxide_io::now_iso8601());
        Ok(gyroxide_io::ecrire_status(&self.status_path, &self.status)?)
    }

    /// Writes the surface of a snapshot and its line of sequence.tsv.
    fn write_surface(&mut self, surface: &SurfaceOut, params: &Params, s: &Snapshot) -> Result<()> {
        let g = &s.diagnostics;
        let (step, t) = (g.step, g.t);
        let file = gyroxide_io::surface_name(&self.prefix, self.hemi, step as u64);
        let v = surface.write_gifti(&self.dir.join(&file), &s.x)?;
        self.sequence.append(&SequenceRow {
            step: step as u64,
            time: Some(t),
            ga_weeks: params.schedule.ga_weeks(t),
            file,
            gi: gyroxide_io::gi(&v, surface.faces()),
            energy: Some(g.elastic),
            n_contacts: Some(g.n_contacts as u64),
        })?;
        self.last_written = Some(step);
        Ok(())
    }
}

/// `path` relative to the directory `from` (both made absolute).
fn relative_path(from: &std::path::Path, path: &std::path::Path) -> Result<String> {
    let from = std::fs::canonicalize(from)?;
    let to = std::fs::canonicalize(path).with_context(|| format!("reading {}", path.display()))?;
    let (a, b): (Vec<_>, Vec<_>) = (from.components().collect(), to.components().collect());
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut rel = PathBuf::new();
    for _ in common..a.len() {
        rel.push("..");
    }
    for c in &b[common..] {
        rel.push(c);
    }
    Ok(rel.to_string_lossy().into_owned())
}

/// A TOML basic string.
fn toml_string(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Prints diagnostics, writes the surfaces and forwards to the NPY writer.
struct Outputs<'a> {
    args: &'a RunArgs,
    params: &'a Params,
    mesh: &'a TetMesh,
    surface: &'a SurfaceOut,
    bids: Option<BidsRun>,
    npy: Option<NpyTraceWriter>,
    start: Instant,
}

impl TraceSink for Outputs<'_> {
    fn wants(&self, step: usize) -> bool {
        let every = |n: usize| n > 0 && step.is_multiple_of(n);
        every(self.args.print_every)
            || every(self.args.surface_every)
            || every(self.args.vtk_every)
            || self.npy.as_ref().is_some_and(|w| w.wants(step))
    }
    fn record(&mut self, s: &Snapshot) -> std::io::Result<()> {
        let g = &s.diagnostics;
        let every = |n: usize| n > 0 && g.step.is_multiple_of(n);
        if !g.elastic.is_finite() || s.x.iter().flatten().any(|v| !v.is_finite()) {
            if let Some(b) = &mut self.bids {
                // keep the step reached in status.json (marked failed by the caller)
                let _ = b.update_status(State::Running, g.step);
            }
            return Err(std::io::Error::other(format!("non-finite state at step {}", g.step)));
        }
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
        if every(self.args.surface_every) {
            match &mut self.bids {
                Some(b) => b
                    .write_surface(self.surface, self.params, s)
                    .map_err(std::io::Error::other)?,
                None => {
                    self.surface
                        .write_gifti(&self.args.out.join(format!("surface_{:07}.surf.gii", g.step)), &s.x)
                        .map_err(std::io::Error::other)?;
                }
            }
        }
        if let Some(b) = &mut self.bids {
            if every(self.args.print_every) || every(self.args.surface_every) {
                b.update_status(State::Running, g.step).map_err(std::io::Error::other)?;
            }
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
    let state_path = dir.join("state.json");
    let state: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&state_path).with_context(|| format!("reading {}", state_path.display()))?,
    )?;
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

#[allow(clippy::too_many_arguments)]
fn run_on<B: Backend>(
    args: RunArgs,
    params: Params,
    mesh: TetMesh,
    mesh_name: String,
    transform: normalize::Transform,
    bids: Option<BidsRun>,
    device: B::Device,
    precision: &str,
) -> Result<()> {
    std::fs::create_dir_all(&args.out)?;
    if bids.is_none() {
        // (in BIDS mode the parameters are in config.toml)
        std::fs::write(args.out.join("params.toml"), params.to_toml()?)?;
    }
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
    let mut sim = Simulation::<B>::new(mesh.clone(), params.clone(), device)?;
    if let gyroxide_model::Schedule::Nodal { a, h, tensor, .. } = &sim.params.schedule {
        let read = |path: &str| -> Result<Vec<f64>> {
            let arr = gyroxide_validate::npy::read(path).with_context(|| format!("reading {path}"))?;
            Ok(arr.data)
        };
        let (va, vh) = (read(a)?, read(h)?);
        let table = gyroxide_model::NodalTable::new(&sim.params.schedule, &va, &vh, mesh.n_nodes())
            .map_err(|e| anyhow::anyhow!(e))?;
        let table = match tensor {
            Some(path) => {
                println!("  anisotropic growth tensors: {path}");
                table
                    .with_tensor(&sim.params.schedule, &read(path)?)
                    .map_err(|e| anyhow::anyhow!(e))?
            }
            None => table,
        };
        println!(
            "  nodal schedule: {} knots x {} nodes ({a}, {h})",
            table.t.len(),
            mesh.n_nodes()
        );
        sim.set_nodal_table(table);
    }
    println!("  setup {:.2}s", t0.elapsed().as_secs_f64());
    println!(
        "{:>9} {:>9} {:>11} {:>11} {:>9} {:>9} {:>8} {:>8} {:>6}",
        "step", "t", "Uk", "Ue", "area", "volume", "dx", "dy", "ninv"
    );

    if let Some(dir) = &args.resume {
        let (step, t) = load_checkpoint(&mut sim, dir)?;
        println!("  resumed from {} at step {step}, t = {t:.4}", dir.display());
    }

    let surface = SurfaceOut::new(&mesh, transform, args.hemi == Hemi::L);
    let start = Instant::now();
    let step0 = sim.step; // > 0 after --resume
    let stop = args.steps.map(|n| sim.step + n);
    let mut bids = bids;
    if let Some(b) = &mut bids {
        b.status.total_steps = Some(stop.unwrap_or(sim.params.n_steps()).min(sim.params.n_steps()) as u64);
        b.update_status(State::Running, sim.step)?;
    }
    let mut out = Outputs {
        args: &args,
        params: &params,
        mesh: &mesh,
        surface: &surface,
        bids,
        npy,
        start,
    };
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
    if args.checkpoint_every > 0 && !sim.step.is_multiple_of(args.checkpoint_every) {
        // Last state (t_end is generally not a multiple of the checkpoint period).
        save_checkpoint(&sim, &args.out.join("checkpoint"))?;
    }
    let x = sim.positions(); // also synchronises the device queue
    let el = start.elapsed().as_secs_f64();
    if x.iter().flatten().any(|v| !v.is_finite()) {
        bail!("non-finite positions at step {}", sim.step);
    }
    match &mut out.bids {
        Some(b) => {
            if b.last_written != Some(sim.step) {
                // Final state, after the last update: diagnosed without advancing.
                b.write_surface(&surface, &params, &sim.snapshot_now())?;
            }
            b.update_status(State::Done, sim.step)?;
        }
        None => {
            surface.write_gifti(&args.out.join("surface_final.surf.gii"), &x)?;
        }
    }
    if args.vtk_every > 0 {
        vtk::write_surface(args.out.join("surface_final.vtk"), &mesh, &x, &[])?;
    }
    println!(
        "done: {} steps, t = {:.4}, {:.2}s ({:.3} ms/step), {} contact rebuilds, {} candidate pairs",
        sim.step,
        sim.t,
        el,
        1e3 * el / (sim.step - step0).max(1) as f64,
        sim.n_rebuilds,
        sim.n_contact_pairs()
    );
    Ok(())
}
