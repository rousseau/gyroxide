//! One-off CPU preprocessing, uploaded once to the device.

use crate::algebra::{floats, ints, I1, M3, T1, V3};
use burn::tensor::backend::Backend;
use gyroxide_mesh::distance::nearest;
use gyroxide_mesh::topology::{surface_normals, Ell, Surface};
use gyroxide_mesh::{add, normalize, sub, TetMesh, Vec3};
use gyroxide_model::Params;

/// CPU-side model fields, kept for outputs and for the contact rebuild.
#[derive(Debug, Clone)]
pub struct HostFields {
    pub surface: Surface,
    /// Growth mask per node.
    pub gr: Vec<f64>,
    /// Nearest surface node per node (`csn`) and distance to it (`d2s`).
    pub csn: Vec<u32>,
    pub d2s: Vec<f64>,
    /// Reference vertex normals per surface node (`N0`).
    pub n0: Vec<Vec3>,
}

/// Device-resident constants.
pub struct Precomputed<B: Backend> {
    pub n_nodes: usize,
    pub n_tets: usize,
    pub tet_nodes: [I1<B>; 4],
    /// Reference edge matrix [x1-x0, x2-x0, x3-x0] per tetrahedron.
    pub ar0: M3<B>,
    /// Growth normal per tetrahedron: normalised sum of N0 at the nearest
    /// surface nodes of its 4 vertices (constant: reference normals).
    pub growth_normal: V3<B>,
    /// Mean distance to the surface and mean growth mask of the 4 vertices.
    pub d_mean: T1<B>,
    pub gr_mean: T1<B>,
    /// Node -> tetrahedron-vertex incidence (ELL), padded with `4 * n_tets`.
    pub ell_tet: I1<B>,
    pub ell_tet_width: usize,
    /// Surface node indices (`SN`).
    pub surface_nodes: I1<B>,
    /// Growth mask per node.
    pub gr: T1<B>,
    /// Midplane masks (1.0 for surface nodes starting below / above it).
    pub midplane_low: T1<B>,
    pub midplane_high: T1<B>,
}

pub fn host_fields(mesh: &TetMesh, params: &Params) -> std::io::Result<HostFields> {
    let surface = Surface::new(mesh);
    let gr = params.growth_mask.evaluate(&mesh.nodes)?;
    let surf_pts: Vec<Vec3> = surface.nodes.iter().map(|&i| mesh.nodes[i as usize]).collect();
    let near = nearest(&mesh.nodes, &surf_pts);
    let n0 = surface_normals(mesh, &surface);
    Ok(HostFields {
        surface,
        gr,
        csn: near.index,
        d2s: near.dist,
        n0,
    })
}

pub fn upload<B: Backend>(mesh: &TetMesh, host: &HostFields, params: &Params, device: &B::Device) -> Precomputed<B> {
    let nt = mesh.n_tets();
    let nn = mesh.n_nodes();
    let tet_nodes = std::array::from_fn(|k| ints::<B>(&mesh.tets.iter().map(|t| t[k]).collect::<Vec<_>>(), device));

    let mut cols: [Vec<Vec3>; 3] = Default::default();
    let mut gnorm = Vec::with_capacity(nt);
    let mut d_mean = Vec::with_capacity(nt);
    let mut gr_mean = Vec::with_capacity(nt);
    for t in &mesh.tets {
        let p = t.map(|i| mesh.nodes[i as usize]);
        for k in 0..3 {
            cols[k].push(sub(p[k + 1], p[0]));
        }
        let ns = t
            .iter()
            .fold([0.0; 3], |acc, &i| add(acc, host.n0[host.csn[i as usize] as usize]));
        gnorm.push(normalize(ns));
        d_mean.push(0.25 * t.iter().map(|&i| host.d2s[i as usize]).sum::<f64>());
        gr_mean.push(0.25 * t.iter().map(|&i| host.gr[i as usize]).sum::<f64>());
    }
    let ar0 = M3::from_cols(
        &V3::from_rows(&cols[0], device),
        &V3::from_rows(&cols[1], device),
        &V3::from_rows(&cols[2], device),
    );

    let ell = Ell::tet_incidence(mesh);

    let (mut low, mut high) = (vec![0.0; nn], vec![0.0; nn]);
    if let Some(mpy) = params.contact.midplane {
        let half = 0.5 * params.dynamics.mesh_spacing;
        for &i in &host.surface.nodes {
            let y = mesh.nodes[i as usize][1];
            if y < mpy - half {
                low[i as usize] = 1.0;
            }
            if y > mpy + half {
                high[i as usize] = 1.0;
            }
        }
    }

    Precomputed {
        n_nodes: nn,
        n_tets: nt,
        tet_nodes,
        ar0,
        growth_normal: V3::from_rows(&gnorm, device),
        d_mean: floats(&d_mean, device),
        gr_mean: floats(&gr_mean, device),
        ell_tet: ints(&ell.idx, device),
        ell_tet_width: ell.width,
        surface_nodes: ints(&host.surface.nodes, device),
        gr: floats(&host.gr, device),
        midplane_low: floats(&low, device),
        midplane_high: floats(&high, device),
    }
}
