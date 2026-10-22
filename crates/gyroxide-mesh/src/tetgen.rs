//! TetGen as an **external program** (gyroxide#5): never linked to gyroxide (AGPL licence). The surface
//! is written as a `.smesh`, `tetgen` is run in a working directory, and its `.node` / `.ele` output is
//! read back. The input vertices are always the first nodes, in input order. With `Y` in the switches
//! the surface triangles are kept exactly; without it TetGen may split them (points inserted in the
//! triangles: same geometry), which avoids the flat tetrahedra spanned by four surface vertices.

use crate::topology::extract_boundary;
use crate::{MeshError, TetMesh, Vec3};
use std::io::Write;
use std::path::Path;
use std::process::Command;

/// Default switches: piecewise linear complex (`p`), quality (radius-edge 1.5, min dihedral 10 deg),
/// surface kept (`Y`), quiet (`Q`).
pub const DEFAULT_SWITCHES: &str = "pq1.5/10YQ";

/// Tetrahedralises the closed surface (`vertices`, `faces`) with the `tetgen` executable `exe`, using
/// `workdir` for the exchange files. The surface vertices are the first nodes of the result.
pub fn tetrahedralise(
    exe: &Path,
    vertices: &[Vec3],
    faces: &[[u32; 3]],
    switches: &str,
    workdir: &Path,
) -> Result<TetMesh, MeshError> {
    std::fs::create_dir_all(workdir)?;
    let smesh = workdir.join("surface.smesh");
    {
        let mut w = std::io::BufWriter::new(std::fs::File::create(&smesh)?);
        writeln!(w, "{} 3 0 0", vertices.len())?;
        for (i, p) in vertices.iter().enumerate() {
            writeln!(w, "{i} {} {} {}", p[0], p[1], p[2])?;
        }
        writeln!(w, "{} 0", faces.len())?;
        for f in faces {
            writeln!(w, "3 {} {} {}", f[0], f[1], f[2])?;
        }
        writeln!(w, "0\n0")?;
        w.flush()?;
    }
    let out = Command::new(exe)
        .arg(format!("-{switches}"))
        .arg(&smesh)
        .current_dir(workdir)
        .output()
        .map_err(|e| MeshError::Invalid(format!("running TetGen ({}): {e}", exe.display())))?;
    if !out.status.success() {
        return Err(MeshError::Invalid(format!(
            "TetGen failed ({}): {}{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let nodes = read_table::<3>(&workdir.join("surface.1.node"))?
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect::<Vec<Vec3>>();
    let mut tets: Vec<[u32; 4]> = read_table::<4>(&workdir.join("surface.1.ele"))?
        .into_iter()
        .map(|r| r.map(|v| v as u32))
        .collect();
    if nodes.len() < vertices.len() || nodes[..vertices.len()].iter().zip(vertices).any(|(a, b)| a != b) {
        return Err(MeshError::Invalid(
            "TetGen moved or reordered the surface vertices (use Y)".into(),
        ));
    }
    let mut mesh = TetMesh {
        nodes,
        tets: Vec::new(),
        faces: Vec::new(),
    };
    // Right-handed tetrahedra (gyroxide and VTK convention).
    for t in &mut tets {
        let [a, b, c, d] = t.map(|i| mesh.nodes[i as usize]);
        let det = crate::det3(crate::sub(b, a), crate::sub(c, a), crate::sub(d, a));
        if det < 0.0 {
            t.swap(2, 3);
        }
    }
    mesh.tets = tets;
    mesh.validate()?;
    mesh.faces = extract_boundary(&mesh);
    // With Y the boundary is exactly the input surface; without it, TetGen may split the surface
    // triangles (Steiner points lying in them): same geometry, more vertices.
    let keep = switches.contains('Y');
    if keep && (mesh.faces.len() != faces.len() || mesh.faces.iter().flatten().any(|&i| i as usize >= vertices.len())) {
        return Err(MeshError::Invalid(
            "the boundary of the tetrahedral mesh is not the input surface".into(),
        ));
    }
    if mesh.faces.len() < faces.len() {
        return Err(MeshError::Invalid(
            "the tetrahedral mesh lost part of the input surface".into(),
        ));
    }
    Ok(mesh)
}

/// Rows of a TetGen `.node` / `.ele` file: the `k` values after the index, index base taken from the
/// first row (0 or 1).
fn read_table<const K: usize>(path: &Path) -> Result<Vec<[f64; K]>, MeshError> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty());
    let header: Vec<usize> = lines
        .next()
        .ok_or_else(|| MeshError::Parse(format!("{}: empty", path.display())))?
        .split_whitespace()
        .map(|v| {
            v.parse()
                .map_err(|_| MeshError::Parse(format!("{}: bad header", path.display())))
        })
        .collect::<Result<_, _>>()?;
    let n = header[0];
    let mut rows = Vec::with_capacity(n);
    let mut base = None;
    for l in lines.take(n) {
        let v: Vec<f64> = l
            .split_whitespace()
            .map(|x| {
                x.parse()
                    .map_err(|_| MeshError::Parse(format!("{}: bad value '{x}'", path.display())))
            })
            .collect::<Result<_, _>>()?;
        if v.len() < K + 1 {
            return Err(MeshError::Parse(format!("{}: short row", path.display())));
        }
        let b = *base.get_or_insert(v[0]);
        if v[0] - b != rows.len() as f64 {
            return Err(MeshError::Parse(format!("{}: indices not consecutive", path.display())));
        }
        rows.push(std::array::from_fn(|j| v[j + 1]));
    }
    if rows.len() != n {
        return Err(MeshError::Parse(format!(
            "{}: {} rows for {n}",
            path.display(),
            rows.len()
        )));
    }
    // .ele rows hold node indices in the file's base; bring them to 0-based.
    if path.extension().is_some_and(|e| e == "ele") && base == Some(1.0) {
        for r in &mut rows {
            for x in r.iter_mut() {
                *x -= 1.0;
            }
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::voxel_ball;

    /// Needs the `tetgen` executable (path in GYROXIDE_TETGEN, else `tetgen` in the PATH).
    #[test]
    #[ignore = "tetgen"]
    fn surface_gardee_et_tetraedres_droits() {
        let ball = voxel_ball(4, 10.0);
        let mut on = vec![u32::MAX; ball.n_nodes()];
        let mut verts = Vec::new();
        for &i in ball.faces.iter().flatten() {
            if on[i as usize] == u32::MAX {
                on[i as usize] = verts.len() as u32;
                verts.push(ball.nodes[i as usize]);
            }
        }
        let faces: Vec<[u32; 3]> = ball.faces.iter().map(|f| f.map(|i| on[i as usize])).collect();
        let exe = std::env::var("GYROXIDE_TETGEN").unwrap_or_else(|_| "tetgen".into());
        let dir = std::env::temp_dir().join(format!("gyroxide-tetgen-{}", std::process::id()));
        let m = tetrahedralise(Path::new(&exe), &verts, &faces, DEFAULT_SWITCHES, &dir).unwrap();
        assert_eq!(&m.nodes[..verts.len()], &verts[..]);
        assert!((0..m.tets.len()).all(|i| m.tet_volume(i) > 0.0));
        assert_eq!(m.faces.len(), faces.len());
        let (lo, hi) = m.dihedral_range_deg();
        assert!(lo > 5.0 && hi < 175.0, "dihedral angles {lo:.1}-{hi:.1} deg");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
