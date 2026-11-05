//! TetGen as an **external program** (gyroxide#5): never linked to gyroxide (AGPL licence). The surface
//! is written as a `.smesh`, `tetgen` is run in a working directory, and its `.node` / `.ele` output is
//! read back. The input vertices are always the first nodes, in input order. With `Y` in the switches
//! the surface triangles are kept exactly; without it TetGen may split them (points inserted in the
//! triangles: same geometry), which avoids the flat tetrahedra spanned by four surface vertices.

use crate::topology::extract_boundary;
use crate::{MeshError, TetMesh, Vec3};
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default switches: piecewise linear complex (`p`), quality (radius-edge 1.5, min dihedral 10 deg),
/// surface kept (`Y`), quiet (`Q`).
pub const DEFAULT_SWITCHES: &str = "pq1.5/10YQ";

/// Name of the conda / mamba environment searched for TetGen (`conda create -n tetgen -c conda-forge tetgen`).
const CONDA_ENV: &str = "tetgen";

/// How to install TetGen, for error messages.
pub const INSTALL_HINT: &str = "install TetGen (Debian/Ubuntu: sudo apt install tetgen; macOS: \
     conda create -n tetgen -c conda-forge tetgen, or the oxide installer), or give its path with \
     --tetgen or GYROXIDE_TETGEN";

/// Finds the TetGen executable, so that it never has to be given by hand: `GYROXIDE_TETGEN`, then
/// `tetgen` on the PATH, then the per-user places where installers put it (`~/.local/bin`, the
/// `tetgen` environment of conda, mamba or micromamba).
pub fn locate() -> Result<PathBuf, MeshError> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let env = |k: &str| std::env::var_os(k);
    candidates(
        env("GYROXIDE_TETGEN"),
        env("PATH"),
        home,
        &[env("CONDA_PREFIX"), env("CONDA_EXE"), env("MAMBA_ROOT_PREFIX")],
    )
    .into_iter()
    .find(|p| is_executable(p))
    .ok_or_else(|| MeshError::Invalid(format!("TetGen not found: {INSTALL_HINT}")))
}

/// Places searched by [`locate`], in order. `conda` holds `CONDA_PREFIX` (active environment),
/// `CONDA_EXE` (`<base>/bin/conda` or `<base>/condabin/conda`) and `MAMBA_ROOT_PREFIX`.
fn candidates(
    explicit: Option<OsString>,
    path: Option<OsString>,
    home: Option<PathBuf>,
    conda: &[Option<OsString>; 3],
) -> Vec<PathBuf> {
    let exe = format!("tetgen{}", std::env::consts::EXE_SUFFIX);
    let mut out: Vec<PathBuf> = explicit.into_iter().map(PathBuf::from).collect();
    if let Some(path) = path {
        out.extend(std::env::split_paths(&path).map(|d| d.join(&exe)));
    }
    let in_env = |base: &Path| base.join("envs").join(CONDA_ENV).join("bin").join(&exe);
    let [prefix, conda_exe, mamba_root] = conda;
    if let Some(prefix) = prefix {
        out.push(Path::new(prefix).join("bin").join(&exe));
    }
    if let Some(base) = conda_exe
        .as_ref()
        .and_then(|c| Path::new(c).parent()?.parent().map(Path::to_path_buf))
    {
        out.push(in_env(&base));
    }
    if let Some(root) = mamba_root {
        out.push(in_env(Path::new(root)));
    }
    if let Some(home) = home {
        out.push(home.join(".local").join("bin").join(&exe));
        for base in [
            "miniforge3",
            "mambaforge",
            "miniconda3",
            "anaconda3",
            "micromamba",
            ".local/share/mamba",
        ] {
            out.push(in_env(&home.join(base)));
        }
    }
    for base in ["/opt/homebrew/Caskroom/miniforge/base", "/opt/miniconda3", "/opt/conda"] {
        out.push(in_env(Path::new(base)));
    }
    out
}

fn is_executable(p: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(p) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.is_file() && meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        meta.is_file()
    }
}

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
        .map_err(|e| MeshError::Invalid(format!("running TetGen ({}): {e}; {INSTALL_HINT}", exe.display())))?;
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

    #[test]
    fn tetgen_searched_explicit_first_then_path_then_user_places() {
        let home = PathBuf::from("/h");
        let path = std::env::join_paths(["/a", "/b"]).unwrap();
        let c = candidates(
            Some("/x/tetgen".into()),
            Some(path),
            Some(home),
            &[
                Some("/env".into()),
                Some("/base/bin/conda".into()),
                Some("/mamba".into()),
            ],
        );
        let at = |s: &str| {
            c.iter()
                .position(|p| p == Path::new(s))
                .unwrap_or_else(|| panic!("{s} absent"))
        };
        assert_eq!(at("/x/tetgen"), 0);
        assert!(at("/a/tetgen") < at("/b/tetgen"));
        assert!(at("/b/tetgen") < at("/env/bin/tetgen"));
        assert!(at("/env/bin/tetgen") < at("/base/envs/tetgen/bin/tetgen"));
        assert!(at("/mamba/envs/tetgen/bin/tetgen") < at("/h/.local/bin/tetgen"));
        assert!(at("/h/.local/bin/tetgen") < at("/h/miniforge3/envs/tetgen/bin/tetgen"));
    }

    #[test]
    fn only_executable_files_are_accepted() {
        let dir = std::env::temp_dir().join(format!("gyroxide-locate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("tetgen");
        std::fs::write(&f, "#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o644)).unwrap();
            assert!(!is_executable(&f));
            std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        assert!(is_executable(&f));
        assert!(!is_executable(&dir));
        assert!(!is_executable(&dir.join("absent")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Needs the `tetgen` executable (found by [`locate`]).
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
        let exe = locate().unwrap();
        let dir = std::env::temp_dir().join(format!("gyroxide-tetgen-{}", std::process::id()));
        let m = tetrahedralise(&exe, &verts, &faces, DEFAULT_SWITCHES, &dir).unwrap();
        assert_eq!(&m.nodes[..verts.len()], &verts[..]);
        assert!((0..m.tets.len()).all(|i| m.tet_volume(i) > 0.0));
        assert_eq!(m.faces.len(), faces.len());
        let (lo, hi) = m.dihedral_range_deg();
        assert!(lo > 5.0 && hi < 175.0, "dihedral angles {lo:.1}-{hi:.1} deg");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
