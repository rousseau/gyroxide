//! Netgen "neutral" `.mesh` format, as read by Tallinen's C code and BrainGrowth:
//!
//! ```text
//! nn
//! x y z            (nn lines)
//! ne
//! r n1 n2 n3 n4    (ne lines, 1-based)
//! nf
//! r n1 n2 n3       (nf lines, 1-based)
//! ```
//!
//! Both reference codes switch the handedness of the tetrahedra on read
//! (`n1 n2 n4 n3`), because Netgen writes left-handed elements. We do the same.

use crate::{MeshError, TetMesh};
use std::io::Write;
use std::path::Path;

pub fn read(path: impl AsRef<Path>) -> Result<TetMesh, MeshError> {
    let text = std::fs::read_to_string(path)?;
    parse(&text)
}

pub fn parse(text: &str) -> Result<TetMesh, MeshError> {
    let mut tok = text.split_ascii_whitespace();
    let mut next = |what: &str| -> Result<&str, MeshError> {
        tok.next()
            .ok_or_else(|| MeshError::Parse(format!("unexpected end of file reading {what}")))
    };
    fn num<T: std::str::FromStr>(s: &str) -> Result<T, MeshError> {
        s.parse().map_err(|_| MeshError::Parse(format!("bad number '{s}'")))
    }

    let nn: usize = num(next("node count")?)?;
    let mut nodes = Vec::with_capacity(nn);
    for _ in 0..nn {
        nodes.push([num(next("x")?)?, num(next("y")?)?, num(next("z")?)?]);
    }
    let ne: usize = num(next("tet count")?)?;
    let mut tets = Vec::with_capacity(ne);
    for _ in 0..ne {
        let _region: i64 = num(next("tet region")?)?;
        let n: [u32; 4] = [
            num::<u32>(next("tet")?)? - 1,
            num::<u32>(next("tet")?)? - 1,
            num::<u32>(next("tet")?)? - 1,
            num::<u32>(next("tet")?)? - 1,
        ];
        tets.push([n[0], n[1], n[3], n[2]]); // handedness switch, as in Brains.cpp
    }
    let nf: usize = num(next("face count")?)?;
    let mut faces = Vec::with_capacity(nf);
    for _ in 0..nf {
        let _region: i64 = num(next("face region")?)?;
        faces.push([
            num::<u32>(next("face")?)? - 1,
            num::<u32>(next("face")?)? - 1,
            num::<u32>(next("face")?)? - 1,
        ]);
    }
    let mesh = TetMesh { nodes, tets, faces };
    mesh.validate()
        .map_err(|e| MeshError::Invalid(format!("{e} (after Netgen handedness switch)")))?;
    Ok(mesh)
}

/// Writes `mesh` in the same format, undoing the handedness switch so that
/// `read(write(m)) == m`.
pub fn write(mesh: &TetMesh, path: impl AsRef<Path>) -> Result<(), MeshError> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "{}", mesh.nodes.len())?;
    for p in &mesh.nodes {
        writeln!(w, "{:.9} {:.9} {:.9}", p[0], p[1], p[2])?;
    }
    writeln!(w, "{}", mesh.tets.len())?;
    for t in &mesh.tets {
        writeln!(w, "1 {} {} {} {}", t[0] + 1, t[1] + 1, t[3] + 1, t[2] + 1)?;
    }
    writeln!(w, "{}", mesh.faces.len())?;
    for f in &mesh.faces {
        writeln!(w, "1 {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1)?;
    }
    Ok(())
}
