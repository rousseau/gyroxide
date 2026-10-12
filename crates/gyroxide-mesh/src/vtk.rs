//! Legacy ASCII VTK writers (readable by ParaView) for periodic outputs.

use crate::{TetMesh, Vec3};
use std::io::Write;
use std::path::Path;

/// Writes the boundary surface with the given node positions and optional
/// per-node scalar fields.
pub fn write_surface(
    path: impl AsRef<Path>,
    mesh: &TetMesh,
    positions: &[Vec3],
    point_data: &[(&str, &[f64])],
) -> std::io::Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        w,
        "# vtk DataFile Version 3.0\ngyroxide surface\nASCII\nDATASET POLYDATA"
    )?;
    writeln!(w, "POINTS {} double", positions.len())?;
    for p in positions {
        writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
    }
    writeln!(w, "POLYGONS {} {}", mesh.faces.len(), 4 * mesh.faces.len())?;
    for f in &mesh.faces {
        writeln!(w, "3 {} {} {}", f[0], f[1], f[2])?;
    }
    write_point_data(&mut w, positions.len(), point_data)
}

/// Writes the full tetrahedral mesh.
pub fn write_volume(
    path: impl AsRef<Path>,
    mesh: &TetMesh,
    positions: &[Vec3],
    point_data: &[(&str, &[f64])],
) -> std::io::Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        w,
        "# vtk DataFile Version 3.0\ngyroxide volume\nASCII\nDATASET UNSTRUCTURED_GRID"
    )?;
    writeln!(w, "POINTS {} double", positions.len())?;
    for p in positions {
        writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
    }
    writeln!(w, "CELLS {} {}", mesh.tets.len(), 5 * mesh.tets.len())?;
    for t in &mesh.tets {
        writeln!(w, "4 {} {} {} {}", t[0], t[1], t[2], t[3])?;
    }
    writeln!(w, "CELL_TYPES {}", mesh.tets.len())?;
    for _ in &mesh.tets {
        writeln!(w, "10")?;
    }
    write_point_data(&mut w, positions.len(), point_data)
}

fn write_point_data(w: &mut impl Write, n: usize, point_data: &[(&str, &[f64])]) -> std::io::Result<()> {
    if point_data.is_empty() {
        return Ok(());
    }
    writeln!(w, "POINT_DATA {n}")?;
    for (name, values) in point_data {
        writeln!(w, "SCALARS {name} double 1\nLOOKUP_TABLE default")?;
        for v in values.iter() {
            writeln!(w, "{v}")?;
        }
    }
    Ok(())
}
