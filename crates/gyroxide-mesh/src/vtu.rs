//! VTK XML `UnstructuredGrid` (`.vtu`) of linear tetrahedra (`VTK_TETRA` = 10), ASCII, with per-node
//! fields: the tetrahedral state of a session (gyroxide#5), in mm in the frame of the image.
//!
//! Only what gyroxide writes is read back: one `Piece`, `Points` (3 components), `Cells`
//! (`connectivity`, `offsets`, `types`, tetrahedra only) and `PointData` arrays of one component, all in
//! `format="ascii"`. The boundary triangles are not stored: they are recomputed on reading.

use crate::topology::extract_boundary;
use crate::{MeshError, TetMesh};
use std::io::Write;
use std::path::Path;

/// VTK cell type of a linear tetrahedron.
pub const VTK_TETRA: u8 = 10;

/// A per-node field of a `.vtu` file.
#[derive(Debug, Clone, PartialEq)]
pub struct PointField {
    pub name: String,
    pub values: Vec<f64>,
}

/// Writes `mesh` (its nodes and tetrahedra) and per-node fields.
pub fn write(path: impl AsRef<Path>, mesh: &TetMesh, fields: &[PointField]) -> Result<(), MeshError> {
    let n = mesh.n_nodes();
    for f in fields {
        if f.values.len() != n {
            return Err(MeshError::Invalid(format!(
                "field {} has {} values for {n} nodes",
                f.name,
                f.values.len()
            )));
        }
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, r#"<?xml version="1.0"?>"#)?;
    writeln!(
        w,
        r#"<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">"#
    )?;
    writeln!(w, "  <UnstructuredGrid>")?;
    writeln!(
        w,
        r#"    <Piece NumberOfPoints="{n}" NumberOfCells="{}">"#,
        mesh.tets.len()
    )?;
    writeln!(w, "      <PointData>")?;
    for f in fields {
        writeln!(
            w,
            r#"        <DataArray type="Float64" Name="{}" format="ascii">"#,
            f.name
        )?;
        for v in &f.values {
            writeln!(w, "{v}")?;
        }
        writeln!(w, "        </DataArray>")?;
    }
    writeln!(w, "      </PointData>")?;
    writeln!(w, "      <Points>")?;
    writeln!(
        w,
        r#"        <DataArray type="Float64" NumberOfComponents="3" format="ascii">"#
    )?;
    for p in &mesh.nodes {
        writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
    }
    writeln!(w, "        </DataArray>")?;
    writeln!(w, "      </Points>")?;
    writeln!(w, "      <Cells>")?;
    writeln!(
        w,
        r#"        <DataArray type="Int64" Name="connectivity" format="ascii">"#
    )?;
    for t in &mesh.tets {
        writeln!(w, "{} {} {} {}", t[0], t[1], t[2], t[3])?;
    }
    writeln!(w, "        </DataArray>")?;
    writeln!(w, r#"        <DataArray type="Int64" Name="offsets" format="ascii">"#)?;
    for k in 1..=mesh.tets.len() {
        writeln!(w, "{}", 4 * k)?;
    }
    writeln!(w, "        </DataArray>")?;
    writeln!(w, r#"        <DataArray type="UInt8" Name="types" format="ascii">"#)?;
    for _ in &mesh.tets {
        writeln!(w, "{VTK_TETRA}")?;
    }
    writeln!(w, "        </DataArray>")?;
    writeln!(w, "      </Cells>")?;
    writeln!(w, "    </Piece>")?;
    writeln!(w, "  </UnstructuredGrid>")?;
    writeln!(w, "</VTKFile>")?;
    w.flush()?;
    Ok(())
}

/// One `<DataArray …>` element: its attributes and its text.
struct Array<'a> {
    tag: &'a str,
    text: &'a str,
    /// Enclosing section: `PointData`, `Points`, `Cells`, `CellData`…
    section: &'a str,
}

fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=\"");
    let i = tag.find(&key)? + key.len();
    Some(&tag[i..i + tag[i..].find('"')?])
}

fn arrays(text: &str) -> Result<Vec<Array<'_>>, MeshError> {
    let mut out = Vec::new();
    let mut rest = text;
    let mut offset = 0;
    while let Some(i) = rest.find("<DataArray") {
        let start = offset + i;
        let tag_end = text[start..]
            .find('>')
            .ok_or_else(|| MeshError::Parse("unterminated DataArray".into()))?;
        let tag = &text[start..start + tag_end + 1];
        let body = start + tag_end + 1;
        let end = text[body..]
            .find("</DataArray>")
            .ok_or_else(|| MeshError::Parse("DataArray without end".into()))?;
        // Enclosing section: the last opening tag among the known ones before this array.
        let before = &text[..start];
        let section = ["PointData", "Points", "Cells", "CellData"]
            .into_iter()
            .filter_map(|s| before.rfind(&format!("<{s}")).map(|k| (k, s)))
            .max()
            .map_or("", |(_, s)| s);
        if attr(tag, "format").is_some_and(|f| f != "ascii") {
            return Err(MeshError::Parse(format!("only ASCII DataArrays are read: {tag}")));
        }
        out.push(Array {
            tag,
            text: &text[body..body + end],
            section,
        });
        offset = body + end + "</DataArray>".len();
        rest = &text[offset..];
    }
    Ok(out)
}

fn numbers<T: std::str::FromStr>(a: &Array) -> Result<Vec<T>, MeshError> {
    a.text
        .split_ascii_whitespace()
        .map(|v| {
            v.parse()
                .map_err(|_| MeshError::Parse(format!("bad number '{v}' in {}", a.tag)))
        })
        .collect()
}

/// Reads a `.vtu` written by [`write`] (or any ASCII one of linear tetrahedra): the tetrahedral mesh,
/// with its boundary triangles recomputed and oriented outwards, and the per-node fields.
pub fn read(path: impl AsRef<Path>) -> Result<(TetMesh, Vec<PointField>), MeshError> {
    let text = std::fs::read_to_string(path)?;
    if !text.contains(r#"type="UnstructuredGrid""#) {
        return Err(MeshError::Parse("not a VTK UnstructuredGrid file".into()));
    }
    let all = arrays(&text)?;
    let find = |section: &str, name: Option<&str>| {
        all.iter()
            .find(|a| a.section == section && name.is_none_or(|n| attr(a.tag, "Name") == Some(n)))
            .ok_or_else(|| MeshError::Parse(format!("missing {section} {}", name.unwrap_or("array"))))
    };
    let p: Vec<f64> = numbers(find("Points", None)?)?;
    if !p.len().is_multiple_of(3) {
        return Err(MeshError::Parse("Points: not 3 components".into()));
    }
    let nodes: Vec<[f64; 3]> = p.as_chunks::<3>().0.to_vec();
    let conn: Vec<u32> = numbers(find("Cells", Some("connectivity"))?)?;
    let offsets: Vec<usize> = numbers(find("Cells", Some("offsets"))?)?;
    let types: Vec<u8> = numbers(find("Cells", Some("types"))?)?;
    if let Some(t) = types.iter().find(|&&t| t != VTK_TETRA) {
        return Err(MeshError::Parse(format!(
            "cell type {t}: only tetrahedra (10) are read"
        )));
    }
    if offsets.len() != types.len()
        || offsets.iter().enumerate().any(|(k, &o)| o != 4 * (k + 1))
        || conn.len() != 4 * types.len()
    {
        return Err(MeshError::Parse(
            "inconsistent cells (connectivity, offsets, types)".into(),
        ));
    }
    let tets: Vec<[u32; 4]> = conn.as_chunks::<4>().0.to_vec();
    let fields = all
        .iter()
        .filter(|a| a.section == "PointData")
        .map(|a| {
            Ok(PointField {
                name: attr(a.tag, "Name").unwrap_or("").to_owned(),
                values: numbers(a)?,
            })
        })
        .collect::<Result<Vec<_>, MeshError>>()?;
    if let Some(f) = fields.iter().find(|f| f.values.len() != nodes.len()) {
        return Err(MeshError::Parse(format!(
            "PointData {}: {} values for {} points",
            f.name,
            f.values.len(),
            nodes.len()
        )));
    }
    let mut mesh = TetMesh {
        nodes,
        tets,
        faces: Vec::new(),
    };
    mesh.validate()?;
    mesh.faces = extract_boundary(&mesh);
    Ok((mesh, fields))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::voxel_ball;

    #[test]
    fn aller_retour() {
        let mesh = voxel_ball(4, 10.0);
        let gm: Vec<f64> = mesh.nodes.iter().map(|p| f64::from(p[0] > 0.0)).collect();
        let path = std::env::temp_dir().join(format!("gyroxide-vtu-{}.vtu", std::process::id()));
        write(
            &path,
            &mesh,
            &[PointField {
                name: "gray_matter".into(),
                values: gm.clone(),
            }],
        )
        .unwrap();
        let (back, fields) = read(&path).unwrap();
        assert_eq!(back.nodes, mesh.nodes);
        assert_eq!(back.tets, mesh.tets);
        let mut f0 = mesh.faces.clone();
        f0.sort_unstable();
        assert_eq!(back.faces, f0, "boundary recomputed, outwards");
        assert_eq!(
            fields,
            [PointField {
                name: "gray_matter".into(),
                values: gm
            }]
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn refuse_ce_qu_il_ne_lit_pas() {
        let path = std::env::temp_dir().join(format!("gyroxide-vtu-bad-{}.vtu", std::process::id()));
        std::fs::write(&path, r#"<VTKFile type="PolyData"></VTKFile>"#).unwrap();
        assert!(read(&path).unwrap_err().to_string().contains("UnstructuredGrid"));
        let mesh = voxel_ball(2, 1.0);
        write(&path, &mesh, &[]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap().replacen(">10\n", ">5\n", 1);
        let text = text.replace("\n10\n", "\n5\n");
        std::fs::write(&path, text).unwrap();
        assert!(read(&path).unwrap_err().to_string().contains("only tetrahedra"));
        std::fs::remove_file(&path).unwrap();
    }
}
