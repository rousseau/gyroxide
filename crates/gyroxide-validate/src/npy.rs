//! Minimal NPY (v1/v2) reader and writer for little-endian f64 / f32 /
//! integer arrays in C order, enough for trace exchange with numpy and the
//! instrumented C code.

use std::io::{Read, Write};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Array {
    pub fn rows3(&self) -> Option<Vec<[f64; 3]>> {
        (self.shape.len() == 2 && self.shape[1] == 3).then(|| {
            self.data
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| [c[0], c[1], c[2]])
                .collect()
        })
    }
}

pub fn write_f64(path: impl AsRef<Path>, shape: &[usize], data: &[f64]) -> std::io::Result<()> {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let dims = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    };
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {dims}, }}");
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

pub fn write_rows3(path: impl AsRef<Path>, rows: &[[f64; 3]]) -> std::io::Result<()> {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    write_f64(path, &[rows.len(), 3], &flat)
}

pub fn read(path: impl AsRef<Path>) -> std::io::Result<Array> {
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("not an NPY file"));
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        _ => return Err(bad("unsupported NPY version")),
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen]).map_err(|_| bad("bad header"))?;
    let field = |key: &str| -> Option<&str> {
        let i = header.find(key)? + key.len();
        let rest = header[i..].trim_start_matches([' ', ':', '\'']);
        Some(rest)
    };
    let descr = field("'descr'").ok_or_else(|| bad("no descr"))?;
    let descr = &descr[..descr.find('\'').ok_or_else(|| bad("bad descr"))?];
    if field("'fortran_order'").is_some_and(|s| s.starts_with("True")) {
        return Err(bad("Fortran order not supported"));
    }
    let shape_s = field("'shape'").ok_or_else(|| bad("no shape"))?;
    let shape_s = &shape_s
        [shape_s.find('(').ok_or_else(|| bad("bad shape"))? + 1..shape_s.find(')').ok_or_else(|| bad("bad shape"))?];
    let shape: Vec<usize> = shape_s
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|_| bad("bad dim")))
        .collect::<Result<_, _>>()?;
    let n: usize = shape.iter().product();
    let body = &bytes[start + hlen..];
    let data: Vec<f64> = match descr {
        "<f8" => body
            .as_chunks::<8>()
            .0
            .iter()
            .take(n)
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
        "<f4" => body
            .as_chunks::<4>()
            .0
            .iter()
            .take(n)
            .map(|c| f32::from_le_bytes(*c) as f64)
            .collect(),
        "<i8" => body
            .as_chunks::<8>()
            .0
            .iter()
            .take(n)
            .map(|c| i64::from_le_bytes(*c) as f64)
            .collect(),
        "<i4" => body
            .as_chunks::<4>()
            .0
            .iter()
            .take(n)
            .map(|c| i32::from_le_bytes(*c) as f64)
            .collect(),
        d => return Err(bad(&format!("unsupported dtype {d}"))),
    };
    if data.len() != n {
        return Err(bad("truncated data"));
    }
    Ok(Array { shape, data })
}

#[cfg(test)]
mod tests {
    #[test]
    fn round_trip() {
        let dir = std::env::temp_dir().join(format!("gyroxide-npy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.npy");
        let rows = vec![[1.0, 2.0, 3.0], [-4.5, 5e-300, 6.0]];
        super::write_rows3(&p, &rows).unwrap();
        let a = super::read(&p).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.rows3().unwrap(), rows);
        std::fs::remove_dir_all(dir).ok();
    }
}
