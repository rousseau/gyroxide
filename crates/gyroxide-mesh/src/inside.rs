//! Points inside a closed triangulated surface (e.g. nodes in the white matter, inside the white
//! surface): parity of the crossings of a ray along +x, triangles bucketed on a grid of the (y, z)
//! plane. The ray origins are shifted by a tiny irrational offset in (y, z) so that no ray passes
//! exactly through a vertex or an edge of the surface.

use crate::Vec3;

/// For each point, whether it lies inside the closed surface (`vertices`, `faces`).
pub fn inside(points: &[Vec3], vertices: &[Vec3], faces: &[[u32; 3]]) -> Vec<bool> {
    if faces.is_empty() {
        return vec![false; points.len()];
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in vertices {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k + 1]);
            hi[k] = hi[k].max(p[k + 1]);
        }
    }
    let size = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(f64::MIN_POSITIVE);
    let shift = [size * 1e-7 * std::f64::consts::PI, size * 1e-7 * std::f64::consts::E];
    let n = ((faces.len() as f64).sqrt() / 2.0).clamp(1.0, 1024.0) as usize;
    let cell = |v: f64, k: usize| {
        (((v - lo[k]) / (hi[k] - lo[k]).max(f64::MIN_POSITIVE) * n as f64) as isize).clamp(0, n as isize - 1) as usize
    };
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); n * n];
    for (fi, f) in faces.iter().enumerate() {
        let [a, b, c] = f.map(|i| vertices[i as usize]);
        let (y0, y1) = (a[1].min(b[1]).min(c[1]), a[1].max(b[1]).max(c[1]));
        let (z0, z1) = (a[2].min(b[2]).min(c[2]), a[2].max(b[2]).max(c[2]));
        for i in cell(y0, 0)..=cell(y1, 0) {
            for j in cell(z0, 1)..=cell(z1, 1) {
                grid[i * n + j].push(fi as u32);
            }
        }
    }
    points
        .iter()
        .map(|p| {
            let (y, z) = (p[1] + shift[0], p[2] + shift[1]);
            if y < lo[0] || y > hi[0] || z < lo[1] || z > hi[1] {
                return false;
            }
            let mut crossings = 0usize;
            for &fi in &grid[cell(y, 0) * n + cell(z, 1)] {
                let [a, b, c] = faces[fi as usize].map(|i| vertices[i as usize]);
                // Barycentric coordinates of (y, z) in the projected triangle.
                let d = (b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2]);
                if d == 0.0 {
                    continue;
                }
                let u = ((y - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (z - a[2])) / d;
                let v = ((b[1] - a[1]) * (z - a[2]) - (y - a[1]) * (b[2] - a[2])) / d;
                if u < 0.0 || v < 0.0 || u + v > 1.0 {
                    continue;
                }
                let x = a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]);
                if x > p[0] {
                    crossings += 1;
                }
            }
            crossings % 2 == 1
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::voxel_ball;

    #[test]
    fn points_dans_une_boule_et_hors() {
        // Boundary of a voxel ball of radius 1: points well inside / outside.
        let m = voxel_ball(8, 1.0);
        let pts = [
            [0.0, 0.0, 0.0],
            [0.3, -0.2, 0.1],
            [1.5, 0.0, 0.0],
            [-2.0, 0.3, 0.0],
            [0.0, 0.0, 1.4],
        ];
        assert_eq!(inside(&pts, &m.nodes, &m.faces), [true, true, false, false, false]);
        // every interior node of the mesh is inside its own boundary
        let interior: Vec<Vec3> = {
            let mut on = vec![false; m.n_nodes()];
            m.faces.iter().flatten().for_each(|&i| on[i as usize] = true);
            m.nodes.iter().zip(&on).filter(|(_, &b)| !b).map(|(p, _)| *p).collect()
        };
        assert!(inside(&interior, &m.nodes, &m.faces).iter().all(|&b| b));
    }
}
