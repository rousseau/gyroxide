//! Nearest surface node and distance to the surface for every node
//! (the C code's `dist2surf`, which is a brute-force O(N * S) loop).
//!
//! We use a uniform grid with an expanding shell search. Ties are broken
//! towards the smallest surface index, like the strict `<` of the C loop.

use crate::{dot, sub, Vec3};
use std::collections::HashMap;

pub struct Nearest {
    /// Index into the query set (here: surface node index, `csn`).
    pub index: Vec<u32>,
    /// Euclidean distance (`d2s`).
    pub dist: Vec<f64>,
}

pub fn nearest(points: &[Vec3], targets: &[Vec3]) -> Nearest {
    assert!(!targets.is_empty(), "nearest: empty target set");
    let mut mn = [f64::INFINITY; 3];
    let mut mx = [f64::NEG_INFINITY; 3];
    for p in targets {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    let ext = (0..3).map(|k| mx[k] - mn[k]).fold(0.0f64, f64::max).max(1e-12);
    // ~ 2 targets per occupied cell for a surface point cloud.
    let cell = ext / (targets.len() as f64 / 2.0).sqrt().max(1.0);
    let key = |p: &Vec3| -> [i64; 3] { std::array::from_fn(|k| ((p[k] - mn[k]) / cell).floor() as i64) };
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for (j, p) in targets.iter().enumerate() {
        grid.entry(key(p)).or_default().push(j as u32);
    }
    let max_ring = ((ext / cell).ceil() as i64) + 2;

    let mut index = Vec::with_capacity(points.len());
    let mut dist = Vec::with_capacity(points.len());
    for p in points {
        let c = key(p);
        // Distance from p to the grid box of its own cell, used to bound rings.
        let mut best = (f64::INFINITY, u32::MAX);
        let mut r = 0i64;
        loop {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue;
                        }
                        if let Some(list) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            for &j in list {
                                let d = sub(targets[j as usize], *p);
                                let d2 = dot(d, d);
                                if d2 < best.0 || (d2 == best.0 && j < best.1) {
                                    best = (d2, j);
                                }
                            }
                        }
                    }
                }
            }
            // Every unvisited cell is at least r * cell away from p.
            let reach = r as f64 * cell;
            if (best.1 != u32::MAX && best.0 < reach * reach)
                || r > max_ring + (c.iter().map(|v| v.abs()).max().unwrap_or(0))
            {
                break;
            }
            r += 1;
        }
        index.push(best.1);
        dist.push(best.0.sqrt());
    }
    Nearest { index, dist }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_brute_force() {
        let mut s = 12345u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let targets: Vec<Vec3> = (0..300).map(|_| [rnd(), rnd(), rnd()]).collect();
        let points: Vec<Vec3> = (0..500).map(|_| [rnd() * 1.5, rnd() * 1.5, rnd() * 1.5]).collect();
        let got = nearest(&points, &targets);
        for (i, p) in points.iter().enumerate() {
            let (mut bd, mut bj) = (f64::INFINITY, 0);
            for (j, t) in targets.iter().enumerate() {
                let d = sub(*t, *p);
                if dot(d, d) < bd {
                    bd = dot(d, d);
                    bj = j;
                }
            }
            assert_eq!(got.index[i] as usize, bj);
            assert!((got.dist[i] - bd.sqrt()).abs() < 1e-12);
        }
    }
}
