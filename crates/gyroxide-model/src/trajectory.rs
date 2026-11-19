//! Per-vertex growth trajectories (P07, E61): a(t) = A (S((t - t0)/r) - S(-t0/r)) / (S((1 - t0)/r) - S(-t0/r)),
//! with S the logistic function (`Curve::Sigmoid`) or the Gompertz function exp(-exp(-z)) (`Curve::Gompertz`).
//!
//! The curve is normalised so that a(0) = 0 (the rest state at SG 21) and a(1) = A (the value at SG 36, t = (SG - 21)/15):
//! `A` is the final amplitude, `t0` the date of the fastest growth and `r` the duration scale. The analytic derivatives
//! with respect to (A, t0, r) are what the adjoint gradient of the joint cost is assembled from; the fits start the
//! trajectory from the growth measured on the atlas (a table with one row per week).

/// Family of growth curves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Curve {
    Sigmoid,
    Gompertz,
}

/// Parameters of one vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Final amplitude a(1).
    pub amp: f64,
    /// Date of the fastest growth (t = 0 at SG 21, 1 at SG 36).
    pub t0: f64,
    /// Duration scale, > 0.
    pub r: f64,
}

impl Curve {
    /// The unnormalised shape S(z).
    fn shape(self, z: f64) -> f64 {
        match self {
            Curve::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            Curve::Gompertz => (-(-z).exp()).exp(),
        }
    }

    /// dS/dz.
    fn dshape(self, z: f64) -> f64 {
        match self {
            Curve::Sigmoid => {
                let s = self.shape(z);
                s * (1.0 - s)
            }
            Curve::Gompertz => self.shape(z) * (-z).exp(),
        }
    }

    /// g(t) = S((t - t0)/r) - S(-t0/r) and its denominator D = g(1), with their derivatives in t0 and r:
    /// (g, D, [dg/dt0, dg/dr], [dD/dt0, dD/dr]).
    fn parts(self, t: f64, t0: f64, r: f64) -> (f64, f64, [f64; 2], [f64; 2]) {
        let (z, z0, z1) = ((t - t0) / r, -t0 / r, (1.0 - t0) / r);
        let (s, s0, s1) = (self.dshape(z), self.dshape(z0), self.dshape(z1));
        let g = self.shape(z) - self.shape(z0);
        let d = self.shape(z1) - self.shape(z0);
        let dg = [(s0 - s) / r, -(z * s - z0 * s0) / r];
        let dd = [(s0 - s1) / r, -(z1 * s1 - z0 * s0) / r];
        (g, d, dg, dd)
    }

    /// a(t) for the parameters `p`.
    pub fn value(self, p: Params, t: f64) -> f64 {
        let (g, d, _, _) = self.parts(t, p.t0, p.r);
        p.amp * g / d
    }

    /// Unit-amplitude curve g(t)/D, the factor of the amplitude.
    pub fn unit(self, t0: f64, r: f64, t: f64) -> f64 {
        let (g, d, _, _) = self.parts(t, t0, r);
        g / d
    }

    /// Derivatives of a(t) with respect to (A, t0, r).
    pub fn gradient(self, p: Params, t: f64) -> [f64; 3] {
        let (g, d, dg, dd) = self.parts(t, p.t0, p.r);
        let dt0 = p.amp * (dg[0] * d - g * dd[0]) / (d * d);
        let dr = p.amp * (dg[1] * d - g * dd[1]) / (d * d);
        [g / d, dt0, dr]
    }

    /// Largest growth rate max_t da/dt (for the maps of the fastest growth). Both shapes peak where S'' = 0:
    /// z = 0 for the logistic function, z = 0 for the Gompertz function (S'(0) = 1/e), when that date lies in [0, 1];
    /// otherwise the maximum is at the nearest end.
    pub fn peak_rate(self, p: Params) -> f64 {
        let (_, d, _, _) = self.parts(1.0, p.t0, p.r);
        let t = p.t0.clamp(0.0, 1.0);
        p.amp * self.dshape((t - p.t0) / p.r) / (p.r * d)
    }
}

/// Amplitude of the least-squares fit at fixed (t0, r): A = sum f_n y_n / sum f_n^2, f_n the unit-amplitude curve at
/// the dates `ts` (closed form, the first stage of the nested fits).
pub fn fit_amplitude(curve: Curve, ts: &[f64], ys: &[f64], t0: f64, r: f64) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (&t, &y) in ts.iter().zip(ys) {
        let f = curve.unit(t0, r, t);
        num += f * y;
        den += f * f;
    }
    num / den
}

/// Sum of squared residuals of the curve against the series.
pub fn cost(curve: Curve, p: Params, ts: &[f64], ys: &[f64]) -> f64 {
    ts.iter().zip(ys).map(|(&t, &y)| (curve.value(p, t) - y).powi(2)).sum()
}

/// Levenberg-Marquardt fit of the free parameters (`free` = [A, t0, r]; r is moved in log scale so that it stays
/// positive) starting from `init`, `iters` iterations at most. Returns the fitted parameters.
pub fn fit_vertex(curve: Curve, ts: &[f64], ys: &[f64], init: Params, free: [bool; 3], iters: usize) -> Params {
    let mut p = init;
    let mut c = cost(curve, p, ts, ys);
    let mut lambda = 1e-3;
    let idx: Vec<usize> = (0..3).filter(|&k| free[k]).collect();
    let m = idx.len();
    for _ in 0..iters {
        // Normal equations in (A, t0, ln r).
        let mut jtj = vec![vec![0.0; m]; m];
        let mut jtr = vec![0.0; m];
        for (&t, &y) in ts.iter().zip(ys) {
            let g = curve.gradient(p, t);
            let col = [g[0], g[1], g[2] * p.r];
            let res = curve.value(p, t) - y;
            for (a, &ka) in idx.iter().enumerate() {
                jtr[a] += col[ka] * res;
                for (b, &kb) in idx.iter().enumerate() {
                    jtj[a][b] += col[ka] * col[kb];
                }
            }
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut a = jtj.clone();
            for (k, row) in a.iter_mut().enumerate() {
                row[k] += lambda * (jtj[k][k] + 1e-12);
            }
            let Some(step) = solve(&a, &jtr) else {
                lambda *= 10.0;
                continue;
            };
            let mut q = [p.amp, p.t0, p.r.ln()];
            for (a, &k) in idx.iter().enumerate() {
                q[k] -= step[a];
            }
            let cand = Params {
                amp: q[0],
                t0: q[1],
                r: q[2].exp(),
            };
            let cc = cost(curve, cand, ts, ys);
            if cc.is_finite() && cc < c {
                p = cand;
                c = cc;
                lambda = (lambda / 10.0).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    p
}

/// Solution of the small dense system `a x = b` by Gaussian elimination with partial pivoting.
fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, &v)| row.iter().copied().chain([v]).collect())
        .collect();
    for c in 0..n {
        let piv = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[piv][c].abs() < 1e-300 {
            return None;
        }
        m.swap(c, piv);
        for r in c + 1..n {
            let f = m[r][c] / m[c][c];
            let pivot_row = m[c].clone();
            for (x, p) in m[r].iter_mut().zip(&pivot_row).skip(c) {
                *x -= f * p;
            }
        }
    }
    let mut x = vec![0.0; n];
    for c in (0..n).rev() {
        let s: f64 = (c + 1..n).map(|k| m[c][k] * x[k]).sum();
        x[c] = (m[c][n] - s) / m[c][c];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CURVES: [Curve; 2] = [Curve::Sigmoid, Curve::Gompertz];
    const P: Params = Params {
        amp: 0.8,
        t0: 0.45,
        r: 0.18,
    };

    #[test]
    fn endpoints_are_zero_and_amplitude() {
        for c in CURVES {
            assert!(c.value(P, 0.0).abs() < 1e-14);
            assert!((c.value(P, 1.0) - P.amp).abs() < 1e-14);
            // growth only increases between the ends
            let v: Vec<f64> = (0..=20).map(|k| c.value(P, k as f64 / 20.0)).collect();
            assert!(v.windows(2).all(|w| w[1] > w[0]));
        }
    }

    #[test]
    fn gradient_matches_finite_differences() {
        for c in CURVES {
            for t in [0.1, 0.3, 0.6, 0.9] {
                let g = c.gradient(P, t);
                let h = 1e-6;
                let fd = [
                    (c.value(Params { amp: P.amp + h, ..P }, t) - c.value(Params { amp: P.amp - h, ..P }, t))
                        / (2.0 * h),
                    (c.value(Params { t0: P.t0 + h, ..P }, t) - c.value(Params { t0: P.t0 - h, ..P }, t)) / (2.0 * h),
                    (c.value(Params { r: P.r + h, ..P }, t) - c.value(Params { r: P.r - h, ..P }, t)) / (2.0 * h),
                ];
                for k in 0..3 {
                    assert!(
                        (g[k] - fd[k]).abs() < 1e-7 * (1.0 + fd[k].abs()),
                        "{c:?} t {t} k {k}: {} vs {}",
                        g[k],
                        fd[k]
                    );
                }
            }
        }
    }

    #[test]
    fn peak_rate_is_the_maximum_slope() {
        for c in CURVES {
            for p in [P, Params { t0: -0.2, ..P }, Params { t0: 1.3, r: 0.4, ..P }] {
                let best = (0..=2000)
                    .map(|k| {
                        let t = k as f64 / 2000.0;
                        (c.value(p, t + 1e-6) - c.value(p, t - 1e-6)) / 2e-6
                    })
                    .fold(f64::MIN, f64::max);
                assert!(
                    (c.peak_rate(p) - best).abs() < 1e-3 * best.abs(),
                    "{c:?} {p:?}: {} vs {best}",
                    c.peak_rate(p)
                );
            }
        }
    }

    #[test]
    fn amplitude_fit_is_exact_on_noiseless_data() {
        let ts: Vec<f64> = (1..=9).map(|k| k as f64 / 15.0).collect();
        for c in CURVES {
            let ys: Vec<f64> = ts.iter().map(|&t| c.value(P, t)).collect();
            assert!((fit_amplitude(c, &ts, &ys, P.t0, P.r) - P.amp).abs() < 1e-12);
        }
    }

    #[test]
    fn three_parameter_fit_recovers_the_curve() {
        let ts: Vec<f64> = (1..=9).map(|k| k as f64 / 15.0).collect();
        for c in CURVES {
            let ys: Vec<f64> = ts.iter().map(|&t| c.value(P, t)).collect();
            let init = Params {
                amp: 0.5,
                t0: 0.6,
                r: 0.3,
            };
            let fit = fit_vertex(c, &ts, &ys, init, [true; 3], 25);
            assert!(cost(c, fit, &ts, &ys) < 1e-12, "{c:?}: {fit:?}");
            // a frozen parameter stays put
            let part = fit_vertex(c, &ts, &ys, init, [true, false, false], 50);
            assert_eq!((part.t0, part.r), (init.t0, init.r));
        }
    }
}
