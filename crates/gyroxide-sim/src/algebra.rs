//! Batched 3-vectors and 3x3 matrices as structures of 1-D Burn tensors
//! (one tensor of length T per component).
//!
//! This is the reference ("tensor") level: each operation is an element-wise
//! Burn op, which the Fusion backend merges into a few kernels. The layout
//! mirrors what a CubeCL kernel holds in registers, one thread per element.

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor, TensorData};

pub type T1<B> = Tensor<B, 1>;
pub type I1<B> = Tensor<B, 1, Int>;

/// Uploads f64 values; they are stored in the backend's float type
/// (f32 on wgpu, f64 with `NdArray<f64>`).
pub fn floats<B: Backend>(v: &[f64], device: &B::Device) -> T1<B> {
    T1::<B>::from_data(TensorData::new(v.to_vec(), [v.len()]), device)
}

pub fn ints<B: Backend>(v: &[u32], device: &B::Device) -> I1<B> {
    let v: Vec<i64> = v.iter().map(|&i| i as i64).collect();
    let n = v.len();
    I1::<B>::from_data(TensorData::new(v, [n]), device)
}

pub fn to_f64<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data()
        .convert::<f64>()
        .to_vec::<f64>()
        .expect("float tensor readback")
}

pub fn scalar<B: Backend>(t: T1<B>) -> f64 {
    to_f64(t)[0]
}

#[derive(Clone, Debug)]
pub struct V3<B: Backend> {
    pub x: T1<B>,
    pub y: T1<B>,
    pub z: T1<B>,
}

impl<B: Backend> V3<B> {
    pub fn new(x: T1<B>, y: T1<B>, z: T1<B>) -> Self {
        Self { x, y, z }
    }
    pub fn zeros(n: usize, device: &B::Device) -> Self {
        let z = T1::<B>::zeros([n], device);
        Self::new(z.clone(), z.clone(), z)
    }
    pub fn from_rows(rows: &[[f64; 3]], device: &B::Device) -> Self {
        let col = |k: usize| floats::<B>(&rows.iter().map(|r| r[k]).collect::<Vec<_>>(), device);
        Self::new(col(0), col(1), col(2))
    }
    pub fn to_rows(&self) -> Vec<[f64; 3]> {
        let (x, y, z) = (to_f64(self.x.clone()), to_f64(self.y.clone()), to_f64(self.z.clone()));
        (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect()
    }
    pub fn len(&self) -> usize {
        self.x.dims()[0]
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn select(&self, idx: &I1<B>) -> Self {
        Self::new(
            self.x.clone().select(0, idx.clone()),
            self.y.clone().select(0, idx.clone()),
            self.z.clone().select(0, idx.clone()),
        )
    }
    pub fn add(&self, o: &Self) -> Self {
        Self::new(
            self.x.clone() + o.x.clone(),
            self.y.clone() + o.y.clone(),
            self.z.clone() + o.z.clone(),
        )
    }
    pub fn sub(&self, o: &Self) -> Self {
        Self::new(
            self.x.clone() - o.x.clone(),
            self.y.clone() - o.y.clone(),
            self.z.clone() - o.z.clone(),
        )
    }
    pub fn neg(&self) -> Self {
        Self::new(-self.x.clone(), -self.y.clone(), -self.z.clone())
    }
    pub fn mul(&self, s: &T1<B>) -> Self {
        Self::new(
            self.x.clone() * s.clone(),
            self.y.clone() * s.clone(),
            self.z.clone() * s.clone(),
        )
    }
    pub fn mul_scalar(&self, s: f64) -> Self {
        Self::new(self.x.clone() * s, self.y.clone() * s, self.z.clone() * s)
    }
    pub fn dot(&self, o: &Self) -> T1<B> {
        self.x.clone() * o.x.clone() + self.y.clone() * o.y.clone() + self.z.clone() * o.z.clone()
    }
    pub fn cross(&self, o: &Self) -> Self {
        Self::new(
            self.y.clone() * o.z.clone() - self.z.clone() * o.y.clone(),
            self.z.clone() * o.x.clone() - self.x.clone() * o.z.clone(),
            self.x.clone() * o.y.clone() - self.y.clone() * o.x.clone(),
        )
    }
    pub fn norm(&self) -> T1<B> {
        self.dot(self).sqrt()
    }
    /// v / |v|, leaving zero vectors at zero.
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        let safe = n.clone().mask_fill(n.equal_elem(0.0), 1.0);
        self.mul(&safe.recip())
    }
    /// Where `mask` is true take `other`, else keep `self`.
    pub fn where_(&self, mask: &Tensor<B, 1, Bool>, other: &Self) -> Self {
        Self::new(
            self.x.clone().mask_where(mask.clone(), other.x.clone()),
            self.y.clone().mask_where(mask.clone(), other.y.clone()),
            self.z.clone().mask_where(mask.clone(), other.z.clone()),
        )
    }
    /// Concatenates several batches along the batch axis.
    pub fn cat(parts: Vec<Self>) -> Self {
        let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
        for p in parts {
            xs.push(p.x);
            ys.push(p.y);
            zs.push(p.z);
        }
        Self::new(T1::cat(xs, 0), T1::cat(ys, 0), T1::cat(zs, 0))
    }
}

/// 3x3 matrices, `m[r][c]`.
#[derive(Clone, Debug)]
pub struct M3<B: Backend> {
    pub m: [[T1<B>; 3]; 3],
}

impl<B: Backend> M3<B> {
    pub fn from_cols(c0: &V3<B>, c1: &V3<B>, c2: &V3<B>) -> Self {
        Self {
            m: [
                [c0.x.clone(), c1.x.clone(), c2.x.clone()],
                [c0.y.clone(), c1.y.clone(), c2.y.clone()],
                [c0.z.clone(), c1.z.clone(), c2.z.clone()],
            ],
        }
    }
    pub fn identity(n: usize, device: &B::Device) -> Self {
        let o = T1::<B>::ones([n], device);
        let z = T1::<B>::zeros([n], device);
        Self {
            m: [
                [o.clone(), z.clone(), z.clone()],
                [z.clone(), o.clone(), z.clone()],
                [z.clone(), z, o],
            ],
        }
    }
    pub fn col(&self, c: usize) -> V3<B> {
        V3::new(self.m[0][c].clone(), self.m[1][c].clone(), self.m[2][c].clone())
    }
    pub fn set_col(&mut self, c: usize, v: &V3<B>) {
        self.m[0][c] = v.x.clone();
        self.m[1][c] = v.y.clone();
        self.m[2][c] = v.z.clone();
    }
    pub fn get(&self, r: usize, c: usize) -> T1<B> {
        self.m[r][c].clone()
    }
    pub fn transpose(&self) -> Self {
        Self {
            m: std::array::from_fn(|r| std::array::from_fn(|c| self.get(c, r))),
        }
    }
    pub fn add(&self, o: &Self) -> Self {
        Self {
            m: std::array::from_fn(|r| std::array::from_fn(|c| self.get(r, c) + o.get(r, c))),
        }
    }
    pub fn sub(&self, o: &Self) -> Self {
        Self {
            m: std::array::from_fn(|r| std::array::from_fn(|c| self.get(r, c) - o.get(r, c))),
        }
    }
    pub fn mul_s(&self, s: &T1<B>) -> Self {
        Self {
            m: std::array::from_fn(|r| std::array::from_fn(|c| self.get(r, c) * s.clone())),
        }
    }
    pub fn matmul(&self, o: &Self) -> Self {
        Self {
            m: std::array::from_fn(|r| {
                std::array::from_fn(|c| {
                    self.get(r, 0) * o.get(0, c) + self.get(r, 1) * o.get(1, c) + self.get(r, 2) * o.get(2, c)
                })
            }),
        }
    }
    pub fn mul_vec(&self, v: &V3<B>) -> V3<B> {
        let row = |r: usize| self.get(r, 0) * v.x.clone() + self.get(r, 1) * v.y.clone() + self.get(r, 2) * v.z.clone();
        V3::new(row(0), row(1), row(2))
    }
    pub fn trace(&self) -> T1<B> {
        self.get(0, 0) + self.get(1, 1) + self.get(2, 2)
    }
    pub fn det(&self) -> T1<B> {
        self.col(0).dot(&self.col(1).cross(&self.col(2)))
    }
    /// Adjugate (transpose of the cofactor matrix): inv = adj / det.
    pub fn adjugate(&self) -> Self {
        let a = |r: usize, c: usize| self.get(r, c);
        let cof = |r0: usize, r1: usize, c0: usize, c1: usize| a(r0, c0) * a(r1, c1) - a(r0, c1) * a(r1, c0);
        Self {
            m: [
                [cof(1, 2, 1, 2), -cof(0, 2, 1, 2), cof(0, 1, 1, 2)],
                [-cof(1, 2, 0, 2), cof(0, 2, 0, 2), -cof(0, 1, 0, 2)],
                [cof(1, 2, 0, 1), -cof(0, 2, 0, 1), cof(0, 1, 0, 1)],
            ],
        }
    }
    pub fn inverse(&self) -> Self {
        self.adjugate().mul_s(&self.det().recip())
    }
    pub fn where_(&self, mask: &Tensor<B, 1, Bool>, other: &Self) -> Self {
        Self {
            m: std::array::from_fn(|r| {
                std::array::from_fn(|c| self.get(r, c).mask_where(mask.clone(), other.get(r, c)))
            }),
        }
    }
    /// diag(d0, d1, d2) * self (scales rows).
    pub fn scale_rows(&self, d: [&T1<B>; 3]) -> Self {
        Self {
            m: std::array::from_fn(|r| std::array::from_fn(|c| self.get(r, c) * d[r].clone())),
        }
    }
    /// self * diag(d0, d1, d2) (scales columns).
    pub fn scale_cols(&self, d: [&T1<B>; 3]) -> Self {
        Self {
            m: std::array::from_fn(|r| std::array::from_fn(|c| self.get(r, c) * d[c].clone())),
        }
    }
}
