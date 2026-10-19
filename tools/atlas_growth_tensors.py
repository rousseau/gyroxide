#!/usr/bin/env python3
"""Anisotropic tangential growth tensors measured on the dHCP atlas (same triangulation every week).

    python3 tools/atlas_growth_tensors.py data/meshes/w21_left_610k [--smooth 200] [--kappa 1.0]

For each week w (knots t = (w - 21)/15) and each triangle, the surface deformation from
week 21 to week w (vertex correspondence of the atlas) gives the right Cauchy-Green tensor
C = F^T F in the reference tangent plane; it is lifted to 3D (B C B^T, B an orthonormal
tangent basis), averaged at the vertices and smoothed like the scalar tables. Per vertex,
with lambda_1 >= lambda_2 the tangent eigenvalues and d_1, d_2 their directions:
    T = (sqrt(lambda_1) - 1) d_1 d_1^T + (sqrt(lambda_2) - 1) d_2 d_2^T
so that the growth G = I + g_m T stretches the reference surface by sqrt(lambda_k) along d_k.
kappa scales the anisotropy around the isotropic part s (I - n n^T), s = mean of the two:
    T_kappa = s (I - n n^T) + kappa (T - s (I - n n^T)).
Writes <base>.nodal_T<kappa>.npy, shape [16, n_nodes, 6] (xx, yy, zz, xy, xz, yz), interior
nodes taking their nearest surface vertex; prints anisotropy statistics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.spatial import cKDTree

ATLAS = next((p for p in [Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas"),
              Path(__file__).resolve().parent.parent / "data/atlas_dhcp"] if p.is_dir()), None)
ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("--smooth", type=int, default=200)
ap.add_argument("--kappa", type=float, default=1.0)
args = ap.parse_args()

frame = json.loads(Path(args.base + ".json").read_text())
ns, c0, d0 = frame["n_surface"], np.array(frame["cog_mm"]), frame["maxd_mm"]
lines = Path(args.base + ".mesh").read_text().split("\n")
n_nodes = int(lines[0])
X = np.loadtxt(lines[1:1 + n_nodes])
nearest = cKDTree(X[:ns]).query(X[ns:])[1]
tri = pv.read(ATLAS / "fetal.week21.left.pial.surf.vtk").extract_surface().faces.reshape(-1, 4)[:, 1:]
e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns))
adj = ((adj + adj.T) > 0).astype(float).tocsr()
deg = np.asarray(adj.sum(1)).ravel()


def frame_pts(w):
    p = np.asarray(pv.read(ATLAS / f"fetal.week{w}.left.pial.surf.vtk").points, float)
    return np.column_stack([-(p[:, 1] - c0[1]), p[:, 0] - c0[0], -(p[:, 2] - c0[2])]) / d0


def smooth(v):
    for _ in range(args.smooth):
        v = 0.5 * v + 0.5 * (adj @ v) / deg[:, None]
    return v


P0 = frame_pts(21)
a, b, c = P0[tri[:, 0]], P0[tri[:, 1]], P0[tri[:, 2]]
e1 = b - a
e2 = c - a
nrm = np.cross(e1, e2)
area0 = 0.5 * np.linalg.norm(nrm, axis=1)
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
t1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
t2 = np.cross(nrm, t1)
B = np.stack([t1, t2], axis=2)  # [nt, 3, 2] reference tangent basis
R = np.stack([np.einsum("ij,ijk->ik", e1, B), np.einsum("ij,ijk->ik", e2, B)], axis=2)  # [nt, 2, 2] 2D ref edges (cols)
Rinv = np.linalg.inv(R)
# vertex normals of the reference surface
vn = np.zeros((ns, 3))
for k in range(3):
    np.add.at(vn, tri[:, k], nrm * area0[:, None])
vn /= np.linalg.norm(vn, axis=1, keepdims=True)
proj = np.eye(3)[None] - vn[:, :, None] * vn[:, None, :]

out = np.zeros((16, n_nodes, 6))
for k, w in enumerate(range(21, 37)):
    P = frame_pts(w)
    E = np.stack([P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]]], axis=2)  # [nt, 3, 2]
    Fm = E @ Rinv  # maps 2D reference tangent coordinates to 3D current
    C2 = np.transpose(Fm, (0, 2, 1)) @ Fm  # [nt, 2, 2]
    # Stretch U = sqrt(C2) per triangle; growth tensor U - I lifted to 3D (exactly 0 at week 21),
    # then averaged linearly at the vertices, smoothed, projected on the vertex tangent plane.
    lam2, vec2 = np.linalg.eigh(C2)
    U = vec2 @ (np.sqrt(np.clip(lam2, 1e-12, None))[:, :, None] * np.transpose(vec2, (0, 2, 1)))
    T3 = B @ (U - np.eye(2)[None]) @ np.transpose(B, (0, 2, 1))
    acc = np.zeros((ns, 3, 3))
    wsum = np.zeros(ns)
    for j in range(3):
        np.add.at(acc, tri[:, j], T3 * area0[:, None, None])
        np.add.at(wsum, tri[:, j], area0)
    T = smooth((acc / wsum[:, None, None]).reshape(ns, 9)).reshape(ns, 3, 3)
    T = proj @ (0.5 * (T + np.transpose(T, (0, 2, 1)))) @ proj
    lam, vec = np.linalg.eigh(T)
    # tangent eigenvalues: the two of largest |value| (the normal one is ~0)
    order = np.argsort(np.abs(lam), axis=1)
    s1 = np.take_along_axis(lam, order[:, 2:3], 1)[:, 0]
    s2 = np.take_along_axis(lam, order[:, 1:2], 1)[:, 0]
    s1, s2 = np.maximum(s1, s2), np.minimum(s1, s2)
    s = 0.5 * np.trace(T, axis1=1, axis2=2)
    Tk = s[:, None, None] * proj + args.kappa * (T - s[:, None, None] * proj)
    comp = np.stack([Tk[:, 0, 0], Tk[:, 1, 1], Tk[:, 2, 2], Tk[:, 0, 1], Tk[:, 0, 2], Tk[:, 1, 2]], axis=1)
    out[k, :ns] = comp
    out[k, ns:] = comp[nearest]
    if w in (25, 29, 33, 36):
        ani = (s1 - s2) / np.maximum(np.abs(s), 1e-6)
        aiso = np.sqrt((1 + s1) * (1 + s2)) - 1
        print(f"SG {w}: s1 median {np.median(s1):.3f}, s2 median {np.median(s2):.3f}, "
              f"(s1 - s2)/s median {np.median(ani):.2f} p90 {np.percentile(ani, 90):.2f}; "
              f"area-equivalent a median {np.median(aiso):.3f}")
path = f"{args.base}.nodal_T{args.kappa:g}.npy"
np.save(path, out)
print(path, out.shape)
