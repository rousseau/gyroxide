#!/usr/bin/env python3
"""Growth tensors of the core (white matter, subplate) measured on the dHCP atlas.

    python3 tools/atlas_core_tensors.py data/meshes/w21_left_610k [--smooth 200]

For each week w (knots t = (w - 21)/15): the displacement of the atlas wm surface from week 21
(vertex correspondence; wm vertex i lies under pial vertex i = surface node i of the mesh),
low-passed by --smooth neighbour averagings on the surface (lobar scale: the folds are left to
the model), is put on the surface nodes and extended harmonically to the interior nodes (graph
Laplacian of the tetrahedral mesh). Per tetrahedron, F = D(X + u) D(X)^-1 and its stretch
U = sqrt(F^T F) (rotation removed); U is averaged at the nodes (volume weights). The simulation
uses G = I + g_m a (I - n n^T) + (1 - g_m)(U - I) (schedule key `core`).
Writes <base>.nodal_core.npy, shape [16, n_nodes, 6] (xx, yy, zz, xy, xz, yz), and prints the
volume growth det U of the core per week next to the atlas wm volume ratio.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

ROOT = Path(__file__).resolve().parent.parent
ATLAS = next((p for p in [Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas"),
              ROOT / "data/atlas_dhcp"] if p.is_dir()), None)
ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("--smooth", type=int, default=200)
args = ap.parse_args()

frame = json.loads(Path(args.base + ".json").read_text())
ns, c0, d0 = frame["n_surface"], np.array(frame["cog_mm"]), frame["maxd_mm"]
lines = Path(args.base + ".mesh").read_text().split("\n")
n_nodes = int(lines[0])
X = np.loadtxt(lines[1:1 + n_nodes])
n_tets = int(lines[1 + n_nodes])
T = np.loadtxt(lines[2 + n_nodes:2 + n_nodes + n_tets], dtype=np.int64)[:, 1:5] - 1


def surf(w, kind):
    g = nib.load(ATLAS / f"fetal.week{w}.left.{kind}.surf.gii")
    p = g.agg_data("NIFTI_INTENT_POINTSET").astype(float)
    return np.column_stack([-(p[:, 1] - c0[1]), p[:, 0] - c0[0], -(p[:, 2] - c0[2])]) / d0, g.agg_data("NIFTI_INTENT_TRIANGLE")


wm21, tri = surf(21, "wm")
assert len(wm21) == ns
e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns))
A = ((A + A.T) > 0).astype(float).tocsr()
S = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A

# Graph Laplacian of the tet mesh, Dirichlet on the surface nodes.
pairs = np.concatenate([T[:, [a, b]] for a in range(4) for b in range(a + 1, 4)])
W = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_nodes, n_nodes))
W = ((W + W.T) > 0).astype(float).tocsr()
L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
interior = np.arange(ns, n_nodes)
solve = spl.factorized(L[interior][:, interior].tocsc())
L_ib = L[interior][:, :ns]

D0 = np.stack([X[T[:, k]] - X[T[:, 0]] for k in (1, 2, 3)], axis=2)  # columns = edges
D0inv = np.linalg.inv(D0)
vol0 = np.abs(np.linalg.det(D0)) / 6
wsum = np.bincount(T.ravel(), weights=np.repeat(vol0, 4), minlength=n_nodes)

out = np.zeros((16, n_nodes, 6))
V21 = None
print(" SG  core det U (median, vol-weighted mean)  atlas wm volume ratio")
for k, w in enumerate(range(21, 37)):
    wmw, _ = surf(w, "wm")
    us = wmw - wm21
    for _ in range(args.smooth):
        us = S @ us
    u = np.zeros((n_nodes, 3))
    u[:ns] = us
    u[ns:] = np.column_stack([solve(-L_ib @ us[:, d]) for d in range(3)])
    Y = X + u
    D = np.stack([Y[T[:, j]] - Y[T[:, 0]] for j in (1, 2, 3)], axis=2)
    F = D @ D0inv
    lam, vec = np.linalg.eigh(np.einsum("nji,njk->nik", F, F))
    U = np.einsum("nij,nj,nkj->nik", vec, np.sqrt(np.clip(lam, 1e-12, None)), vec)
    comp = np.stack([U[:, 0, 0], U[:, 1, 1], U[:, 2, 2], U[:, 0, 1], U[:, 0, 2], U[:, 1, 2]], axis=1)
    for c in range(6):
        out[k, :, c] = np.bincount(T.ravel(), weights=np.repeat(comp[:, c] * vol0, 4), minlength=n_nodes) / wsum
    detu = np.linalg.det(U)
    a = np.abs(np.einsum("ij,ij->i", wmw[tri[:, 0]], np.cross(wmw[tri[:, 1]], wmw[tri[:, 2]])).sum())
    V21 = V21 or a
    print(f"{w:3d}  {np.median(detu):.3f}  {(detu * vol0).sum() / vol0.sum():.3f}   {a / V21:.3f}")
path = args.base + ".nodal_core.npy"
np.save(path, out)
print(path, out.shape)
