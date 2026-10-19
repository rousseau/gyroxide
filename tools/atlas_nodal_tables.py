#!/usr/bin/env python3
"""Per-node a_t and H tables from the dHCP atlas, for `law = "nodal"`.

    python3 tools/atlas_nodal_tables.py data/meshes/w21_left_610k [--smooth 20]

All atlas weeks share one triangulation, so vertex i is the same point of the
(registered) cortex at every week. For weeks 21..36 (knots t = (SG - 21)/15):
  a[k, i] = sqrt(A_i(SG) / A_i(21)) - 1      A_i = vertex area (1/3 of the incident
                                             triangles), smoothed on the surface
  h[k, i] = |pial_i - white_i| / 32.986 mm  smoothed, floored at --h-min mm
Interior mesh nodes take the value of the nearest surface vertex (as the growth
mask). Writes <base>.nodal_a.npy and <base>.nodal_h.npy, shape [16, n_nodes].
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
ap.add_argument("base", help="mesh path without extension (reads .mesh and .json)")
ap.add_argument("--smooth", type=int, default=20, help="neighbour-averaging iterations")
ap.add_argument("--h-min", type=float, default=0.3, help="thickness floor (mm)")
args = ap.parse_args()

frame = json.loads(Path(args.base + ".json").read_text())
ns, d_mm = frame["n_surface"], frame["maxd_mm"]
lines = Path(args.base + ".mesh").read_text().split("\n", 2 + 0)
n_nodes = int(lines[0])
X = np.loadtxt(Path(args.base + ".mesh").read_text().split("\n")[1:1 + n_nodes])
nearest = cKDTree(X[:ns]).query(X[ns:])[1]

tri = pv.read(ATLAS / "fetal.week21.left.pial.surf.vtk").extract_surface().faces.reshape(-1, 4)[:, 1:]
e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns))
adj = ((adj + adj.T) > 0).astype(float).tocsr()
deg = np.asarray(adj.sum(1)).ravel()


def smooth(v):
    for _ in range(args.smooth):
        v = 0.5 * v + 0.5 * (adj @ v) / deg
    return v


def vertex_area(p):
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    ta = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return np.bincount(tri.ravel(), weights=np.repeat(ta / 3, 3), minlength=ns)


def to_nodes(v):
    out = np.empty(n_nodes)
    out[:ns] = v
    out[ns:] = v[nearest]
    return out


weeks = range(21, 37)
pts = {w: np.asarray(pv.read(ATLAS / f"fetal.week{w}.left.pial.surf.vtk").points, float) for w in weeks}
area21 = smooth(vertex_area(pts[21]))
A, H = [], []
for w in weeks:
    ratio = smooth(vertex_area(pts[w])) / area21
    a = np.sqrt(ratio) - 1
    white = np.asarray(pv.read(ATLAS / f"fetal.week{w}.left.wm.surf.vtk").points, float)
    h = np.maximum(smooth(np.linalg.norm(pts[w] - white, axis=1)), args.h_min) / d_mm
    A.append(to_nodes(a))
    H.append(to_nodes(h))
    g = np.sqrt(vertex_area(pts[w]).sum() / vertex_area(pts[21]).sum()) - 1
    print(f"SG {w}: a median {np.median(a):.3f} (p10 {np.percentile(a, 10):.3f}, p90 {np.percentile(a, 90):.3f}), "
          f"global sqrt(A/A21)-1 {g:.3f} | H median {np.median(h) * d_mm:.2f} mm")
np.save(args.base + ".nodal_a.npy", np.array(A))
np.save(args.base + ".nodal_h.npy", np.array(H))
print(args.base + ".nodal_a.npy", args.base + ".nodal_h.npy", np.array(A).shape)
