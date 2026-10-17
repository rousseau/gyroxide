#!/usr/bin/env python3
"""Inputs of examples/adjoint_week.rs for one week: atlas target, surface edges, cortex mask,
surface triangles, owner surface vertex of every node, initial per-vertex growth (atlas measurement).

    python3 tools/adjoint_prep.py 26 out/adjoint_w26
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
ATLAS = Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas")
week, out = int(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
F = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
NS, c, d = F["n_surface"], np.array(F["cog_mm"]), F["maxd_mm"]
p = np.asarray(pv.read(ATLAS / f"fetal.week{week}.left.pial.surf.vtk").points, float)
target = np.column_stack([-(p[:, 1] - c[1]), p[:, 0] - c[0], -(p[:, 2] - c[2])]) / d
tri = pv.read(ATLAS / "fetal.week21.left.pial.surf.vtk").extract_surface().faces.reshape(-1, 4)[:, 1:]
e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
edges = np.unique(e, axis=0)
X = np.load(ROOT / "out/atlas_fit/mesh_cache.npz")["X0"]
owner = np.arange(len(X))
owner[NS:] = cKDTree(X[:NS]).query(X[NS:])[1]
cortex = (np.loadtxt(ROOT / "data/meshes/w21_left_610k.gr.txt")[:NS] > 0.5).astype(float)
theta0 = np.load(ROOT / "data/meshes/w21_left_610k.nodal_a.npy")[week - 21, :NS]
for name, arr in [("target", target), ("edges", edges.astype(float)), ("faces", tri.astype(float)), ("owner", owner.astype(float)),
                  ("cortex", cortex), ("theta0", theta0)]:
    np.save(out / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float64))
print(out, target.shape, edges.shape, owner.shape, int(cortex.sum()), "theta0 median %.3f" % np.median(theta0))
