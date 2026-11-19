#!/usr/bin/env python3
"""Inputs of examples/adjoint_week.rs for one week: atlas target, surface edges, cortex mask,
surface triangles, owner surface vertex of every node, initial per-vertex growth (atlas measurement).

    python3 tools/adjoint_prep.py 26 out/adjoint_w26 [data/meshes/w21_right_610k]

Le troisième argument est le maillage de référence sans extension (défaut : l'hémisphère gauche) ; le côté de
l'atlas se lit dans son nom.
"""
import json
import sys
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hemi import atlas_surface, side_of, to_frame  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# dHCP fetal atlas (.vtk and .gii surfaces, weeks 21-36): $OXIDE_DHCP_ATLAS, else the repository's data/atlas_dhcp (tools/hemi.py).
week, out = int(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
BASE = ROOT / (sys.argv[3] if len(sys.argv) > 3 else "data/meshes/w21_left_610k")
side = side_of(BASE.name)
F = json.loads(BASE.with_suffix(".json").read_text())
NS, c, d = F["n_surface"], np.array(F["cog_mm"]), F["maxd_mm"]
target = to_frame(atlas_surface(week, side, "pial")[0], c, d)
tri = atlas_surface(21, side, "pial")[1]
e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
edges = np.unique(e, axis=0)
nn = int(BASE.with_suffix(".mesh").read_text().split("\n", 1)[0])
X = np.loadtxt(BASE.with_suffix(".mesh").read_text().split("\n")[1:1 + nn])
owner = np.arange(len(X))
owner[NS:] = cKDTree(X[:NS]).query(X[NS:])[1]
cortex = (np.loadtxt(BASE.with_suffix(".gr.txt"))[:NS] > 0.5).astype(float)
theta0 = np.load(BASE.with_suffix(".nodal_a.npy"))[week - 21, :NS]
for name, arr in [("target", target), ("edges", edges.astype(float)), ("faces", tri.astype(float)), ("owner", owner.astype(float)),
                  ("cortex", cortex), ("theta0", theta0)]:
    np.save(out / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float64))
print(out, target.shape, edges.shape, owner.shape, int(cortex.sum()), "theta0 median %.3f" % np.median(theta0))
