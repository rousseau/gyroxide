#!/usr/bin/env python3
"""Report of an adjoint run (examples/adjoint_week.rs): fold placement vs iteration and growth maps.

    python3 tools/adjoint_report.py out/adjoint_w26/b1e-4 26

Prints, per saved iteration, the correlation of smoothed mean curvature and of convex-hull
depth with the atlas of the week (cortex only), next to the baselines "atlas SG 21" and the
starting equilibrium (measured growth); renders theta_0, theta_last and theta_last - theta_0
on the week-21 surface (lateral view) into <run>/maps.png.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
ATLAS = Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas")
run, week = Path(sys.argv[1]), int(sys.argv[2])
prep = run.parent
F = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
NS, c0, d0 = F["n_surface"], np.array(F["cog_mm"]), F["maxd_mm"]
TRI = pv.read(ATLAS / "fetal.week21.left.pial.surf.vtk").extract_surface().faces.reshape(-1, 4)[:, 1:]
FACES = np.column_stack([np.full(len(TRI), 3), TRI]).ravel()
cortex = np.load(prep / "cortex.npy") > 0.5
e = np.concatenate([TRI[:, [0, 1]], TRI[:, [1, 2]], TRI[:, [2, 0]]])
A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(NS, NS))
A = ((A + A.T) > 0).astype(float).tocsr()
D = np.asarray(A.sum(1)).ravel()


def atlas(w):
    p = np.asarray(pv.read(ATLAS / f"fetal.week{w}.left.pial.surf.vtk").points, float)
    return np.column_stack([-(p[:, 1] - c0[1]), p[:, 0] - c0[0], -(p[:, 2] - c0[2])]) / d0


def curvature(p):
    a, b, c = p[TRI[:, 0]], p[TRI[:, 1]], p[TRI[:, 2]]
    Vs = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    q = (p - p.mean(0)) / abs(Vs) ** (1 / 3)
    h = pv.PolyData(q, FACES).curvature("mean") * (1 if Vs > 0 else -1)
    for _ in range(20):
        h = 0.5 * h + 0.5 * (A @ h) / D
    eq = ConvexHull(q).equations
    depth = np.clip(-(q @ eq[:, :3].T + eq[:, 3]).max(1), 0, None)
    return h, depth


def corr(a, b):
    return np.corrcoef(a[cortex], b[cortex])[0, 1]


ha, da = curvature(atlas(week))
h21, d21 = curvature(atlas(21))
print(f"baseline atlas SG 21: r_curv {corr(h21, ha):.3f}  r_depth {corr(d21, da):.3f}")
its = sorted(int(f.stem.split("_")[1]) for f in run.glob("x_[0-9]*.npy"))
for k in its:
    x = np.load(run / f"x_{k}.npy")[:NS]
    h, dd = curvature(x)
    tag = "start (measured growth)" if k == 0 else f"iteration {k}"
    print(f"{tag:24s}: r_curv {corr(h, ha):.3f}  r_depth {corr(dd, da):.3f}")

theta0 = np.load(prep / "theta0.npy")
last = max((int(f.stem.split("_")[1]) for f in run.glob("theta_[0-9]*.npy")), default=None)
if last is None:
    sys.exit()
theta = np.load(run / f"theta_{last}.npy")
dth = theta - theta0
print(f"theta change: rms {np.sqrt((dth[cortex] ** 2).mean()):.4f}, p5 {np.percentile(dth[cortex], 5):.4f}, "
      f"p95 {np.percentile(dth[cortex], 95):.4f}; relative rms {np.sqrt((dth[cortex] ** 2).mean()) / np.sqrt((theta0[cortex] ** 2).mean()):.2f}")

X = np.load(ROOT / "out/atlas_fit/mesh_cache.npz")["X0"][:NS]
surf = pv.PolyData(X, FACES)
vmax = float(np.percentile(np.abs(theta0[cortex]), 98))
dmax = float(np.percentile(np.abs(dth[cortex]), 98))
panels = [("a mesuré (atlas)", theta0, "viridis", (0, vmax)), (f"a optimisé (it. {last})", theta, "viridis", (0, vmax)),
          ("différence", dth, "RdBu_r", (-dmax, dmax))]
pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1500, 420), border=False)
for j, (lab, v, cmap, clim) in enumerate(panels):
    pl.subplot(0, j)
    m = surf.copy()
    m["v"] = np.where(cortex, v, np.nan)
    pl.add_mesh(m, scalars="v", cmap=cmap, clim=clim, nan_color="#d9d6cf", smooth_shading=True,
                scalar_bar_args={"title": lab, "color": "black", "vertical": False, "position_x": 0.2, "width": 0.6})
    pl.view_xz()
    pl.camera.up = (0.0, 0.0, -1.0)
    pl.camera.zoom(1.3)
pl.set_background("white")
pl.screenshot(run / "maps.png")
print(run / "maps.png")
