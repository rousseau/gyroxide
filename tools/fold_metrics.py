#!/usr/bin/env python3
"""Fold placement and cortical thickness metrics, simulation vs dHCP atlas.

    python3 tools/fold_metrics.py [--out out/metrics]

For every run and week with a checkpoint (vertex i of the mesh surface = atlas vertex i):
  depth   distance of each vertex to the convex hull of its own surface, after
          scaling the surface by V^(1/3) (sulcal depth proxy, same for atlas and sim)
  r_depth Pearson correlation of depth with the atlas at the same week (cortex only)
  dice    overlap of the 30 % deepest vertices ("sulcal fate") with the atlas
  thick   cortical thickness in the deformed configuration: distance between the
          surface vertex and the material point that sat at depth H_21 below it at
          rest (H_21 = atlas pial-white distance at SG 21); atlas: pial-white distance
  ratio   median thickness on crowns (30 % shallowest) / fundi (30 % deepest)
Baseline "atlas SG 21": the week-21 atlas geometry used as a prediction of week N.
Writes <out>/fold_metrics.csv and <out>/fold_metrics.png.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parent.parent
ATLAS = next((p for p in [Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas"),
              Path(__file__).resolve().parent.parent / "data/atlas_dhcp"] if p.is_dir()), None)
FRAME = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
NS, MM = FRAME["n_surface"], FRAME["maxd_mm"]
RUNS = {
    "ajusté global": lambda w: ROOT / f"out/atlas_fit/w{w}/checkpoint",
    "locaux μ 0,86": lambda w: ROOT / f"out/atlas_w21_nodal/checkpoint_w{w}",
    "locaux μ 2": lambda w: ROOT / f"out/atlas_w21_nodal_mu2/checkpoint_w{w}",
    "locaux μ 5": lambda w: ROOT / f"out/atlas_w21_nodal_mu5/checkpoint_w{w}",
    "locaux μ 1,5": lambda w: ROOT / f"out/atlas_w21_nodal_mu15/checkpoint_w{w}",
    "locaux μ 2→1,33": lambda w: ROOT / f"out/atlas_w21_nodal_mut/checkpoint_w{w}",
}
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out/metrics")
ap.add_argument("--weeks", type=int, nargs="+", default=list(range(23, 37)))
args = ap.parse_args()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)

TRI = pv.read(ATLAS / "fetal.week21.left.pial.surf.vtk").extract_surface().faces.reshape(-1, 4)[:, 1:]
CORTEX = np.loadtxt(ROOT / "data/meshes/w21_left_610k.gr.txt")[:NS] > 0.5
X0 = np.load(ROOT / "out/atlas_fit/mesh_cache.npz")["X0"]


def frame(p):
    c = np.array(FRAME["cog_mm"])
    return np.column_stack([-(p[:, 1] - c[1]), p[:, 0] - c[0], -(p[:, 2] - c[2])]) / MM


def atlas(week, kind="pial"):
    return frame(np.asarray(pv.read(ATLAS / f"fetal.week{week}.left.{kind}.surf.vtk").points, float))


def depth(p):
    a, b, c = p[TRI[:, 0]], p[TRI[:, 1]], p[TRI[:, 2]]
    V = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
    q = p / V ** (1 / 3)
    eq = ConvexHull(q).equations
    return np.clip(-(q @ eq[:, :3].T + eq[:, 3]).max(1), 0, None)


def vertex_normals(p):
    fn = np.cross(p[TRI[:, 1]] - p[TRI[:, 0]], p[TRI[:, 2]] - p[TRI[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, TRI[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    c = p.mean(0)
    return n if ((p - c) * n).sum(1).mean() > 0 else -n  # outward


# Material points at depth H_21 below each surface vertex (rest configuration).
h21 = np.linalg.norm(atlas(21) - atlas(21, "wm"), axis=1)  # already normalised
inner = cKDTree(X0).query(X0[:NS] - vertex_normals(X0[:NS]) * h21[:, None])[1]


def crown_fundus_ratio(th, d):
    lo, hi = np.percentile(d[CORTEX], [30, 70])
    return np.median(th[CORTEX & (d <= lo)]) / np.median(th[CORTEX & (d >= hi)])


def dice(d, d_ref):
    a = d[CORTEX] >= np.percentile(d[CORTEX], 70)
    b = d_ref[CORTEX] >= np.percentile(d_ref[CORTEX], 70)
    return 2 * (a & b).sum() / (a.sum() + b.sum())


d21 = depth(atlas(21))
rows = []
for w in args.weeks:
    pa = atlas(w)
    da = depth(pa)
    tha = np.linalg.norm(pa - atlas(w, "wm"), axis=1) * MM
    rows.append(dict(week=w, run="atlas", r_depth=1.0, dice=1.0, thick_med=np.median(tha[CORTEX]),
                     ratio=crown_fundus_ratio(tha, da)))
    rows.append(dict(week=w, run="atlas SG 21", r_depth=np.corrcoef(d21[CORTEX], da[CORTEX])[0, 1],
                     dice=dice(d21, da), thick_med=np.nan, ratio=np.nan))
    for name, ck in RUNS.items():
        f = ck(w) / "x.npy"
        if not f.exists():
            continue
        x = np.load(f)
        ds = depth(x[:NS])
        ths = np.linalg.norm(x[:NS] - x[inner], axis=1) * MM
        rows.append(dict(week=w, run=name, r_depth=np.corrcoef(ds[CORTEX], da[CORTEX])[0, 1], dice=dice(ds, da),
                         thick_med=np.median(ths[CORTEX]), ratio=crown_fundus_ratio(ths, ds)))
    print(w, "done", flush=True)

with open(out / "fold_metrics.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)

# ---- figure -----------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"atlas": "#0b0b0b", "atlas SG 21": "#898781", "ajusté global": "#eb6834", "locaux μ 0,86": "#1baf7a",
          "locaux μ 2": "#2a78d6", "locaux μ 5": "#4a3aa7", "locaux μ 1,5": "#e87ba4", "locaux μ 2→1,33": "#eda100"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), dpi=110)
panels = [("r_depth", "(a) Corrélation de profondeur\navec l'atlas (même SG)", None),
          ("dice", "(b) Dice des 30 % les plus profonds\n(destin sillon) avec l'atlas", None),
          ("thick_med", "(c) Épaisseur corticale médiane (mm)\n(configuration déformée)", None),
          ("ratio", "(d) Épaisseur crêtes / fonds\n(30 % moins / plus profonds)", None)]
for ax, (key, title, _) in zip(axs, panels):
    for name, col in COLORS.items():
        pts = [(r["week"], r[key]) for r in rows if r["run"] == name and np.isfinite(r[key])]
        if key in ("r_depth", "dice") and name == "atlas":
            continue
        if not pts:
            continue
        x, y = zip(*pts)
        ls = "--" if name.startswith("atlas") else "-"
        ax.plot(x, y, ls, color=col, lw=2, marker="o", ms=4, label=name)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel("semaine de gestation (SG)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
axs[0].legend(frameon=False, fontsize=8)
axs[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out / "fold_metrics.png", facecolor="white")
print(out / "fold_metrics.png")
