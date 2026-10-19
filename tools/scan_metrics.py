#!/usr/bin/env python3
"""Fold metrics of several runs vs the dHCP atlas, from the weekly surface files.

    python3 tools/scan_metrics.py out/scan_metrics "label=out/run_dir" ... [--weeks 22 ... 36]

Per run and week (surface_<(w-21)*8769>.vtk, vertex i = atlas vertex i): gyrification index
FI = A / V^(2/3), correlation with the atlas (cortex only) of the smoothed mean curvature and of
the convex-hull depth (surface centred and scaled by V^(1/3)). Baseline "atlas SG 21": the
week-21 atlas geometry used as a prediction of week N. Writes <out>/scan_metrics.csv and .png.
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
import scipy.sparse as sp
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
ATLAS = next((p for p in [Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas"),
              Path(__file__).resolve().parent.parent / "data/atlas_dhcp"] if p.is_dir()), None)
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("runs", nargs="+", help='"label=run dir"')
ap.add_argument("--weeks", type=int, nargs="+", default=list(range(22, 37)))
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

F = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
NS, c0, d0 = F["n_surface"], np.array(F["cog_mm"]), F["maxd_mm"]
TRI = pv.read(ATLAS / "fetal.week21.left.pial.surf.vtk").extract_surface().faces.reshape(-1, 4)[:, 1:]
FACES = np.column_stack([np.full(len(TRI), 3), TRI]).ravel()
CORTEX = np.loadtxt(ROOT / "data/meshes/w21_left_610k.gr.txt")[:NS] > 0.5
e = np.concatenate([TRI[:, [0, 1]], TRI[:, [1, 2]], TRI[:, [2, 0]]])
A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(NS, NS))
A = ((A + A.T) > 0).astype(float).tocsr()
D = np.asarray(A.sum(1)).ravel()


def atlas(w):
    p = np.asarray(pv.read(ATLAS / f"fetal.week{w}.left.pial.surf.vtk").points, float)
    return np.column_stack([-(p[:, 1] - c0[1]), p[:, 0] - c0[0], -(p[:, 2] - c0[2])]) / d0


def features(p):
    a, b, c = p[TRI[:, 0]], p[TRI[:, 1]], p[TRI[:, 2]]
    cr = np.cross(b - a, c - a)
    V = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    fi = 0.5 * np.linalg.norm(cr, axis=1).sum() / abs(V) ** (2 / 3)
    q = (p - p.mean(0)) / abs(V) ** (1 / 3)
    h = pv.PolyData(q, FACES).curvature("mean") * (1 if V > 0 else -1)
    for _ in range(20):
        h = 0.5 * h + 0.5 * (A @ h) / D
    eq = ConvexHull(q).equations
    return fi, h, np.clip(-(q @ eq[:, :3].T + eq[:, 3]).max(1), 0, None)


def corr(x, y):
    return np.corrcoef(x[CORTEX], y[CORTEX])[0, 1]


runs = [r.rsplit("=", 1) for r in args.runs]
_, h21, d21 = features(atlas(21))
rows = []
for w in args.weeks:
    fa, ha, da = features(atlas(w))
    rows.append(dict(week=w, run="atlas", fi=fa, r_curv=1.0, r_depth=1.0))
    rows.append(dict(week=w, run="atlas SG 21", fi=np.nan, r_curv=corr(h21, ha), r_depth=corr(d21, da)))
    for lab, d in runs:
        f = ROOT / d / f"surface_{(w - 21) * 8769:07d}.vtk"
        if not f.exists():
            continue
        x = np.asarray(pv.read(f).points, float)[:NS]
        if not np.isfinite(x).all():
            continue
        fs, hs, ds = features(x)
        if fs > 2 * fa:  # surface written while the run was blowing up
            continue
        rows.append(dict(week=w, run=lab, fi=fs, r_curv=corr(hs, ha), r_depth=corr(ds, da)))
    print(w, " ".join(f"{r['run']}:{r['fi']:.2f}/{r['r_curv']:.3f}" for r in rows if r["week"] == w), flush=True)

with open(out / "scan_metrics.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
names = ["atlas", "atlas SG 21"] + [lab for lab, _ in runs]
cmap = plt.get_cmap("tab10")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=110)
for ax, (key, title) in zip(axs, [("fi", "(a) Indice de gyrification FI = A/V^(2/3)"),
                                  ("r_curv", "(b) Corrélation de courbure moyenne lissée\navec l'atlas (même SG)"),
                                  ("r_depth", "(c) Corrélation de profondeur (enveloppe convexe)\navec l'atlas")]):
    for i, name in enumerate(names):
        if name == "atlas" and key != "fi":
            continue
        pts = [(r["week"], r[key]) for r in rows if r["run"] == name and np.isfinite(r[key])]
        if not pts:
            continue
        x, y = zip(*pts)
        col = INK if name == "atlas" else "#898781" if name == "atlas SG 21" else cmap(i - 2)
        ax.plot(x, y, "--" if name.startswith("atlas") else "-", color=col, lw=2, marker="o", ms=3.5,
                label=name.replace("\\n", " "))
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel("semaine de gestation (SG)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out / "scan_metrics.png", facecolor="white")
print(out / "scan_metrics.png")
