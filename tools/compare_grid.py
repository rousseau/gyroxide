#!/usr/bin/env python3
"""One row per method, one column per week: dHCP atlas, Tallinen, global fit, local parameters.

    python3 tools/compare_grid.py [out/compare_grid.png] [--weeks 21 ... 32]

Same rendering as tools/atlas_grid.py (atlas triangulation, each surface centred and
divided by V^(1/3), mean curvature with one shared scale, lateral view).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("png", nargs="?", default=str(ROOT / "out/compare_grid.png"))
ap.add_argument("--weeks", type=int, nargs="+", default=list(range(21, 33)))
ap.add_argument("--rows", nargs="*", help='rows as "label=run dir" (one surface per week, step = 8769 k); default: the 4 methods')
ap.add_argument("--caption", default=None)
args = ap.parse_args()

# Reuse the surface helpers of atlas_grid.py (everything before its figure code).
sys.argv = ["atlas_grid.py", str(ROOT / "out/atlas_fit")]
src = (ROOT / "tools/atlas_grid.py").read_text()
exec(src[: src.index("weeks = [")].replace("Path(__file__)", f"Path({str(ROOT / 'tools/atlas_grid.py')!r})"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib import cm, colors

S0 = ROOT / "out/atlas_w21_tab/surface_0000000.vtk"


def step_vtk(run, w):
    return ROOT / run / f"surface_{(w - 21) * 8769:07d}.vtk"


ROWS = [
    ("Atlas dHCP", lambda w: ("atlas", w)),
    ("Tallinen\na_t = 1,85·t", lambda w: step_vtk("out/atlas_w21_tab", w) if w <= 25 else None),
    ("Tallinen ajusté\nparamètres globaux", lambda w: S0 if w == 21 else ROOT / f"out/atlas_fit/w{w}/surface_final.vtk"),
    ("Tallinen\nparamètres locaux", lambda w: step_vtk("out/atlas_w21_nodal_mut", w)),
]
REASON = {1: "arrêt numérique\n(NaN) vers SG 25,4", 3: "arrêt numérique\nvers SG 32,3"}
if args.rows:
    ROWS = [("Atlas dHCP", lambda w: ("atlas", w))] + [
        (lab.replace("\\n", "\n"), (lambda d: (lambda w: step_vtk(d, w)))(d)) for lab, d in (r.rsplit("=", 1) for r in args.rows)]
    REASON = {}

panels = []
for i, (_, src_fn) in enumerate(ROWS):
    row = []
    for w in args.weeks:
        s = src_fn(w)
        if s == ("atlas", w):
            row.append(surface(atlas_points(w)))
        elif s is not None and Path(s).exists():
            row.append(surface(read_points(s)))
        else:
            row.append(None)
    panels.append(row)
lim = np.percentile(np.abs(np.concatenate([m["H"] for r in panels for p in r if p for m in [p[0]]])), 95)

n, nr = len(args.weeks), len(ROWS)
w_px = 230
pl = pv.Plotter(shape=(nr, n), off_screen=True, window_size=(w_px * n, int(w_px * 0.8) * nr), border=False)
for i, row in enumerate(panels):
    for j, p in enumerate(row):
        pl.subplot(i, j)
        if p is None:
            continue
        m, fi = p
        pl.add_mesh(m, scalars="H", cmap="RdBu_r", clim=(-lim, lim), smooth_shading=True, show_scalar_bar=False,
                    specular=0.2, ambient=0.2)
        pl.add_text(f"FI {fi:.2f}", font_size=8, color="black", position="upper_left")
pl.subplot(0, 0)
pl.view_xz()
pl.camera.up = (0.0, 0.0, -1.0)
pl.camera.zoom(1.45)
cam = pl.camera.copy()
for i in range(nr):
    for j in range(n):
        pl.subplot(i, j)
        pl.camera = cam.copy()
pl.set_background("white")
img = pl.screenshot(return_img=True)
pl.close()

H_, W_ = img.shape[:2]
L, R, T, B = 190, 150, 50, 70
fig = plt.figure(figsize=((W_ + L + R) / 100, (H_ + T + B) / 100), dpi=100)
ax = fig.add_axes([L / (W_ + L + R), B / (H_ + T + B), W_ / (W_ + L + R), H_ / (H_ + T + B)])
ax.imshow(img)
ax.set_xlim(0, W_)
ax.set_ylim(H_, 0)
ax.axis("off")
cw, ch = W_ / n, H_ / nr
for j, w in enumerate(args.weeks):
    ax.text((j + 0.5) * cw, -12, f"SG {w}", ha="center", va="bottom", fontsize=12, color="#0b0b0b")
for i, (name, _) in enumerate(ROWS):
    ax.text(-14, (i + 0.5) * ch, name, ha="right", va="center", fontsize=11, color="#0b0b0b")
    for j, p in enumerate(panels[i]):
        if p is None:
            ax.text((j + 0.5) * cw, (i + 0.5) * ch, REASON.get(i, "—") if j == next(
                k for k, q in enumerate(panels[i]) if q is None) else "", ha="center", va="center", fontsize=8.5,
                    color="#52514e")
cax = fig.add_axes([(L + W_ + 40) / (W_ + L + R), 0.25, 14 / (W_ + L + R), 0.5])
cb = fig.colorbar(cm.ScalarMappable(colors.Normalize(-lim, lim), "RdBu_r"), cax=cax)
cb.set_ticks([-lim, 0, lim])
cb.set_ticklabels(["fond de\nsillon", "0", "crête de\ngyrus"], fontsize=8, color="#52514e")
cb.outline.set_visible(False)
fig.text(0.01, 0.012, args.caption or "Courbure moyenne de la surface normalisée par V^(1/3) (échelle commune), vue latérale, hémisphère "
         "gauche. FI = A/V^(2/3). Tallinen : a_t = 1,85·t, H atlas. Ajusté global : a_t choisi chaque semaine "
         "sur le FI. Locaux : a_t et H mesurés par sommet sur l'atlas, μg/μw 2 → 1,33.", fontsize=8.5, color="#52514e")
fig.savefig(args.png, facecolor="white")
print(args.png)
