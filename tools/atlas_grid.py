#!/usr/bin/env python3
"""Atlas (row 1) vs simulation (row 2), one column per computed week, same camera.

    python3 tools/atlas_grid.py out/atlas_fit [out/atlas_fit/grid.png]

Week 21 is the initial mesh surface; weeks 22.. are <fit>/wNN/surface_final.vtk
for the weeks kept in <fit>/fit.json. Both rows use the atlas triangulation (the
first 32,492 mesh nodes are the atlas vertices). The atlas is mapped into the
simulation frame (data/meshes/w21_left_610k.json); every surface is then centred
and divided by its own V^(1/3) so shapes compare despite the growth in size.
Colour: mean curvature H of the scaled surface, one shared scale; H > 0 on convex
parts (gyral crowns), H < 0 in concave parts (sulcal fundi), checked on a sphere.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
ROOT = Path(__file__).resolve().parent.parent
ATLAS = Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas")
FRAME = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
NS = FRAME["n_surface"]
fit = Path(sys.argv[1])
png = sys.argv[2] if len(sys.argv) > 2 else str(fit / "grid.png")
state = json.loads((fit / "fit.json").read_text()) if (fit / "fit.json").exists() else None


def read_points(vtk):
    L = Path(vtk).read_text().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    return np.loadtxt(L[i + 1:i + 1 + NS])


def atlas_points(week):
    p = np.asarray(pv.read(ATLAS / f"fetal.week{week}.left.pial.surf.vtk").points, float)
    c, d = np.array(FRAME["cog_mm"]), FRAME["maxd_mm"]
    return np.column_stack([-(p[:, 1] - c[1]), p[:, 0] - c[0], -(p[:, 2] - c[2])]) / d


TRI = pv.read(ATLAS / "fetal.week21.left.pial.surf.vtk").extract_surface().faces.reshape(-1, 4)[:, 1:]
FACES = np.column_stack([np.full(len(TRI), 3), TRI]).ravel()


def surface(points):
    """Atlas-topology surface, its folding index, then centred and scaled to unit volume."""
    a, b, c = points[TRI[:, 0]], points[TRI[:, 1]], points[TRI[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    Vs = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    V = abs(Vs)
    q = (points - points.mean(0)) / V ** (1 / 3)
    m = pv.PolyData(q, FACES)
    # The curvature sign follows the triangle winding: orient it so H > 0 on convex parts.
    m["H"] = m.curvature("mean") * (1.0 if Vs > 0 else -1.0)
    return m, A / V ** (2 / 3)


weeks = [(21, ROOT / "out/atlas_w21_tab/surface_0000000.vtk")]
if state:  # tools/fit_weekly.py output
    weeks += [(w["week"], fit / f"w{w['week']}" / "surface_final.vtk") for w in state["weeks"]]
else:  # a plain run with --vtk-every 8769 (one surface per week)
    for f in sorted(fit.glob("surface_[0-9]*.vtk")):
        step = int(f.stem.split("_")[1])
        if step > 0 and step % 8769 == 0:
            weeks.append((21 + step // 8769, f))
    if weeks[-1][0] == 35 and (fit / "surface_final.vtk").exists():  # ran to t = 1
        weeks.append((36, fit / "surface_final.vtk"))
panels = [[surface(atlas_points(w)) for w, _ in weeks], [surface(read_points(f)) for _, f in weeks]]
lim = np.percentile(np.abs(np.concatenate([m["H"] for row in panels for m, _ in row])), 95)

n = len(weeks)
w_px = max(2000 // n, 220)
pl = pv.Plotter(shape=(2, n), off_screen=True, window_size=(w_px * n, int(w_px * 2 * 0.85) + 60), border=False)
for i, row in enumerate(panels):
    for j, (m, fi) in enumerate(row):
        pl.subplot(i, j)
        pl.add_mesh(m, scalars="H", cmap="RdBu_r", clim=(-lim, lim), smooth_shading=True,
                    show_scalar_bar=False, specular=0.2, ambient=0.2)
        title = f"SG {weeks[j][0]}\n" if i == 0 else ""
        who = "atlas" if i == 0 else "simulation"
        pl.add_text(f"{title}{who}  FI {fi:.2f}", font_size=9, color="black", position="upper_left")
# One camera for all panels: lateral view of the left hemisphere (as the earlier renders).
pl.subplot(0, 0)
pl.view_xz()
pl.camera.up = (0.0, 0.0, -1.0)
pl.camera.zoom(1.35)
cam = pl.camera.copy()
for i in range(2):
    for j in range(n):
        pl.subplot(i, j)
        pl.camera = cam.copy()
pl.set_background("white")
img = pl.screenshot(return_img=True)
pl.close()

# Assemble with matplotlib: row labels, colour bar with its legend, caption.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

h_px, w_img = img.shape[:2]
dpi = 100
fig = plt.figure(figsize=((w_img + 380) / dpi, (h_px + 120) / dpi), dpi=dpi)
ax = fig.add_axes([150 / (w_img + 380), 90 / (h_px + 120), w_img / (w_img + 380), h_px / (h_px + 120)])
ax.imshow(img)
ax.axis("off")
for k, label in enumerate(["Atlas dHCP", "Simulation"]):
    y = (90 + h_px * (0.75 - 0.5 * k)) / (h_px + 120)
    fig.text(0.012, y, label, rotation=90, va="center", ha="left", fontsize=13, color="#0b0b0b")
cax = fig.add_axes([(150 + w_img + 40) / (w_img + 380), 0.3, 18 / (w_img + 380), 0.55])
cb = fig.colorbar(cm.ScalarMappable(colors.Normalize(-lim, lim), "RdBu_r"), cax=cax)
cb.set_ticks([-lim, 0, lim])
cb.set_ticklabels([f"{-lim:.2f}\nfond de sillon", "0", f"+{lim:.2f}\ncrête de gyrus"], fontsize=8, color="#52514e")
cb.outline.set_visible(False)
fig.text(
    0.012,
    0.03,
    f"Semaines {weeks[0][0]}–{weeks[-1][0]} : atlas dHCP (haut) vs simulation (bas), vue latérale, hémisphère gauche. "
    "Chaque surface est centrée et divisée par V^(1/3) pour comparer les formes malgré la croissance.\n"
    f"Couleur : courbure moyenne H de la surface normalisée (colormap RdBu_r, échelle commune ±{lim:.2f}, 95e percentile de |H|) ; "
    "H > 0 (rouge) : convexe, crête de gyrus ; H < 0 (bleu) : concave, fond de sillon. FI = A / V^(2/3).",
    fontsize=9.5,
    color="#52514e",
    va="bottom",
)
fig.savefig(png, dpi=dpi, facecolor="white")
# Dated copy per refresh, so earlier versions stay available.
last = weeks[-1][0]
Path(png).with_name(f"grid_w{last}.png").write_bytes(Path(png).read_bytes())
print(png)
