#!/usr/bin/env python3
"""Parameters of the per-node (local) runs vs gestational week, and maps of a_t.

    python3 tools/nodal_params_plot.py [--out out/atlas_w21_nodal_mut]

params_local.png: (a) a_t per vertex (median, p10-p90 on the cortex) with Tallinen's
1.85 t and the global weekly fit; (b) H per vertex (mm); (c) mu_g / mu_w of each local
run; (d) folding index A/V^(2/3) of each local run vs the atlas; (e) relative error.
params_maps.png: a_t on the week-21 surface at a few weeks (lateral view).
"""
import argparse
import json
import re
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
ATLAS = next((p for p in [Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas"),
              Path(__file__).resolve().parent.parent / "data/atlas_dhcp"] if p.is_dir()), None)
FRAME = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
NS, MM = FRAME["n_surface"], FRAME["maxd_mm"]
FI_ATLAS = [6.18, 6.18, 6.20, 6.23, 6.29, 6.40, 6.60, 6.83, 7.09, 7.43, 7.82, 8.10, 8.34, 8.59, 8.78, 8.92]
RUNS = {  # label -> run dir (config read from <dir>/params.toml)
    "μ 0,86 (Tallinen)": "out/atlas_w21_nodal",
    "μ 1,5": "out/atlas_w21_nodal_mu15",
    "μ 2": "out/atlas_w21_nodal_mu2",
    "μ 2 → 1,33": "out/atlas_w21_nodal_mut",
    "μ 5": "out/atlas_w21_nodal_mu5",
}
COLORS = ["#1baf7a", "#e87ba4", "#2a78d6", "#eda100", "#4a3aa7"]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out/atlas_w21_nodal_mut")
ap.add_argument("--map-weeks", type=int, nargs="+", default=[24, 27, 30, 33, 36])
args = ap.parse_args()
out = ROOT / args.out

a = np.load(ROOT / "data/meshes/w21_left_610k.nodal_a.npy")[:, :NS]
h = np.load(ROOT / "data/meshes/w21_left_610k.nodal_h.npy")[:, :NS] * MM
cortex = np.loadtxt(ROOT / "data/meshes/w21_left_610k.gr.txt")[:NS] > 0.5
sg = np.arange(21, 37)
fit = json.loads((ROOT / "out/atlas_fit/fit.json").read_text())
fit_sg = [21] + [w["week"] for w in fit["weeks"]]
fit_a = [0.0] + [w["a"] for w in fit["weeks"]]

TRI = pv.read(ATLAS / "fetal.week21.left.pial.surf.vtk").extract_surface().faces.reshape(-1, 4)[:, 1:]


def fi(vtk):
    L = Path(vtk).read_text().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    p = np.loadtxt(L[i + 1:i + 1 + NS])
    x, y, z = p[TRI[:, 0]], p[TRI[:, 1]], p[TRI[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(y - x, z - x), axis=1).sum()
    V = abs(np.einsum("ij,ij->i", x, np.cross(y, z)).sum() / 6)
    return A / V ** (2 / 3)


def mu_ratio(run):
    prm = tomllib.loads((ROOT / run / "params.toml").read_text())
    mg, mw = prm["material"]["mu_gray"], prm["material"]["mu_white"]
    s = prm["schedule"]
    mgs = np.interp(sg, 21 + 15 * np.array(s["t"]), s["mu_gray"]) if "mu_gray" in s else np.full(16, mg)
    mws = np.interp(sg, 21 + 15 * np.array(s["t"]), s["mu_white"]) if "mu_white" in s else np.full(16, mw)
    return mgs / mws


def run_fi(run):
    res = {21: 6.176}
    for f in (ROOT / run).glob("surface_[0-9]*.vtk"):
        step = int(f.stem.split("_")[1])
        if step > 0 and step % 8769 == 0:
            res[21 + step // 8769] = fi(f)
    final = ROOT / run / "surface_final.vtk"
    log = (ROOT / (run + ".log")).read_text() if (ROOT / (run + ".log")).exists() else ""
    if final.exists() and "t = 1.0000" in log:
        res[36] = fi(final)
    return dict(sorted(res.items()))


INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 3, figsize=(16, 8.4), dpi=110)
axs = axs.ravel()


def frame(ax, title, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xlabel("semaine de gestation (SG)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_xticks(range(21, 37, 2))


ax = axs[0]
frame(ax, "(a) Expansion tangentielle a_t par sommet (atlas)", "a_t")
q = np.percentile(a[:, cortex], [10, 50, 90], axis=1)
ax.fill_between(sg, q[0], q[2], color="#2a78d6", alpha=0.18, lw=0, label="sommets, p10–p90")
ax.plot(sg, q[1], color="#2a78d6", lw=2, marker="o", ms=4, label="médiane par sommet (imposée)")
ax.plot(sg, 1.85 * (sg - 21) / 15, "--", color="#eb6834", lw=1.5, label="Tallinen 1,85·t")
ax.plot(fit_sg, fit_a, color="#898781", lw=1.5, marker="s", ms=3, label="ajustement global (SG 22–32)")
ax.legend(frameon=False, fontsize=8)

ax = axs[1]
frame(ax, "(b) Épaisseur H par sommet (atlas, piale–blanche)", "H (mm)")
q = np.percentile(h[:, cortex], [10, 50, 90], axis=1)
ax.fill_between(sg, q[0], q[2], color="#1baf7a", alpha=0.18, lw=0, label="sommets, p10–p90")
ax.plot(sg, q[1], color="#1baf7a", lw=2, marker="o", ms=4, label="médiane")
ax.legend(frameon=False, fontsize=8)

ax = axs[2]
frame(ax, "(c) Rapport des raideurs μ_g / μ_w", "μ_g / μ_w")
for (lab, run), col in zip(RUNS.items(), COLORS):
    ax.plot(sg, mu_ratio(run), color=col, lw=2, label=lab)
ax.axhline(1, color=INK2, lw=0.8)
ax.legend(frameon=False, fontsize=8)

fis = {lab: run_fi(run) for lab, run in RUNS.items()}
ax = axs[3]
frame(ax, "(d) Indice de plissement A/V^(2/3)", "FI")
ax.plot(sg, FI_ATLAS, "--", color=INK, lw=1.8, label="atlas")
for (lab, f), col in zip(fis.items(), COLORS):
    ax.plot(list(f), list(f.values()), color=col, lw=2, marker="o", ms=3, label=lab)
ax.legend(frameon=False, fontsize=8)

ax = axs[4]
frame(ax, "(e) Erreur relative sur FI", "(FI sim − FI atlas) / FI atlas (%)")
ax.axhline(0, color=INK2, lw=0.8)
for (lab, f), col in zip(fis.items(), COLORS):
    w = np.array(list(f))
    ax.plot(w, 100 * (np.array(list(f.values())) - np.array(FI_ATLAS)[w - 21]) / np.array(FI_ATLAS)[w - 21],
            color=col, lw=2, marker="o", ms=3, label=lab)
ax.set_ylim(-8, 12)

ax = axs[5]
ax.axis("off")
last = {lab: max(f) for lab, f in fis.items()}
ax.text(0, 1, "\n".join([
    "Modèles à paramètres locaux : a_t et H mesurés par sommet",
    "sur l'atlas dHCP (aire locale, distance piale–blanche,",
    "lissage ≈ 4–5 mm), sans ajustement de la croissance.",
    "Seul le rapport μ_g/μ_w diffère (panneau c).",
    "",
    "Dernière semaine atteinte (NaN ensuite) :",
    *[f"  {lab} : SG {w}" for lab, w in last.items()],
    "",
    "Pas de temps 7,6e-6, contact vérifié à chaque pas,",
    "k_inverted = 0,5, maillage atlas SG 21 (610k tétraèdres).",
]), va="top", fontsize=9, color=INK2)
fig.tight_layout()
fig.savefig(out / "params_local.png", facecolor="white")
print(out / "params_local.png")

# ---- maps of a_t on the week-21 surface --------------------------------------------------
X = np.load(ROOT / "out/atlas_fit/mesh_cache.npz")["X0"][:NS]
surf = pv.PolyData(X, np.column_stack([np.full(len(TRI), 3), TRI]).ravel())
n = len(args.map_weeks)
pl = pv.Plotter(shape=(1, n), off_screen=True, window_size=(420 * n, 380), border=False)
vmax = float(np.percentile(a[args.map_weeks[-1] - 21][cortex], 98))
for j, w in enumerate(args.map_weeks):
    pl.subplot(0, j)
    m = surf.copy()
    m["a"] = np.where(cortex, a[w - 21], np.nan)
    pl.add_mesh(m, scalars="a", cmap="viridis", clim=(0, vmax), nan_color="#d9d6cf", smooth_shading=True,
                show_scalar_bar=False)
    pl.add_text(f"SG {w}", font_size=10, color="black")
pl.subplot(0, 0)
pl.view_xz()
pl.camera.up = (0.0, 0.0, -1.0)
pl.camera.zoom(1.35)
cam = pl.camera.copy()
for j in range(n):
    pl.subplot(0, j)
    pl.camera = cam.copy()
pl.set_background("white")
img = pl.screenshot(return_img=True)
pl.close()
fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100 + 0.9), dpi=100)
axi = fig.add_axes([0, 0.18, 0.92, 0.8])
axi.imshow(img)
axi.axis("off")
cax = fig.add_axes([0.935, 0.3, 0.012, 0.55])
from matplotlib import cm, colors

fig.colorbar(cm.ScalarMappable(colors.Normalize(0, vmax), "viridis"), cax=cax).set_label("a_t", color=INK2)
fig.text(0.01, 0.03, "a_t imposé par sommet = √(aire locale(SG) / aire locale(SG 21)) − 1, mesuré sur l'atlas "
         "(lissage ≈ 4–5 mm), affiché sur la surface de SG 21 ; gris = paroi médiale sans croissance.",
         fontsize=9, color=INK2)
fig.savefig(out / "params_maps.png", facecolor="white")
print(out / "params_maps.png")
