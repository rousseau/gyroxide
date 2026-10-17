#!/usr/bin/env python3
"""Coronal slices: atlas (pial / white ribbon) vs simulation checkpoints, with volume fields.

    python3 tools/coronal_slices.py --out out/slices

Simulation fields, per tetrahedron, from a checkpoint (x, growth_prev) and the rest mesh:
  g_m      cortical indicator gr / (1 + exp(10 (d / H - 1)))   (1 = growing cortex)
  J_e      elastic volume ratio det(F G^-1)                   (< 1: compressed)
  lam_min  smallest principal elastic stretch                 (< 1: compressed in some direction)
  sig_vm   von Mises Cauchy stress / mu_gray                   (Tallinen's neo-Hookean, K = 5)
  disp     displacement |x - X| in mm
with G = I + (I - n n^T) growth_prev, n the reference growth normal of the tet.
The plane is coronal (constant anterior-posterior coordinate x' of Tallinen's frame)
through each surface's own centroid, so the same anatomy is cut at every age.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
ATLAS = Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas")
FRAME = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
MM = FRAME["maxd_mm"]
NS = FRAME["n_surface"]
H_MM = [1.45, 1.45, 1.45, 1.47, 1.50, 1.53, 1.50, 1.45, 1.36, 1.29, 1.18, 1.12, 1.07, 1.04, 1.03, 1.03]
MU_G, MU_W, K = 1.0, 1.167, 5.0

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out/slices")
ap.add_argument("--weeks", type=int, nargs="+", default=[22, 24, 25, 27, 29, 32])
ap.add_argument("--field-weeks", type=int, nargs="+", default=[25, 29])
ap.add_argument("--offset", type=float, default=0.0, help="plane offset from the centroid, in units of the AP half-extent")
args = ap.parse_args()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)

# Models: name -> {week: checkpoint dir}
MODELS = {
    "Tallinen\n(a_t = 1,85·t)": {w: ROOT / f"out/atlas_w21_tab/checkpoint_w{w}" for w in range(22, 26)},
    "Ajusté global\n(a_t par semaine)": {w: ROOT / f"out/atlas_fit/w{w}/checkpoint" for w in range(22, 37)},
    "Locaux, μg/μw = 0,86\n(a_t, H par sommet)": {w: ROOT / f"out/atlas_w21_nodal/checkpoint_w{w}" for w in range(22, 37)},
    "Locaux, μg/μw = 2": {w: ROOT / f"out/atlas_w21_nodal_mu2/checkpoint_w{w}" for w in range(22, 37)},
    "Locaux, μg/μw = 5": {w: ROOT / f"out/atlas_w21_nodal_mu5/checkpoint_w{w}" for w in range(22, 37)},
    "Locaux, μg/μw 2 → 1,33\n(variable dans le temps)": {w: ROOT / f"out/atlas_w21_nodal_mut/checkpoint_w{w}" for w in range(22, 37)},
}

# ---- rest mesh and per-tet reference quantities -------------------------------------
cache = ROOT / "out/atlas_fit/mesh_cache.npz"
z = np.load(cache)
X0, T = z["X0"], z["T"]
lines = (ROOT / "data/meshes/w21_left_610k.mesh").read_text().split("\n")
n, m = len(X0), len(T)
k = 2 + n + m
F_s = np.loadtxt(lines[k + 1:k + 1 + int(lines[k])], dtype=np.int64)[:, 1:] - 1
surf = pv.PolyData(X0[:NS], np.column_stack([np.full(len(F_s), 3), F_s]).ravel())
N0 = surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True).point_data["Normals"]
dist, near = cKDTree(X0[:NS]).query(X0)
gr = np.loadtxt(ROOT / "data/meshes/w21_left_610k.gr.txt")
n_t = N0[near[T]].sum(1)
n_t /= np.linalg.norm(n_t, axis=1, keepdims=True)
d_t = dist[T].mean(1)
gr_t = gr[T].mean(1)
E0 = np.stack([X0[T[:, i]] - X0[T[:, 0]] for i in (1, 2, 3)], axis=2)  # columns = rest edges


def fields(ckpt, week):
    import tomllib

    prm = ckpt.parent / "params.toml"
    mat = tomllib.loads(prm.read_text())["material"] if prm.exists() else {}
    mu_g, mu_w = mat.get("mu_gray", MU_G), mat.get("mu_white", MU_W)
    x = np.load(ckpt / "x.npy")
    g = np.load(ckpt / "growth_prev.npy")
    t = (week - 21) / 15
    H = np.interp(t, np.arange(16) / 15, np.array(H_MM) / MM)
    gm = gr_t / (1 + np.exp(10 * (d_t / H - 1)))
    P = np.eye(3) - n_t[:, :, None] * n_t[:, None, :]
    G = np.eye(3) + P * g[:, None, None]
    E = np.stack([x[T[:, i]] - x[T[:, 0]] for i in (1, 2, 3)], axis=2)
    Fe = E @ np.linalg.inv(G @ E0)
    J = np.linalg.det(Fe)
    C = np.transpose(Fe, (0, 2, 1)) @ Fe
    lam = np.sqrt(np.clip(np.linalg.eigvalsh(C)[:, 0], 0, None))
    B = Fe @ np.transpose(Fe, (0, 2, 1))
    mu = mu_w * (1 - gm) + mu_g * gm
    Jc = np.clip(J, 1e-3, None)
    dev = B - np.trace(B, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    s = (mu * Jc ** (-5 / 3))[:, None, None] * dev
    vm = np.sqrt(1.5 * (s * s).sum((1, 2))) / mu_g
    disp = np.linalg.norm(x - X0, axis=1)[T].mean(1) * MM
    return x, {"g_m": gm, "J_e": J, "lam_min": lam, "sig_vm": vm, "disp": disp}


def plane_x(points):
    c = points.mean(0)
    half = 0.5 * np.ptp(points[:, 0])
    return c[0] + args.offset * half


def sim_slice(x, cell_fields, x_cut):
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: T}, x)
    for k_, v in cell_fields.items():
        grid.cell_data[k_] = v
    s = grid.slice(normal="x", origin=(x_cut, 0, 0))
    return s


def atlas_curves(week, x_cut_fn):
    out_ = {}
    c, d = np.array(FRAME["cog_mm"]), MM
    for kind in ("pial", "wm"):
        mm = pv.read(ATLAS / f"fetal.week{week}.left.{kind}.surf.vtk").extract_surface()
        p = np.asarray(mm.points, float)
        mm.points = np.column_stack([-(p[:, 1] - c[1]), p[:, 0] - c[0], -(p[:, 2] - c[2])]) / d
        out_[kind] = mm
    x_cut = x_cut_fn(np.asarray(out_["pial"].points))
    return {k_: v.slice(normal="x", origin=(x_cut, 0, 0)) for k_, v in out_.items()}


def loops(sl):
    """Ordered (y', z') polylines of a surface slice, in mm."""
    st = sl.strip(join=True, max_length=100000)
    res = []
    for i in range(st.n_cells):
        ids = st.get_cell(i).point_ids
        res.append(np.asarray(st.points)[ids][:, 1:] * MM)
    return res


def draw_atlas(ax, week):
    sl = atlas_curves(week, plane_x)
    for poly in loops(sl["pial"]):
        ax.fill(poly[:, 0], poly[:, 1], color="#9e9a91", lw=0)
    for poly in loops(sl["wm"]):
        ax.fill(poly[:, 0], poly[:, 1], color="#f4f2ec", lw=0)
    for poly in loops(sl["pial"]):
        ax.plot(poly[:, 0], poly[:, 1], color="#0b0b0b", lw=0.6)
    for poly in loops(sl["wm"]):
        ax.plot(poly[:, 0], poly[:, 1], color="#52514e", lw=0.5)


def draw_sim(ax, s, field, cmap, clim):
    pts = np.asarray(s.points)[:, 1:] * MM
    polys, vals = [], s.cell_data[field]
    for i in range(s.n_cells):
        polys.append(pts[s.get_cell(i).point_ids])
    pc = PolyCollection(polys, array=vals, cmap=cmap, edgecolors="face", linewidths=0.05)
    pc.set_clim(*clim)
    ax.add_collection(pc)
    ax.autoscale_view()
    return pc


def style(ax):
    ax.set_aspect("equal")
    ax.invert_yaxis()  # z' points down (Tallinen's frame): put superior at the top
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


plt.rcParams.update({"font.size": 9})
INK, INK2 = "#0b0b0b", "#52514e"

# ---- Figure 1: geometry and cortex, per model and week -----------------------------
rows = ["Atlas dHCP"] + list(MODELS)
fig, axs = plt.subplots(len(rows), len(args.weeks), figsize=(2.6 * len(args.weeks), 2.5 * len(rows)), dpi=120)
cache_fields = {}
for j, w in enumerate(args.weeks):
    axs[0, j].set_title(f"SG {w}", color=INK)
    draw_atlas(axs[0, j], w)
    for i, (name, ck) in enumerate(MODELS.items(), start=1):
        ax = axs[i, j]
        if w in ck and (ck[w] / "x.npy").exists():
            x, f = fields(ck[w], w)
            cache_fields[(name, w)] = (x, f)
            s = sim_slice(x, f, plane_x(x[:NS]))
            draw_sim(ax, s, "g_m", "Greys", (0, 1.4))
        else:
            ax.text(0.5, 0.5, "ajustement arrêté\nà SG 32" if name.startswith("Ajusté") else "arrêt numérique (NaN)",
                    ha="center", va="center", color=INK2, transform=ax.transAxes)
for i, r in enumerate(rows):
    axs[i, 0].set_ylabel(r, color=INK, fontsize=10)
for ax in axs.ravel():
    style(ax)
for i in range(len(rows)):
    axs[i, 0].yaxis.label.set_visible(True)
fig.suptitle("Coupes coronales (plan passant par le centre de chaque hémisphère gauche, vue de face, mm)",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.text(0.01, 0.005, "Atlas : ruban cortical (gris) entre surface piale et surface blanche. "
         "Simulation : indicateur cortical g_m (gris foncé = cortex en croissance, épaisseur H(t) de l'atlas), "
         "configuration déformée.\nTallinen = loi a_t = 1,85·t de Tallinen et al. (2016) dans gyroxide sur le maillage atlas SG 21 : le calcul explicite s'arrête (NaN, éléments écrasés dans l'insula) vers SG 25,4 ;\nparamètres locaux : a_t et H mesurés par sommet sur l'atlas (lissage ~4-5 mm), g_m affiché avec le H médian.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.045, 1, 0.97))
fig.savefig(out / "coronal_geometry.png", facecolor="white")
print(out / "coronal_geometry.png")

# ---- Figure 2: mechanical fields at selected weeks ---------------------------------
FIELDS = [
    ("g_m", "Cortex g_m", "Greys", (0, 1.4)),
    ("J_e", "J_e = det(F_e)\n(< 1 : compression)", "RdBu", (0.8, 1.2)),
    ("lam_min", "Étirement principal min\n(< 1 : compression)", "magma", (0.6, 1.0)),
    ("sig_vm", "Contrainte de von Mises / μ_g", "viridis", (0, 0.6)),
    ("disp", "Déplacement (mm)", "cividis", (0, 6)),
]
cases = [(name, w) for w in args.field_weeks for name in MODELS if (name, w) in cache_fields or
         (w in MODELS[name] and (MODELS[name][w] / "x.npy").exists())]
fig, axs = plt.subplots(len(cases), len(FIELDS), figsize=(3.0 * len(FIELDS), 2.7 * len(cases) + 0.8), dpi=120,
                        squeeze=False)
for i, (name, w) in enumerate(cases):
    x, f = cache_fields.get((name, w)) or fields(MODELS[name][w], w)
    s = sim_slice(x, f, plane_x(x[:NS]))
    for j, (key, label, cmap, clim) in enumerate(FIELDS):
        ax = axs[i, j]
        pc = draw_sim(ax, s, key, cmap, clim)
        style(ax)
        if i == 0:
            ax.set_title(label, color=INK, fontsize=9)
        if i == len(cases) - 1:
            fig.colorbar(pc, ax=axs[:, j], orientation="horizontal", fraction=0.04, pad=0.02)
    axs[i, 0].set_ylabel(f"{name.splitlines()[0]}\nSG {w}", color=INK, fontsize=9)
    axs[i, 0].yaxis.label.set_visible(True)
fig.suptitle("Champs mécaniques sur la coupe coronale (configuration déformée)", color=INK, fontsize=11, x=0.01,
             ha="left")
fig.savefig(out / "coronal_fields.png", facecolor="white", bbox_inches="tight")
print(out / "coronal_fields.png")
