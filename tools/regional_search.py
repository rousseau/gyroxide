#!/usr/bin/env python3
"""Gradient-free search of regional growth multipliers for one week (full simulations).

    python3 tools/regional_search.py --week 29 --regions 10 --out out/regional_w29

Week w-1 -> w from the checkpoint of the local mu(t) run. Region r (k-means of the cortex
vertices on the week-21 surface) scales that week's growth increment:
    a_w'(node) = a_{w-1} + m_r (a_w - a_{w-1})       (other knots unchanged)
Each evaluation resumes the checkpoint and simulates the week (8769 steps); metrics on
the final surface vs the atlas of week w: normalised symmetric Chamfer (objective),
smoothed mean-curvature and convex-hull-depth correlations (cortex), folding index.
Plan: baseline (twice, noise), one-sided sensitivities (m_r = 1 + delta), then a few
steps along the estimated descent direction. Results cached in <out>/evals.json.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.cluster.vq import kmeans2
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parent.parent
ATLAS = next((p for p in [Path("/Users/rousseau/Sync-Exp/Data/template_dHCP/dhcp_fetal_brain_surface_atlas/atlas"),
              Path(__file__).resolve().parent.parent / "data/atlas_dhcp"] if p.is_dir()), None)
FI_ATLAS = {w: v for w, v in zip(range(21, 37), [6.18, 6.18, 6.20, 6.23, 6.29, 6.40, 6.60, 6.83, 7.09, 7.43, 7.82, 8.10, 8.34, 8.59, 8.78, 8.92])}
ap = argparse.ArgumentParser()
ap.add_argument("--week", type=int, default=29)
ap.add_argument("--regions", type=int, default=10)
ap.add_argument("--delta", type=float, default=0.6)
ap.add_argument("--steps", type=float, nargs="+", default=[0.5, 1.0, 2.0])
ap.add_argument("--run", default="out/atlas_w21_nodal_mut")
ap.add_argument("--config", default="configs/atlas_w21_nodal_mut.toml")
ap.add_argument("--out", default="out/regional_w29")
args = ap.parse_args()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
W = args.week
k_prev, k_cur = W - 22, W - 21  # knot indices of weeks W-1 and W

F = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
NS, c0, d0 = F["n_surface"], np.array(F["cog_mm"]), F["maxd_mm"]
X0 = np.load(ROOT / "out/atlas_fit/mesh_cache.npz")["X0"]
cortex = np.loadtxt(ROOT / "data/meshes/w21_left_610k.gr.txt")[:NS] > 0.5
owner = np.arange(len(X0))
owner[NS:] = cKDTree(X0[:NS]).query(X0[NS:])[1]
A_TAB = np.load(ROOT / "data/meshes/w21_left_610k.nodal_a.npy")
TRI = pv.read(ATLAS / "fetal.week21.left.pial.surf.vtk").extract_surface().faces.reshape(-1, 4)[:, 1:]
FACES = np.column_stack([np.full(len(TRI), 3), TRI]).ravel()
e = np.concatenate([TRI[:, [0, 1]], TRI[:, [1, 2]], TRI[:, [2, 0]]])
ADJ = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(NS, NS))
ADJ = ((ADJ + ADJ.T) > 0).astype(float).tocsr()
DEG = np.asarray(ADJ.sum(1)).ravel()

# Regions: k-means on the week-21 cortex vertices (fixed seed), -1 outside the cortex.
reg_file = out / "regions.npy"
if reg_file.exists():
    region = np.load(reg_file).astype(int)
else:
    np.random.seed(0)
    _, lab = kmeans2(X0[:NS][cortex], args.regions, minit="++", seed=0)
    region = -np.ones(NS, int)
    region[cortex] = lab
    np.save(reg_file, region)
node_region = region[owner]


def atlas(w):
    p = np.asarray(pv.read(ATLAS / f"fetal.week{w}.left.pial.surf.vtk").points, float)
    return np.column_stack([-(p[:, 1] - c0[1]), p[:, 0] - c0[0], -(p[:, 2] - c0[2])]) / d0


def normalise(p):
    q = p[cortex]
    q = q - q.mean(0)
    return q / np.sqrt((q ** 2).sum(1).mean())


def shape_maps(p):
    a, b, c = p[TRI[:, 0]], p[TRI[:, 1]], p[TRI[:, 2]]
    Vs = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    q = (p - p.mean(0)) / abs(Vs) ** (1 / 3)
    h = pv.PolyData(q, FACES).curvature("mean") * (1 if Vs > 0 else -1)
    for _ in range(20):
        h = 0.5 * h + 0.5 * (ADJ @ h) / DEG
    eq = ConvexHull(q).equations
    depth = np.clip(-(q @ eq[:, :3].T + eq[:, 3]).max(1), 0, None)
    return h, depth, A / abs(Vs) ** (2 / 3)


Y = atlas(W)
YH = normalise(Y)
TREE_Y = cKDTree(YH)
HA, DA, _ = shape_maps(Y)
H21, D21, _ = shape_maps(atlas(21))


def metrics(p):
    xh = normalise(p)
    dxy = TREE_Y.query(xh)[0]
    dyx = cKDTree(xh).query(YH)[0]
    h, dd, fi = shape_maps(p)
    return {"chamfer": 0.5 * (dxy ** 2).mean() + 0.5 * (dyx ** 2).mean(),
            "r_curv": float(np.corrcoef(h[cortex], HA[cortex])[0, 1]),
            "r_depth": float(np.corrcoef(dd[cortex], DA[cortex])[0, 1]), "fi": float(fi)}


def read_vtk_points(f):
    L = Path(f).read_text().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    return np.loadtxt(L[i + 1:i + 1 + NS])


cache_file = out / "evals.json"
cache = json.loads(cache_file.read_text()) if cache_file.exists() else []


def evaluate(m, tag):
    m = np.asarray(m, float)
    for c in cache:
        if c["tag"] == tag:
            return c
    d = out / tag
    d.mkdir(exist_ok=True)
    a = A_TAB.copy()
    mult = np.where(node_region >= 0, m[np.maximum(node_region, 0)], 1.0)
    a[k_cur] = a[k_prev] + mult * (A_TAB[k_cur] - A_TAB[k_prev])
    np.save(d / "nodal_a.npy", a)
    cfg = (ROOT / args.config).read_text()
    cfg = re.sub(r'(?m)^a = ".*"$', f'a = "{d / "nodal_a.npy"}"', cfg)
    (d / "config.toml").write_text(cfg)
    t0 = time.time()
    log = subprocess.run([str(ROOT / "target-dev2/release/gyroxide"), "run", "--config", str(d / "config.toml"),
                          "--mesh", str(ROOT / "data/meshes/w21_left_610k.mesh"), "--resume",
                          str(ROOT / args.run / f"checkpoint_w{W - 1}"), "--steps", "8769", "--print-every", "2000",
                          "--out", str(d)], cwd=ROOT, capture_output=True, text=True)
    (d / "run.log").write_text(log.stdout + log.stderr)
    rows = [l for l in log.stdout.split("\n") if re.match(r"^\s+\d+\s", l)]
    ninv = int(rows[-1].split()[8]) if rows else -1
    nan = "NaN" in log.stdout
    res = {"tag": tag, "m": m.tolist(), "ninv": ninv, "nan": nan, "wall_s": round(time.time() - t0)}
    if not nan and (d / "surface_final.vtk").exists():
        res.update(metrics(read_vtk_points(d / "surface_final.vtk")))
    cache.append(res)
    cache_file.write_text(json.dumps(cache, indent=1))
    print(f"{tag:12s} chamfer {res.get('chamfer', float('nan')):.4e}  r_curv {res.get('r_curv', float('nan')):.3f}  "
          f"r_depth {res.get('r_depth', float('nan')):.3f}  FI {res.get('fi', float('nan')):.3f} (atlas {FI_ATLAS[W]})  "
          f"ninv {ninv}  {res['wall_s']}s", flush=True)
    return res


R = args.regions
base = np.ones(R)
b1 = evaluate(base, "base")
b2 = evaluate(base, "base_repeat")
noise = abs(b1["chamfer"] - b2["chamfer"])
print(f"baseline noise (|repeat - base|): {noise:.2e}; atlas SG 21 baseline r_curv "
      f"{np.corrcoef(H21[cortex], HA[cortex])[0, 1]:.3f} r_depth {np.corrcoef(D21[cortex], DA[cortex])[0, 1]:.3f}")
sens = np.zeros(R)
for r in range(R):
    m = base.copy()
    m[r] += args.delta
    res = evaluate(m, f"region_{r:02d}")
    sens[r] = (res["chamfer"] - b1["chamfer"]) / args.delta if "chamfer" in res else 0.0
print("sensitivities d(chamfer)/dm_r:", np.array2string(sens, precision=2))
direction = -sens / np.abs(sens).max()
best = b1
for s in args.steps:
    m = np.clip(base + s * direction, 0.0, 3.0)
    res = evaluate(m, f"step_{s:g}")
    if "chamfer" in res and res["chamfer"] < best["chamfer"]:
        best = res
print("best:", best["tag"], "m =", np.round(best["m"], 2), f"chamfer {best['chamfer']:.4e} (base {b1['chamfer']:.4e})")
