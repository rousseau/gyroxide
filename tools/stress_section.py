"""Coronal sections of a fitted model coloured by its stress (tools/stress_run.sh output): tangential
stress (along the cortex, in the section plane) and radial stress (along the growth normal), with
the atlas pial contour overlaid; plus statistics of the stress at gyral crowns and sulcal fundi.

    python3 tools/stress_section.py out/stress/lbm30 30 out/gallery/stress_sg30.png
Stress in units of the grey-matter shear modulus mu_g (Cauchy, tension > 0).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from make_slab import read_mesh  # noqa: E402
from model_gallery import Surf, rigid_to  # noqa: E402

INK, INK2, MUTED, BG = "#1B2430", "#4A5361", "#6B7280", "#FCFCFB"
DIV = LinearSegmentedColormap.from_list("div", ["#1F4E8C", "#7FA6D6", "#F4F1EA", "#E8A06B", "#9C3A0E"])


def section(x, tets, x0):
    """Polygons of the intersection of the tetrahedra with the plane x[:, 0] = x0, and their tets."""
    s = x[tets, 0] - x0
    cut = (s.min(1) < 0) & (s.max(1) > 0)
    idx = np.nonzero(cut)[0]
    polys = []
    for t in idx:
        p, d = x[tets[t]], s[t]
        pts = [p[i] + d[i] / (d[i] - d[j]) * (p[j] - p[i])
               for i in range(4) for j in range(i + 1, 4) if (d[i] > 0) != (d[j] > 0)]
        q = np.array(pts)[:, 1:]
        c = q.mean(0)
        q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
        polys.append(q)
    return polys, idx


def contour(v, faces, x0):
    s = v[faces, 0] - x0
    cut = (s.min(1) < 0) & (s.max(1) > 0)
    segs = []
    for f, d in zip(faces[cut], s[cut]):
        p = v[f]
        pts = [p[i] + d[i] / (d[i] - d[j]) * (p[j] - p[i]) for i, j in ((0, 1), (1, 2), (2, 0)) if (d[i] > 0) != (d[j] > 0)]
        if len(pts) == 2:
            segs.append([pts[0][1:], pts[1][1:]])
    return segs


def main():
    run, w, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    meta = json.loads(Path("data/meshes/w21_left_610k_s.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    _, tets, _, _, _ = read_mesh("data/meshes/w21_left_610k_s.mesh")
    x = np.load(run / "x_stress.npy")
    sig = np.load(run / "stress.npy").reshape(-1, 3, 3)
    gm = np.load(run / "stress_gm.npy")
    assert len(sig) == len(tets)
    P = Path("data/full/prep") / f"w{w}"
    y = np.load(P / "target.npy")[:ns]
    m = np.load(P / "cortex.npy")[:ns] > 0.5
    faces = np.load(P / "faces.npy").astype(int)
    S = Surf(faces, y, mm)
    # atlas brought into the model frame (inverse of the rigid fit of the model onto the atlas)
    ya = rigid_to(y, x[:ns], m)
    # growth normal of each tet ~ outward normal of the nearest model surface vertex
    xs = x[:ns]
    nrm = S.normals(xs)
    cen = x[tets].mean(1)
    dist, near = cKDTree(xs).query(cen)
    shallow = dist * mm < 3.0                  # core tets within 3 mm of the surface
    n = nrm[near]
    snn = np.einsum("ti,tij,tj->t", n, sig, n)
    # sulcal fundi / gyral crowns from the model's depth (p10 / p90), on the cortex tets
    dep = S.depth(xs)
    lo, hi = np.percentile(dep[m], [10, 90])
    fund, crown = (dep[near] < lo) & m[near], (dep[near] > hi) & m[near]
    ctx, core = gm > 0.5, gm < 0.5
    # tangential stress: mean of the two in-plane (surface) principal stresses, sigma_tt = (tr - snn) / 2
    stt = (np.trace(sig, axis1=1, axis2=2) - snn) / 2
    stats = {}
    for name, sel in [("crêtes, cortex", crown & ctx), ("fonds, cortex", fund & ctx),
                      ("sous les crêtes, cœur (< 3 mm)", crown & core & shallow), ("sous les fonds, cœur (< 3 mm)", fund & core & shallow)]:
        stats[name] = {"tangentielle": float(np.median(stt[sel])), "radiale": float(np.median(snn[sel])), "n": int(sel.sum())}
    (Path(out).with_suffix(".json")).write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    for k, v in stats.items():
        print(f"{k:24s} tangentielle {v['tangentielle']:+.3f}  radiale {v['radiale']:+.3f}  ({v['n']} tets)")

    # coronal planes: anterior (central region), middle (sylvian), posterior
    xr = xs[:, 0]
    planes = [(np.percentile(xr, q), lab) for q, lab in ((62, "antérieure"), (45, "moyenne"), (28, "postérieure"))]
    lim_t = np.percentile(np.abs(stt[ctx]), 97)
    lim_n = np.percentile(np.abs(snn), 97)
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), dpi=110, facecolor=BG)
    for j, (x0, lab) in enumerate(planes):
        polys, idx = section(x, tets, x0)
        segs_m = contour(xs, S.f, x0)
        segs_a = contour(ya, S.f, x0)
        for i, (val, lim, title) in enumerate([(stt, lim_t, "Contrainte tangentielle (le long du cortex)"),
                                               (snn, lim_n, "Contrainte radiale (le long de la normale)")]):
            ax = axes[i, j]
            pc = PolyCollection([p * mm for p in polys], array=val[idx], cmap=DIV,
                                norm=TwoSlopeNorm(0, -lim, lim), edgecolors="face", linewidths=0.2)
            ax.add_collection(pc)
            ax.add_collection(LineCollection([np.array(s_) * mm for s_ in segs_a], colors=INK, linewidths=1.6))
            ax.set_aspect("equal"); ax.autoscale()
            ax.invert_yaxis()                      # z is flipped in the mesh frame: superior up
            ax.axis("off")
            if j == 0:
                ax.set_title(title, loc="left", fontsize=15, color=INK)
            ax.text(0.0, 0.0, f"coupe {lab}", transform=ax.transAxes, fontsize=12, color=INK2)
            if j == 2:
                cb = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.02)
                cb.set_label("× μ_g (traction > 0, compression < 0)", fontsize=11, color=INK2)
    fig.text(0.01, 0.99, f"SG {w} · modèle enrichi (étape 5), coupes coronales à l’équilibre ; trait noir : surface piale de l’atlas",
             fontsize=15, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
