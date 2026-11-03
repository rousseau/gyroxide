"""Figure of slab fits (tools/make_slab.py): the region seen from outside, coloured by mean curvature,
for the atlas, the start and each run, and cross-sections of the surfaces through the sulcus.

    python3 tools/slab_figure.py data/slab_cs 27 out/fig.png --sulcus cs.npy \\
        "Sommets=out/slab_cs/fit27_vertex" "Normales=out/slab_cs/fit27_normal" ...
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

INK, GREY, ACCENT = "#1B2430", "#6B7280", "#C2571A"
COLOURS = ["#2A5DA8", "#C2571A", "#3B8C6E", "#8E5BB5", "#B8860B", "#C0392B"]


def last_x(run):
    ks = [int(p.stem.split("_")[1]) for p in Path(run).glob("x_*.npy") if p.stem.split("_")[1].isdigit()]
    return np.load(Path(run) / f"x_{max(ks)}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slab"); ap.add_argument("week", type=int); ap.add_argument("out")
    ap.add_argument("--sulcus", required=True, help="full-mesh surface indices of the sulcus (its axis sets the cut)")
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--cuts", type=float, nargs="+", default=[-0.25, 0.0, 0.25], help="cut positions along the sulcus (fractions)")
    args = ap.parse_args()

    slab = Path(args.slab)
    meta = json.loads((slab / "slab.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    P = slab / "prep" / f"w{args.week}"
    y = np.load(P / "target.npy")
    m = np.load(P / "cortex.npy") > 0.5
    F = np.load(P / "faces.npy").astype(int)
    nodes = np.load(slab / "slab_nodes.npy")
    full2slab = {int(n): i for i, n in enumerate(nodes[:ns])}
    cs = np.array([full2slab[i] for i in np.load(args.sulcus) if int(i) in full2slab])
    runs = [("Atlas", y), ("Départ (croissance mesurée)", np.load(Path(args.runs[0].split("=")[1]) / "x_0.npy")[:ns])]
    runs += [(r.split("=")[0], last_x(r.split("=")[1])[:ns]) for r in args.runs]

    # frame: z along the mean outward normal of the region, x across the sulcus, y along it
    def vnormals(p):
        fn = np.cross(p[F[:, 1]] - p[F[:, 0]], p[F[:, 2]] - p[F[:, 0]])
        vn = np.zeros_like(p)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        return vn / np.linalg.norm(vn, axis=1, keepdims=True)
    nz = vnormals(y)[m].mean(0); nz /= np.linalg.norm(nz)
    c = y[cs].mean(0)
    u, s, vt = np.linalg.svd(y[cs] - c)
    ax_y = vt[0] - (vt[0] @ nz) * nz; ax_y /= np.linalg.norm(ax_y)
    ax_x = np.cross(ax_y, nz)
    R = np.stack([ax_x, ax_y, nz])                       # rows: frame axes
    to_frame = lambda p: (p - c) @ R.T * mm
    half = np.abs((y[cs] - c) @ ax_y).max() * mm

    def curvature(p):
        a, b, cc = p[F[:, 0]], p[F[:, 1]], p[F[:, 2]]
        cot = lambda uu, vv: np.einsum("ij,ij->i", uu, vv) / (np.linalg.norm(np.cross(uu, vv), axis=1) + 1e-30)
        W = np.concatenate([cot(b - a, cc - a), cot(cc - b, a - b), cot(a - cc, b - cc)])
        I, J = np.concatenate([F[:, 1], F[:, 2], F[:, 0]]), np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
        L = sp.coo_matrix((W, (I, J)), shape=(ns, ns)); L = L + L.T
        L = sp.diags(np.asarray(L.sum(1)).ravel()) - L
        area = np.zeros(ns); fa = 0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1) / 3
        for k in range(3):
            np.add.at(area, F[:, k], fa)
        return -0.5 * np.einsum("ij,ij->i", -(L @ p) / (2 * area[:, None]), vnormals(p)) / mm   # 1/mm

    fm = m[F].all(1)
    hmax = np.quantile(np.abs(curvature(y)[m]), 0.90)
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=-hmax, vmax=hmax)
    light = np.array([-0.3, 0.4, 0.86]); light /= np.linalg.norm(light)

    ncol = 4
    nrow_s = int(np.ceil(len(runs) / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 3.9 * nrow_s + 4.4), dpi=150, facecolor="white")
    gs = fig.add_gridspec(nrow_s + 1, ncol, height_ratios=[3.9] * nrow_s + [4.4], hspace=0.18, wspace=0.05)
    q = to_frame(y)[m]
    lim = (q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max())
    for k, (label, p) in enumerate(runs):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        v = to_frame(p)
        h = curvature(p)
        Pv = v[F[fm]]
        nrm = np.cross(Pv[:, 1] - Pv[:, 0], Pv[:, 2] - Pv[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        col = plt.get_cmap("RdBu_r")(norm(np.clip(h[F[fm]].mean(1), -hmax, hmax)))
        col[:, :3] *= (0.55 + 0.45 * np.clip(nrm @ light, 0, 1))[:, None]
        order = np.argsort(Pv[:, :, 2].mean(1))          # far faces first
        ax.add_collection(PolyCollection(Pv[order][:, :, :2], facecolors=col[order], edgecolors=col[order], linewidths=0.2))
        for t in args.cuts:
            ax.axhline(t * half, color=INK, lw=0.6, ls=(0, (3, 3)), alpha=0.6)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal"); ax.axis("off")
        d = np.linalg.norm(p - y, axis=1)[m] * mm
        sub = "" if k == 0 else f"{np.sqrt((d ** 2).mean()):.2f} mm".replace(".", ",")
        ax.set_title(f"{label}\n{sub}", fontsize=12, color=ACCENT if k > 1 else INK, loc="left")

    # cross-sections: intersection of each surface with planes y = const (frame), restricted to the loss faces
    gsec = gs[nrow_s, :].subgridspec(1, len(args.cuts), wspace=0.12)
    for j, t in enumerate(args.cuts):
        ax = fig.add_subplot(gsec[0, j])
        y0 = t * half
        for k, (label, p) in enumerate(runs):
            v = to_frame(p)
            segs = []
            for f in F[fm]:
                a = v[f]
                sgn = a[:, 1] - y0
                if sgn.min() >= 0 or sgn.max() <= 0:
                    continue
                pts = []
                for i0, i1 in ((0, 1), (1, 2), (2, 0)):
                    if (sgn[i0] > 0) != (sgn[i1] > 0):
                        w = sgn[i0] / (sgn[i0] - sgn[i1])
                        pts.append(a[i0] + w * (a[i1] - a[i0]))
                if len(pts) == 2:
                    segs.append(pts)
            style = dict(color=INK, lw=2.6) if k == 0 else dict(color=GREY, lw=1.6, ls="--") if k == 1 else dict(color=COLOURS[(k - 2) % len(COLOURS)], lw=1.4)
            for s_ in segs:
                ax.plot([s_[0][0], s_[1][0]], [s_[0][2], s_[1][2]], **style)
            ax.plot([], [], label=label, **style)
        ax.set_aspect("equal"); ax.set_title(f"Coupe {j + 1}", fontsize=12, color=INK, loc="left")
        ax.tick_params(labelsize=9, colors=GREY)
        for sp_ in ax.spines.values():
            sp_.set_color("#D0D4D9")
        ax.set_xlabel("mm (en travers du sillon)", fontsize=9, color=GREY)
        if j == 0:
            ax.set_ylabel("mm (le long de la normale)", fontsize=9, color=GREY)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, frameon=False, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.06))
    fig.savefig(args.out, facecolor="white", bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
