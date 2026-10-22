"""Maps of the fitted fields of a slab fit (adjoint_week --fit-fields): a - a_measured, thickening
correction c, core growth correction k, on the atlas surface of the region, with the sulcal fundi
(atlas mean curvature) outlined.

    python3 tools/slab_fields.py data/slab_cs 27 out/slab_cs/fit27_band_ck out.png
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def main():
    slab, week, run, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    meta = json.loads((slab / "slab.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    P = slab / "prep" / f"w{week}"
    y = np.load(P / "target.npy"); m = np.load(P / "cortex.npy") > 0.5; F = np.load(P / "faces.npy").astype(int)
    th = np.load(run / "theta_final.npy"); na = ns
    a0 = np.load(slab / "slab.nodal_a.npy")[week - 21][:na]
    fields = [("a − a mesuré (croissance tangentielle)", th[:na] - a0)]
    names = ["c (épaississement)", "k (croissance du cœur)"]
    for i in range(1, len(th) // na):
        fields.append((names[i - 1] if len(th) // na == 3 else f"champ {i}", th[i * na:(i + 1) * na]))
    # frame: mean outward normal of the region as the viewing direction
    fn = np.cross(y[F[:, 1]] - y[F[:, 0]], y[F[:, 2]] - y[F[:, 0]])
    nz = fn[m[F].all(1)].sum(0); nz /= np.linalg.norm(nz)
    ax_x = np.cross(nz, [0, 0, 1.0]); ax_x /= np.linalg.norm(ax_x); ax_y = np.cross(nz, ax_x)
    q = (y - y[m].mean(0)) @ np.stack([ax_x, ax_y, nz]).T * mm
    a, b, c = y[F[:, 0]], y[F[:, 1]], y[F[:, 2]]
    cot = lambda u, v: np.einsum("ij,ij->i", u, v) / (np.linalg.norm(np.cross(u, v), axis=1) + 1e-30)
    W = np.concatenate([cot(b - a, c - a), cot(c - b, a - b), cot(a - c, b - c)])
    I, J = np.concatenate([F[:, 1], F[:, 2], F[:, 0]]), np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
    L = sp.coo_matrix((W, (I, J)), shape=(ns, ns)); L = L + L.T; L = sp.diags(np.asarray(L.sum(1)).ravel()) - L
    area = np.zeros(ns); fa = 0.5 * np.linalg.norm(fn, axis=1) / 3
    for k in range(3):
        np.add.at(area, F[:, k], fa)
    vn = np.zeros_like(y)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    H = -0.5 * np.einsum("ij,ij->i", -(L @ y) / (2 * area[:, None]), vn)
    fundus = H < np.quantile(H[m], 0.2)
    fm = m[F].all(1)
    order = np.argsort(q[F[fm]][:, :, 2].mean(1))
    fig, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 6), dpi=150)
    for ax, (title, f) in zip(np.atleast_1d(axes), fields):
        v = np.abs(f[m]).max()
        col = plt.get_cmap("RdBu_r")(np.clip(f[F[fm]].mean(1) / v * 0.5 + 0.5, 0, 1))
        pc = q[F[fm]][order][:, :, :2]
        ax.add_collection(PolyCollection(pc, facecolors=col[order], edgecolors=col[order], linewidths=0.2))
        fu = fundus[F[fm]].all(1)[order]
        ax.add_collection(PolyCollection(pc[fu], facecolors="none", edgecolors="#1B2430", linewidths=0.15, alpha=0.5))
        ax.set_xlim(q[m, 0].min(), q[m, 0].max()); ax.set_ylim(q[m, 1].min(), q[m, 1].max())
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{title}\n±{v:.2f} (bleu < 0 < rouge)", fontsize=11, loc="left")
    fig.suptitle("Champs ajustés ; contours : fonds de sillon de l'atlas (20 % les plus concaves)", fontsize=11, y=0.02)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
