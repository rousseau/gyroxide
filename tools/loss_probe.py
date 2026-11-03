"""Which loss rewards folds? Synthetic surfaces built from the atlas on a slab region
(tools/make_slab.py), compared with each loss of examples/adjoint_week.rs (same definitions):

  flat         fold band removed: S^K y (K smoothing steps)
  half / x1.5  S^K y + f (y - S^K y), f = 0.5 / 1.5
  shift s mm   fold band moved by s mm across the sulcus (direction: across the sulcus axis)
  + optional real surfaces (label=x.npy, e.g. the start equilibrium and fitted states)

Each loss is reported relative to its value on the flat surface. A loss that does not push towards
erasing folds must rank the shifted and the attenuated folds below the flat surface (ratio < 1).

    python3 tools/loss_probe.py data/slab_cs 27 --sulcus cs.npy [label=x.npy ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slab"); ap.add_argument("week", type=int)
    ap.add_argument("--sulcus", required=True)
    ap.add_argument("--smooth", type=int, default=30, help="smoothing steps K of the flat surface")
    ap.add_argument("surfaces", nargs="*")
    args = ap.parse_args()
    slab = Path(args.slab)
    meta = json.loads((slab / "slab.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    P = slab / "prep" / f"w{args.week}"
    y = np.load(P / "target.npy")
    m = np.load(P / "cortex.npy") > 0.5
    F = np.load(P / "faces.npy").astype(int)
    Fm = F[m[F].all(1)]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel()
    S = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / deg) @ A
    M = sp.identity(ns) - sp.diags(1 / deg) @ A                 # umbrella Laplacian

    def smooth(v, k):
        for _ in range(k):
            v = S @ v
        return v

    def vnormals(p):
        fn = np.cross(p[F[:, 1]] - p[F[:, 0]], p[F[:, 2]] - p[F[:, 0]])
        vn = np.zeros_like(p)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        return vn / np.linalg.norm(vn, axis=1, keepdims=True)

    def fnormals(p):
        n = np.cross(p[Fm[:, 1]] - p[Fm[:, 0]], p[Fm[:, 2]] - p[Fm[:, 0]])
        a = 0.5 * np.linalg.norm(n, axis=1)
        return n / (2 * a[:, None]), a

    nu, at = fnormals(y)
    ny = vnormals(y)
    lap_y = M @ y

    def losses(x):
        d = x - y
        L = {}
        L["sommets"] = (np.linalg.norm(d[m], axis=1) ** 2).mean()
        L["sommets lissés (k=10)"] = (np.linalg.norm(smooth(d, 10)[m], axis=1) ** 2).mean()
        u, a = fnormals(x)
        L["normales"] = (at * (1 - np.einsum("ij,ij->i", u, nu))).sum() / at.sum()
        L["laplacien"] = (np.linalg.norm((M @ x - lap_y)[m], axis=1) ** 2).mean()
        L["aire"] = (np.log(a / at) ** 2).mean()
        L["point-plan"] = (np.einsum("ij,ij->i", d, ny)[m] ** 2).mean()
        return L

    ys = smooth(y, args.smooth)
    fold = y - ys
    cands = {"aplatie": ys, "demi-amplitude": ys + 0.5 * fold, "amplitude x1.5": ys + 1.5 * fold}
    # across-sulcus direction: tangent, perpendicular to the sulcus axis
    nodes = np.load(slab / "slab_nodes.npy")
    full2slab = {int(n): i for i, n in enumerate(nodes[:ns])}
    cs = np.array([full2slab[i] for i in np.load(args.sulcus) if int(i) in full2slab])
    axis = np.linalg.svd(y[cs] - y[cs].mean(0))[2][0]
    nys = vnormals(ys)
    across = np.cross(nys, axis)
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    tree = cKDTree(ys)
    for s in (1.0, 2.5):
        j = tree.query(ys - (s / mm) * across)[1]         # value taken from s mm upstream
        cands[f"décalée {s:g} mm"] = ys + fold[j]
    for spec in args.surfaces:
        label, path = spec.split("=")
        cands[label] = np.load(path)[:ns]
    base = losses(ys)
    names = list(base)
    print(f"SG {args.week}, {m.sum()} vertices; values relative to the flat surface (fold band removed by S^{args.smooth}):")
    print(f"{'surface':28s} " + " ".join(f"{n:>11s}" for n in names))
    for c, x in cands.items():
        L = losses(x)
        print(f"{c:28s} " + " ".join(f"{L[n] / base[n]:11.3f}" for n in names))


if __name__ == "__main__":
    main()
