"""Compare loss functions on a slab fit (tools/make_slab.py): per run, on the loss vertices, vertex
deviation (mm), mean angle between face normals and the atlas ones (deg), correlation of the mean
curvature with the atlas, and the fold depth (spread of the surface along the normals of its
smoothed version, mm) against the atlas.

    python3 tools/slab_eval.py data/slab_cs 27 out/slab_cs/fit27_vertex out/slab_cs/fit27_normal ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def last_x(run):
    ks = [int(p.stem.split("_")[1]) for p in Path(run).glob("x_*.npy") if p.stem.split("_")[1].isdigit()]
    return np.load(Path(run) / f"x_{max(ks)}.npy"), max(ks)


def main():
    slab, w, runs = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    meta = json.loads((slab / "slab.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    P = slab / "prep" / f"w{w}"
    y = np.load(P / "target.npy")
    m = np.load(P / "cortex.npy") > 0.5
    F = np.load(P / "faces.npy").astype(int)
    Fm = F[m[F].all(1)]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    S = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A

    def normals(p, faces):
        n = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def curvature(p):
        # mean curvature from the cotangent Laplacian, signed by the vertex normals
        a, b, c = p[F[:, 0]], p[F[:, 1]], p[F[:, 2]]
        cot = lambda u, v: np.einsum("ij,ij->i", u, v) / (np.linalg.norm(np.cross(u, v), axis=1) + 1e-30)
        W = np.concatenate([cot(b - a, c - a), cot(c - b, a - b), cot(a - c, b - c)])
        I, J = np.concatenate([F[:, 1], F[:, 2], F[:, 0]]), np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
        L = sp.coo_matrix((W, (I, J)), shape=(ns, ns)); L = L + L.T
        L = sp.diags(np.asarray(L.sum(1)).ravel()) - L
        area = np.zeros(ns); fa = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) / 3
        for k in range(3):
            np.add.at(area, F[:, k], fa)
        vn = np.zeros_like(p); fn = np.cross(b - a, c - a)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True)
        return -0.5 * np.einsum("ij,ij->i", -(L @ p) / (2 * area[:, None]), vn)

    def depth(p):
        sm = p.copy()
        for _ in range(30):
            sm = S @ sm
        vn = np.zeros_like(sm); fn = np.cross(sm[F[:, 1]] - sm[F[:, 0]], sm[F[:, 2]] - sm[F[:, 0]])
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True)
        h = np.einsum("ij,ij->i", p - sm, vn)[m] * mm
        return np.quantile(h, 0.95) - np.quantile(h, 0.05)

    hy, ny, dy = curvature(y), normals(y, Fm), depth(y)
    print(f"SG {w}: {m.sum()} loss vertices, {len(Fm)} faces; atlas fold depth (p95 - p5 along the smoothed normals) {dy:.2f} mm")
    print(f"{'run':28s} {'iter':>4s} | {'vertex mm':>9s} | {'normal angle':>12s} | {'corr H':>6s} | {'fold depth':>10s}")
    for run in runs:
        for tag, (x, k) in (("start", (np.load(Path(run) / "x_0.npy"), 0)), ("end", last_x(run))):
            if tag == "start" and run != runs[0]:
                continue
            x = x[:ns]
            d = np.linalg.norm(x - y, axis=1)[m] * mm
            ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", normals(x, Fm), ny), -1, 1))).mean()
            hx = curvature(x)
            name = "start (measured growth)" if tag == "start" else Path(run).name
            print(f"{name:28s} {k:4d} | {np.sqrt((d ** 2).mean()):9.3f} | {ang:10.2f} ° | {np.corrcoef(hx[m], hy[m])[0, 1]:6.3f} | {depth(x):7.2f} mm")


if __name__ == "__main__":
    main()
