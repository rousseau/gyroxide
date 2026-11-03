"""Whole-hemisphere fits (tools/make_full.py) region by region: on the cortex, in the slab regions
(data/slab_*), and outside them, the vertex deviation (mm, after a rigid alignment over the cortex),
the mean angle between face normals and the atlas ones (deg), the correlation of the mean
curvature with the atlas, and the fold depth (p95 - p5 of the depth along the normals of the S^30
smoothed surface, mm) against the atlas.

    python3 tools/full_eval.py 27 out/full/fit27_combo_band out/full/fit27_lawfit100 ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def last_x(run):
    ks = [int(p.stem.split("_")[1]) for p in Path(run).glob("x_*.npy") if p.stem.split("_")[1].isdigit()]
    return np.load(Path(run) / f"x_{max(ks)}.npy")


def rigid(a, b):
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    return lambda p: (p - ca) @ r + cb


def main():
    w, runs = int(sys.argv[1]), sys.argv[2:]
    meta = json.loads(Path("data/meshes/w21_left_610k_s.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    p = Path("data/full/prep") / f"w{w}"
    y = np.load(p / "target.npy")[:ns]
    cortex = np.load(p / "cortex.npy")[:ns] > 0.5
    faces = np.load(p / "faces.npy").astype(int)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    deg = np.asarray(a.sum(1)).ravel()
    s = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / deg) @ a
    lap = sp.identity(ns) - sp.diags(1 / deg) @ a

    def vnormals(q):
        fn = np.cross(q[faces[:, 1]] - q[faces[:, 0]], q[faces[:, 2]] - q[faces[:, 0]])
        vn = np.zeros_like(q)
        for k in range(3):
            np.add.at(vn, faces[:, k], fn)
        return vn / np.linalg.norm(vn, axis=1, keepdims=True)

    def fnormals(q):
        fn = np.cross(q[faces[:, 1]] - q[faces[:, 0]], q[faces[:, 2]] - q[faces[:, 0]])
        return fn / np.linalg.norm(fn, axis=1, keepdims=True)

    def depth(q):
        qs = q
        for _ in range(30):
            qs = s @ qs
        return np.einsum("ij,ij->i", q - qs, vnormals(qs)) * mm

    def curv(q):
        return np.einsum("ij,ij->i", lap @ q, vnormals(q))

    regions = {"cortex": cortex}
    inside = np.zeros(ns, bool)
    for slab in sorted(Path("data").glob("slab_*")):
        nodes = np.load(slab / "slab_nodes.npy")
        n = len(np.load(sorted((slab / "prep").glob("w*"))[0] / "cortex.npy"))    # surface vertices
        r = np.zeros(ns, bool)
        r[nodes[:n]] = True
        regions[slab.name.removeprefix("slab_")] = r & cortex
        inside |= r
    regions["ailleurs"] = cortex & ~inside
    fmask = {k: v[faces].all(1) for k, v in regions.items()}
    dy, hy, ny = depth(y), curv(y), fnormals(y)
    print(f"SG {w}: " + ", ".join(f"{k} {v.sum()} vertices, atlas fold depth "
                                  f"{np.percentile(dy[v], 95) - np.percentile(dy[v], 5):.2f} mm"
                                  for k, v in regions.items()))
    print(f"{'run':30s} {'region':9s} | vertex mm | normal angle | corr H | fold depth")
    start = Path(runs[0]) / "x_0.npy" if runs else None
    for label, x in [("start (" + Path(runs[0]).name + ")", np.load(start)[:ns])] + \
            [(Path(r).name, last_x(r)[:ns]) for r in runs]:
        x = rigid(x[cortex], y[cortex])(x)
        dx, hx, nx = depth(x), curv(x), fnormals(x)
        for k, v in regions.items():
            dev = np.linalg.norm(x[v] - y[v], axis=1) * mm
            ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", nx[fmask[k]], ny[fmask[k]]), -1, 1)))
            print(f"{label:30s} {k:9s} | {np.sqrt((dev ** 2).mean()):9.3f} | {ang.mean():10.2f} ° | "
                  f"{np.corrcoef(hx[v], hy[v])[0, 1]:6.3f} | {np.percentile(dx[v], 95) - np.percentile(dx[v], 5):6.2f} mm")


if __name__ == "__main__":
    main()
