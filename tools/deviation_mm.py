#!/usr/bin/env python3
"""Deviation in mm of simulated surfaces from the dHCP atlas, week by week, at several scales.

    python3 tools/deviation_mm.py <run dir> [--weeks 22-36] [--ckpt '{run}/w{w}/checkpoint']

For every week whose checkpoint exists (x.npy) and whose prep dir exists (data/adjoint_prep/w<w>,
tools/adjoint_prep.py): over the cortex vertices, |S^k x_i - S^k y_i| (S: neighbour averaging on the
surface, as in examples/adjoint_week.rs; k = 200 lobes, 50 primary folds, 10 secondary folds,
0 vertices), absolute (common frame), rigid (after the best rotation + translation, no scaling:
the placement of the hemisphere in the atlas frame is not a model property) and shape-only
(centred, rescaled to the atlas rms radius); RMS / p95 in mm, and the volume ratio model / atlas.
Writes <run>/deviation_mm.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
LEVELS = [200, 50, 10, 0]

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--weeks", default="22-36")
ap.add_argument("--ckpt", default="{run}/w{w}/checkpoint")
args = ap.parse_args()
w0, w1 = map(int, args.weeks.split("-"))
F = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
MM, NS = F["maxd_mm"], F["n_surface"]


def smoother(prep):
    e = np.load(prep / "edges.npy").astype(int)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(NS, NS))
    A = ((A + A.T) > 0).astype(float).tocsr()
    return 0.5 * sp.identity(NS) + 0.5 * sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A


def volume(p, tri):
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def stats(d):
    return {"rms": float(np.sqrt((d ** 2).mean())), "p95": float(np.percentile(d, 95))}


out = {}
S = None
print(f"{'SG':>3} {'V/Vatlas':>8} | " + " | ".join(f"k={k:<3} abs  rigid (p95)  shape" for k in LEVELS))
for w in range(w0, w1 + 1):
    ck = Path(args.ckpt.format(run=args.run, w=w))
    prep = ROOT / f"data/adjoint_prep/w{w}"
    if not (ck / "x.npy").exists() or not prep.exists():
        continue
    if S is None:
        S = smoother(prep)
    x = np.load(ck / "x.npy")[:NS]
    y = np.load(prep / "target.npy")
    cx = np.load(prep / "cortex.npy") > 0.5
    tri = np.load(prep / "faces.npy").astype(int)
    row = {"volume_ratio": volume(x, tri) / volume(y, tri)}
    sx, sy = x.copy(), y.copy()
    done = 0
    for k in sorted(LEVELS):
        for _ in range(k - done):
            sx, sy = S @ sx, S @ sy
        done = k
        a = np.linalg.norm(sx[cx] - sy[cx], axis=1) * MM
        xc, yc = sx[cx] - sx[cx].mean(0), sy[cx] - sy[cx].mean(0)
        rx, ry = np.sqrt((xc ** 2).sum(1).mean()), np.sqrt((yc ** 2).sum(1).mean())
        s = np.linalg.norm(xc / rx - yc / ry, axis=1) * ry * MM
        u, _, vt = np.linalg.svd(xc.T @ yc)
        dfix = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
        r = u @ dfix @ vt  # xc @ r ~ yc
        g = np.linalg.norm(xc @ r - yc, axis=1) * MM
        row[f"k{k}"] = {"absolute_mm": stats(a), "rigid_mm": stats(g), "shape_mm": stats(s)}
    out[w] = row
    print(f"{w:3d} {row['volume_ratio']:8.3f} | " + " | ".join(
        f"{row[f'k{k}']['absolute_mm']['rms']:5.2f} {row[f'k{k}']['rigid_mm']['rms']:5.2f} ({row[f'k{k}']['rigid_mm']['p95']:5.2f}) "
        f"{row[f'k{k}']['shape_mm']['rms']:5.2f}" for k in LEVELS))
(args.run / "deviation_mm.json").write_text(json.dumps(out, indent=1))
