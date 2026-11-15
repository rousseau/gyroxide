#!/usr/bin/env python3
"""Distance entre équilibres (P05, hypothèse 5, déflation) : écart quadratique moyen des nœuds en mm, au mouvement
rigide près (même définition que gyroxide_sim::deflation), et maximum.

    python3 tools/defl_dist.py <réf x_0.npy> <x_0.npy> [<x_0.npy> ...]      (ou : --matrice a.npy b.npy c.npy ...)
"""
import sys

import numpy as np

MM = 32.99  # unité de position -> mm (--mm-per-unit)


def aligned_residual(x, k):
    cx, ck = x.mean(0), k.mean(0)
    u, _, vt = np.linalg.svd((x - cx).T @ (k - ck))
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return (x - cx) - (k - ck) @ r.T


def dist(x, k):
    d = MM * np.linalg.norm(aligned_residual(x, k), axis=1)
    return float(np.sqrt((d**2).mean())), float(d.max())


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--matrice":
        xs = [np.load(p) for p in a[1:]]
        for i, p in enumerate(a[1:]):
            print(f"{p:50s}", " ".join(f"{dist(xs[i], xs[j])[0]:6.3f}" for j in range(len(xs))))
    else:
        ref = np.load(a[0])
        for p in a[1:]:
            r, m = dist(np.load(p), ref)
            print(f"{p:60s} rms {r:.4f} mm  max {m:.3f} mm")
