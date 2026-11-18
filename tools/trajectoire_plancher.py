#!/usr/bin/env python3
"""Plancher de représentation du paramétrage de trajectoire de P07 (E61, tâche 1) : à quel point une sigmoïde ou une loi de
Gompertz par sommet, a_i(t) = A_i f((t - t0_i) / r_i) avec f(0) = 0, reproduit les a_i(t_w) de SG 22 à 30 ?

    python3 tools/trajectoire_plancher.py <dossier gyroxide> [--semaines 22-30]

Trois séries : la croissance mesurée (table nodale de l'atlas), les a ajustés de la chaîne (out/paper_r0857/fit<SG>_lbm), et leur
différence (la correction que l'inversion apporte). Pour chaque famille, recherche en grille de (t0, r) et A en moindres carrés
(linéaire en A). Rend, par famille, le résidu relatif ||Y - modèle||² / ||Y||² ; la référence est le rang 1 (A_i fois une courbe commune).
"""
import sys
from pathlib import Path

import numpy as np

NS = 32492


def famille(kind, z, z0):
    f = 1 / (1 + np.exp(-z)) - 1 / (1 + np.exp(-z0)) if kind == "sigmoide" else np.exp(-np.exp(-z)) - np.exp(-np.exp(-z0))
    return f / (np.abs(f).max() + 1e-12)


def ajuste(kind, t, Y):
    """Y : (semaines, sommets). Rend (résidu par sommet, paramètres (A, t0, r) par sommet)."""
    best = np.full(Y.shape[1], np.inf)
    par = np.zeros((Y.shape[1], 3))
    for t0 in np.linspace(0.0, 1.2, 25):
        for r in np.geomspace(0.03, 0.6, 20):
            f = famille(kind, (t - t0) / r, (0 - t0) / r)
            ff = (f * f).sum()
            if ff < 1e-12:
                continue
            A = (f[:, None] * Y).sum(0) / ff
            res = ((Y - A[None, :] * f[:, None]) ** 2).sum(0)
            m = res < best
            best[m] = res[m]
            par[m] = np.stack([A[m], np.full(m.sum(), t0), np.full(m.sum(), r)], 1)
    return best, par


def main():
    root = Path(sys.argv[1])
    lo, hi = 22, 30
    if "--semaines" in sys.argv:
        lo, hi = map(int, sys.argv[sys.argv.index("--semaines") + 1].split("-"))
    W = np.arange(lo, hi + 1)
    t = (W - 21) / 15.0
    mes = np.load(root / "data/meshes/w21_left_610k.nodal_a.npy")[W - 21, :NS]
    cha = np.array([np.load(root / f"out/paper_r0857/fit{w}_lbm/theta_final.npy")[:NS] for w in W])
    for nom, Y in (("mesuré", mes), ("chaîne (a ajusté)", cha), ("correction chaîne - mesuré", cha - mes)):
        u, s, vt = np.linalg.svd(Y, full_matrices=False)
        tot = (Y ** 2).sum()
        out = [f"rang 1 {((Y - np.outer(u[:, 0] * s[0], vt[0])) ** 2).sum() / tot:.4f}"]
        for kind in ("sigmoide", "gompertz"):
            res, par = ajuste(kind, t, Y)
            out.append(f"{kind} {res.sum() / tot:.4f} (t0 médian {np.median(par[:, 1]):.2f}, r médian {np.median(par[:, 2]):.2f})")
        print(f"SG {lo}-{hi}, {nom} : résidu relatif ||.||² : " + " ; ".join(out))


if __name__ == "__main__":
    main()
