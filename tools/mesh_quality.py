#!/usr/bin/env python3
"""Qualité d'un maillage Netgen de gyroxide : volumes, angles dièdres, éléments dans l'épaisseur du cortex.

    python3 tools/mesh_quality.py mesh.mesh [mesh.nodal_h.npy]

Imprime : nombre de nœuds et de tétraèdres, volume minimal, angle dièdre minimal (degrés) et part des tétraèdres sous
5 et 10 degrés, rapport rayon circonscrit sur arête la plus courte médian, et, avec la table d'épaisseur (ligne 0),
le nombre de tétraèdres traversés le long de l'épaisseur du cortex (H / arête équivalente des tétraèdres de la
couche corticale, médiane et premier décile).
"""
import sys

import numpy as np

nl = open(sys.argv[1]).read().split("\n")
n = int(nl[0])
X = np.array([l.split() for l in nl[1:1 + n]], float)
nt = int(nl[1 + n])
T = np.array([l.split()[1:5] for l in nl[2 + n:2 + n + nt]], int) - 1
P = X[T]
a, b, c, d = (P[:, k] for k in range(4))
vol = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6
# angles dièdres : pour chaque arête (i, j), plans des deux faces contenant l'arête
dih = []
for i, j, k, l in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]:
    e = P[:, j] - P[:, i]
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    u = P[:, k] - P[:, i]
    v = P[:, l] - P[:, i]
    u -= np.einsum("ij,ij->i", u, e)[:, None] * e
    v -= np.einsum("ij,ij->i", v, e)[:, None] * e
    cos = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    dih.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
dih = np.array(dih)
mn = dih.min(0)
edges = np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)])
print(f"{n} nœuds, {nt} tétraèdres ; volume min {vol.min():.2e} (médian {np.median(vol):.2e})")
print(f"angle dièdre min {mn.min():.2f}° ; sous 5° : {100 * (mn < 5).mean():.2f} % ; sous 10° : {100 * (mn < 10).mean():.2f} % ; max {dih.max():.1f}°")
print(f"arête min {edges.min():.2e} ; rapport arête max / min médian {np.median(edges.max(0) / edges.min(0)):.2f}")
if len(sys.argv) > 2:
    h = np.load(sys.argv[2])[0]
    ns = 32492
    from scipy.spatial import cKDTree
    dc = cKDTree(X[:ns]).query(P.mean(1))[0]
    hn = h[T].mean(1)
    cort = dc < hn
    eq = np.cbrt(6 * np.sqrt(2) * vol)
    layers = hn[cort] / eq[cort]
    print(f"tétraèdres corticaux {cort.sum()} ; épaisseur / arête équivalente : médiane {np.median(layers):.2f}, décile bas {np.percentile(layers, 10):.2f}")
