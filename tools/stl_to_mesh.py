#!/usr/bin/env python3
"""Closed surface (STL) -> tetrahedral Netgen .mesh for gyroxide.

Decimates the surface, tetrahedralises it with TetGen (pip install tetgen),
normalises it like the commented `geom_norm.mesh` block of Tallinen's
Brains.cpp (x' = -(y-cy)/d, y' = (x-cx)/d, z' = -(z-cz)/d) and writes the
Netgen format read by Brains.cpp, BrainGrowth and gyroxide.

    python3 tools/stl_to_mesh.py brain.stl out.mesh --maxvol 2e-6

The surface triangles are kept as they are (TetGen -Y): use a surface with
uniform edges and set gyroxide's mesh spacing a to its mean edge.
"""
import argparse

import numpy as np
import pyvista as pv
import tetgen

ap = argparse.ArgumentParser()
ap.add_argument("stl")
ap.add_argument("out")
ap.add_argument("--faces", type=int, default=0, help="target surface triangles (0 = keep the surface)")
ap.add_argument("--maxvol", type=float, default=2e-5, help="max tet volume (normalised units)")
ap.add_argument("--quality", type=float, default=1.5, help="TetGen radius-edge bound")
ap.add_argument("--cog", type=float, nargs=3, help="normalisation centre (default: mean of the surface points)")
ap.add_argument("--maxd", type=float, help="normalisation scale (default: max |p - cog| component)")
args = ap.parse_args()

surf = pv.read(args.stl).clean().triangulate()
if args.faces and args.faces < surf.n_cells:
    # Quadric decimation leaves very uneven edges; short edges create spurious
    # self-contact (edge < 0.2 a), so prefer an already uniform surface.
    surf = surf.decimate(1.0 - args.faces / surf.n_cells).clean()

# Tallinen normalisation (a reflection: orientation is fixed below).
p = np.asarray(surf.points, dtype=float)
# Pass the cog/maxd printed by Brains.cpp (or found in a trace manifest) to land
# in the same normalised frame as the reference mesh, so that the hard-coded
# growth-mask ellipsoid and the midplane sit where Tallinen put them.
c = np.array(args.cog) if args.cog else p.mean(axis=0)
d = args.maxd if args.maxd else np.abs(p - c).max()
q = np.column_stack([-(p[:, 1] - c[1]), p[:, 0] - c[0], -(p[:, 2] - c[2])]) / d
surf = pv.PolyData(q, surf.faces)

tg = tetgen.TetGen(surf)
nodes, tets = tg.tetrahedralize(order=1, mindihedral=10, minratio=args.quality, maxvolume=args.maxvol, nobisect=True)[:2]
tets = np.asarray(tets, dtype=np.int64)

def vol(t):
    a, b, cc, dd = (nodes[t[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(cc - a, dd - a)) / 6.0

neg = vol(tets) < 0
tets[neg] = tets[neg][:, [0, 1, 3, 2]]  # right-handed
assert (vol(tets) > 0).all()

# Boundary faces, oriented outwards.
local = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
allf = tets[:, local].reshape(-1, 3)
opp = tets[:, [0, 1, 2, 3]].reshape(-1)
key = np.sort(allf, axis=1)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
bnd = cnt[inv.ravel()] == 1
faces, oppn = allf[bnd], opp[bnd]
a, b, cc = (nodes[faces[:, k]] for k in range(3))
inward = np.einsum("ij,ij->i", np.cross(b - a, cc - a), nodes[oppn] - a) > 0
faces[inward] = faces[inward][:, [0, 2, 1]]

e = np.concatenate([np.linalg.norm(nodes[faces[:, i]] - nodes[faces[:, j]], axis=1) for i, j in [(0, 1), (1, 2), (2, 0)]])
print(f"{len(nodes)} nodes, {len(tets)} tets, {len(faces)} faces; surface edge min {e.min():.4f} mean {e.mean():.4f} max {e.max():.4f}")

with open(args.out, "w") as f:
    f.write(f"{len(nodes)}\n")
    np.savetxt(f, nodes, fmt="%.9f")
    f.write(f"{len(tets)}\n")
    # Netgen order is left-handed: readers swap the last two nodes back.
    np.savetxt(f, np.column_stack([np.ones(len(tets), int), tets[:, [0, 1, 3, 2]] + 1]), fmt="%d")
    f.write(f"{len(faces)}\n")
    np.savetxt(f, np.column_stack([np.ones(len(faces), int), faces + 1]), fmt="%d")
