"""Remodelled core growth: absorb a fraction alpha of the elastic deformation of the core.

At positions x, the total deformation of a tetrahedron is F = A(x) A(X)^-1 (X: reference
mesh). In the core (g_m ~ 0) the growth is the tensor U of the `core` table, and the elastic
part F U^-1 stores the energy that the core returns on release. Remodelling replaces U by

    U' = exp(alpha log S + (1 - alpha) log U),   S = sqrt(F^T F)   (right stretch of F),

so that alpha = 1 leaves the core stress free at x (F U'^-1 is a rotation) and alpha = 0 keeps
U. Writes U' per tetrahedron, [n_tets, 6] (xx, yy, zz, xy, xz, yz), for `--core-override`.

    python3 tools/absorb_core.py mesh.mesh core.npy knot x.npy alpha out.npy
"""
import sys

import numpy as np


def sym_fun(m, f):
    w, v = np.linalg.eigh(m)
    return np.einsum("tij,tj,tkj->tik", v, f(w), v)


def main():
    mesh, core, knot, xpath, alpha, out = sys.argv[1:7]
    knot, alpha = int(knot), float(alpha)
    tok = open(mesh).read().split()
    n = int(tok[0])
    X = np.array(tok[1:1 + 3 * n], dtype=float).reshape(n, 3)
    nt = int(tok[1 + 3 * n])
    T = np.array(tok[2 + 3 * n:2 + 3 * n + 5 * nt], dtype=np.int64).reshape(nt, 5)[:, 1:] - 1
    x = np.load(xpath)
    edges = lambda p: np.stack([p[T[:, k]] - p[T[:, 0]] for k in (1, 2, 3)], axis=2)
    F = edges(x) @ np.linalg.inv(edges(X))
    S = sym_fun(np.einsum("tki,tkj->tij", F, F), np.sqrt)
    c = np.load(core, mmap_mode="r")[knot]                     # [n_nodes, 6]
    c = c[T].mean(1)                                           # per tet, as the simulation does
    U = np.empty((nt, 3, 3))
    for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
        U[:, i, j] = U[:, j, i] = c[:, k]
    Lu = sym_fun(U, np.log)
    Ls = sym_fun(S, np.log)
    Up = sym_fun(alpha * Ls + (1 - alpha) * Lu, np.exp)
    np.save(out, np.ascontiguousarray(np.stack([Up[:, 0, 0], Up[:, 1, 1], Up[:, 2, 2], Up[:, 0, 1], Up[:, 0, 2], Up[:, 1, 2]], 1)))
    e = np.linalg.norm((S - U).reshape(nt, 9), axis=1)
    print(f"alpha {alpha}: |S - U| per tet median {np.median(e):.3f}, p99 {np.quantile(e, .99):.3f}; "
          f"det U median {np.median(np.linalg.det(U)):.3f}, det U' median {np.median(np.linalg.det(Up)):.3f}")


if __name__ == "__main__":
    main()
