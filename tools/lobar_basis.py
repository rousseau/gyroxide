"""Smooth basis maps for a lobe-scale core growth correction (adjoint_week --field-law-k): the first
Laplace-Beltrami eigenfunctions of the reference (SG 21) surface, constant mode left out, centred
and scaled to unit standard deviation on the cortex, / 100 like the depth maps of data/full/law.

    python3 tools/lobar_basis.py --n 40 --out data/full/law
    python3 tools/lobar_basis.py --n 80 --probe out/full/fit30_law 30   # lobe-scale error captured

writes <out>/lb{n}/basis_lb{j:03d}_s100.npy (j = 1..n) and <out>/lb{n}/eigenvalues.npy. With --probe,
prints the share of the S^200 (lobe) deviation of a whole-hemisphere fit that the first modes span.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parent.parent


def cotan_laplacian(x, faces):
    """Cotangent stiffness matrix L (positive semi-definite) and lumped mass M of a triangle mesh."""
    n = len(x)
    rows, cols, vals = [], [], []
    area = np.zeros(n)
    for k in range(3):
        i, j, o = faces[:, k], faces[:, (k + 1) % 3], faces[:, (k + 2) % 3]
        u, v = x[i] - x[o], x[j] - x[o]
        cot = np.einsum("ij,ij->i", u, v) / np.linalg.norm(np.cross(u, v), axis=1)
        rows += [i, j]
        cols += [j, i]
        vals += [-0.5 * cot, -0.5 * cot]
    fa = 0.5 * np.linalg.norm(np.cross(x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]]), axis=1)
    for k in range(3):
        np.add.at(area, faces[:, k], fa / 3)
    w = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    lap = w - sp.diags(np.asarray(w.sum(1)).ravel())
    return lap, sp.diags(area)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default="data/meshes/w21_left_610k_s.mesh")
    ap.add_argument("--prep", default="data/full/prep/w27", help="faces.npy, cortex.npy")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out")
    ap.add_argument("--probe", nargs=2, metavar=("RUN", "WEEK"))
    args = ap.parse_args()
    meta = json.loads((ROOT / args.mesh).with_suffix(".json").read_text())
    ns = meta["n_surface"]
    x = np.loadtxt((ROOT / args.mesh).read_text().split("\n")[1:1 + ns])
    prep = ROOT / args.prep
    faces = np.load(prep / "faces.npy").astype(int)
    cortex = np.load(prep / "cortex.npy")[:ns] > 0.5
    lap, mass = cotan_laplacian(x, faces)
    lam, phi = eigsh(lap, k=args.n + 1, M=mass, sigma=-1e-6, which="LM")
    order = np.argsort(lam)
    lam, phi = lam[order][1:], phi[:, order][:, 1:]
    print("eigenvalues (normalised units):", np.array2string(lam[[0, 9, 19, 39, -1] if args.n >= 40 else [0, -1]], precision=1))
    if args.out:
        d = ROOT / args.out / f"lb{args.n}"
        d.mkdir(parents=True, exist_ok=True)
        for j in range(args.n):
            f = phi[:, j] - phi[cortex, j].mean()
            np.save(d / f"basis_lb{j + 1:03d}_s100.npy", f / f[cortex].std() / 100)
        np.save(d / "eigenvalues.npy", lam)
        print(f"{args.n} maps in {d}")
    if args.probe:
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        from full_eval import last_x, rigid
        run, w = Path(args.probe[0]), int(args.probe[1])
        p = ROOT / "data/full/prep" / f"w{w}"
        y = np.load(p / "target.npy")[:ns]
        xm = last_x(run)[:ns]
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns)).tocsr()
        a = ((a + a.T) > 0).astype(float)
        s = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a
        r = rigid(xm[cortex], y[cortex])(xm) - y
        for _ in range(200):
            r = s @ r
        # M-orthogonal projection of each component of the lobe-scale error onto the first modes
        tot = np.einsum("ij,i,ij->", r, mass.diagonal(), r)
        c0 = (mass.diagonal() @ r) / mass.diagonal().sum()
        r0 = r - c0
        for m in (5, 10, 20, 40, 80, 160):
            if m > args.n:
                break
            coef = phi[:, :m].T @ (mass @ r0)
            print(f"first {m:3d} modes: {np.sum(coef ** 2) / tot:.0%} of the S^200 deviation (constant: {mass.diagonal().sum() * c0 @ c0 / tot:.0%})")


if __name__ == "__main__":
    main()
