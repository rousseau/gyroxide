"""Local refinement of a slab (tools/make_slab.py) where its tetrahedra degenerate in a fitted state.

Deepening sulci crush tetrahedra of the core 2-3.5 mm below the fundi (quality down to 0.01 in
SG 33 fits). Tetrahedra whose quality in the given state falls below --q-bad, and those sharing a
node with them, get their longest interior edge bisected (reference configuration), in every
tetrahedron around that edge, for --passes passes. Surface edges are never split (surface and atlas
correspondence unchanged), nor edges with both ends fixed (the cut faces keep their Dirichlet nodes).

New nodes are appended. Per-node data are extended by the mean of the two edge ends (positions,
tables, fixed positions; new nodes are free), per-tet data are inherited by the children, the owner
of a new node is that of the first edge end, slab_nodes is -1 for new nodes.

    python3 tools/refine_slab.py data/slab_front data/slab_front_ref out/slab_front/fit33_free
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_boundary_tets import extend, read_mesh, write_mesh  # noqa: E402


def quality(p, T):
    """Mean-ratio quality 12 (3|V|)^(2/3) / sum l^2, signed by the reference orientation (V < 0)."""
    q = p[T]
    e = [q[:, j] - q[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    l2 = sum((v ** 2).sum(1) for v in e)
    vol = np.einsum("ij,ij->i", e[0], np.cross(e[1], e[2])) / 6
    return -12 * np.sign(vol) * (3 * np.abs(vol)) ** (2 / 3) / l2


def bisect(X, T, edges):
    """Bisect the edges in order, in every tetrahedron around each; returns the new tetrahedra and
    their parents."""
    tets = [list(t) for t in T]
    parent = list(range(len(T)))
    around = defaultdict(set)
    for k, t in enumerate(tets):
        for a in t:
            around[a].add(k)
    m = len(X)
    for u, v in edges:
        for k in list(around[u] & around[v]):
            t = tets[k]
            t1 = [m if a == v else a for a in t]   # replacing a vertex by a point of the edge
            t2 = [m if a == u else a for a in t]   # keeps the orientation
            tets[k] = t1
            tets.append(t2); parent.append(parent[k])
            k2 = len(tets) - 1
            around[v].discard(k); around[u].discard(k2)
            for a in t1:
                around[a].add(k)
            for a in t2:
                around[a].add(k2)
        m += 1
    return np.array(tets, dtype=np.int64), np.array(parent)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("run", type=Path, help="fitted run (last x_<k>.npy) or positions .npy")
    ap.add_argument("--q-bad", type=float, default=0.3)
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()
    src, dst = args.src, args.dst
    meta = json.loads((src / "slab.json").read_text())
    ns = meta["n_surface"]
    X, T, treg, F, freg = read_mesh(src / "slab.mesh")
    if args.run.suffix == ".npy":
        x = np.load(args.run)
    else:
        ks = [int(p.stem.split("_")[1]) for p in args.run.glob("x_*.npy") if p.stem.split("_")[1].isdigit()]
        x = np.load(args.run / f"x_{max(ks)}.npy")
    fixed = np.load(src / "fixed.npy") > 0.5
    surf_edges = {tuple(sorted(e)) for f in F for e in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2]))}

    all_edges, parent_all = [], np.arange(len(T))
    Xc, Tc, xc, fixed_c = X, T, x, fixed.copy()
    for ps in range(args.passes):
        q = quality(xc, Tc)
        bad = q < args.q_bad
        nodes = np.unique(Tc[bad])
        mark = np.isin(Tc, nodes).any(1)
        cand = set()
        for t in Tc[mark]:
            best, lmax = None, -1.0
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = sorted((t[i], t[j]))
                    if (u, v) in surf_edges or (fixed_c[u] and fixed_c[v]):
                        continue
                    ln = np.linalg.norm(Xc[u] - Xc[v])
                    if ln > lmax:
                        best, lmax = (u, v), ln
            if best is not None:
                cand.add(best)
        edges = sorted(cand)
        T2, par = bisect(Xc, Tc, edges)
        Xc = extend(Xc, edges, 0)
        xc = extend(xc, edges, 0)
        fixed_c = np.concatenate([fixed_c, np.zeros(len(edges), bool)])
        print(f"pass {ps + 1}: {bad.sum()} tets below quality {args.q_bad}, {mark.sum()} marked, "
              f"{len(edges)} edges split -> {len(Xc)} nodes, {len(T2)} tets")
        parent_all = parent_all[par]
        Tc = T2
        all_edges += edges
    edges = np.array(all_edges, dtype=np.int64)
    V = lambda p, t: np.linalg.det(np.stack([p[t[:, k]] - p[t[:, 0]] for k in (1, 2, 3)], axis=2))
    v0, v2 = V(X, T), V(Xc, Tc)
    assert np.all(np.sign(v2) == np.sign(v0[parent_all])), "orientation changed"
    assert abs(abs(v2).sum() - abs(v0).sum()) < 1e-9 * abs(v0).sum(), "volume changed"
    q2 = quality(xc, Tc)
    print(f"quality in the given state: min {quality(x, T).min():.3f} -> {q2.min():.3f}, "
          f"n<0.1 {(quality(x, T) < 0.1).sum()} -> {(q2 < 0.1).sum()}")

    dst.mkdir(parents=True, exist_ok=True)
    write_mesh(dst / "slab.mesh", Xc, Tc, treg[parent_all], F, freg)
    meta.update(n_nodes=len(Xc), n_tets=len(Tc), refined_from=str(src), refined_edges=len(edges))
    (dst / "slab.json").write_text(json.dumps(meta, indent=2))
    np.save(dst / "refine_edges.npy", edges)
    np.save(dst / "refine_parent.npy", parent_all)
    for f in src.glob("slab.nodal_*.npy"):
        np.save(dst / f.name, extend(np.load(f), edges, 1))
    np.savetxt(dst / "slab.gr.txt", extend(np.loadtxt(src / "slab.gr.txt"), edges, 0), fmt="%.6f")
    np.save(dst / "fixed.npy", np.concatenate([np.load(src / "fixed.npy"), np.zeros(len(edges))]))
    np.save(dst / "slab_nodes.npy", np.concatenate([np.load(src / "slab_nodes.npy"), -np.ones(len(edges), np.int64)]))
    for f in src.glob("fixed_pos_w*.npy"):
        np.save(dst / f.name, extend(np.load(f), edges, 0))
    for f in src.glob("theta_init*.npy"):
        shutil.copy(f, dst / f.name)
    for ck in src.glob("ckpt_w*"):
        d = dst / ck.name
        d.mkdir(exist_ok=True)
        for f in ("x.npy", "v.npy"):
            np.save(d / f, extend(np.load(ck / f), edges, 0))
        np.save(d / "growth_prev.npy", np.load(ck / "growth_prev.npy")[parent_all])
        shutil.copy(ck / "state.json", d / "state.json")
    for w in (src / "prep").glob("w*"):
        d = dst / "prep" / w.name
        d.mkdir(parents=True, exist_ok=True)
        for f in w.iterdir():
            if f.name == "owner.npy":
                o = np.load(f)
                for u, _ in edges:
                    o = np.append(o, o[u])
                np.save(d / f.name, o)
            else:
                shutil.copy(f, d / f.name)


if __name__ == "__main__":
    main()
