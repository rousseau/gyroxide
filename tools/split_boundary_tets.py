"""Remove tetrahedra whose four nodes lie on the surface by splitting one of their interior edges.

A tetrahedron with four surface nodes (typically astride a surface edge, owning two boundary
faces) has no interior node: its volume is set by the dihedral angle of the surface at that edge
and goes through zero when the surface turns from convex to concave there, i.e. when a sulcus
forms across it. Bisecting its interior edge (u, v) with a new interior node m, in every
tetrahedron around that edge, leaves children that all have an interior node.

New nodes are appended (surface numbering unchanged). Per-node data are extended by the mean of
the two edge ends, per-tet data are inherited by the children. The split is exact for the
deformed states: the midpoint of a deformed edge is the image of the reference midpoint.

    python3 tools/split_boundary_tets.py data/meshes/w21_left_610k.mesh data/meshes/w21_left_610k_s.mesh \\
        --node-npy data/meshes/w21_left_610k.nodal_a.npy:1 ... --node-txt data/meshes/w21_left_610k.gr.txt \\
        --ckpt out/eq_thick/w28/ckpt:out/eq_thick_s/w28/ckpt --prep data/adjoint_prep:data/adjoint_prep_s

`--node-npy path:axis` extends the node axis `axis` of an .npy array; outputs are written next to
the new mesh with the same suffix (w21_left_610k.nodal_a.npy -> w21_left_610k_s.nodal_a.npy).
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_mesh(path):
    tok = Path(path).read_text().split()
    i = 0
    n = int(tok[i]); i += 1
    X = np.array(tok[i:i + 3 * n], dtype=float).reshape(n, 3); i += 3 * n
    nt = int(tok[i]); i += 1
    T = np.array(tok[i:i + 5 * nt], dtype=np.int64).reshape(nt, 5); i += 5 * nt
    nf = int(tok[i]); i += 1
    F = np.array(tok[i:i + 4 * nf], dtype=np.int64).reshape(nf, 4)
    return X, T[:, 1:] - 1, T[:, 0], F[:, 1:] - 1, F[:, 0]


def write_mesh(path, X, T, treg, F, freg):
    with open(path, "w") as f:
        f.write(f"{len(X)}\n")
        np.savetxt(f, X, fmt="%.10g")
        f.write(f"{len(T)}\n")
        np.savetxt(f, np.column_stack([treg, T + 1]), fmt="%d")
        f.write(f"{len(F)}\n")
        np.savetxt(f, np.column_stack([freg, F + 1]), fmt="%d")


def split(T, n_surface, faces):
    """Bisect one interior edge of every tetrahedron with four surface nodes. Returns the new
    tetrahedra, the parent of each, and the split edges (u, v) in the order of the new nodes."""
    surf_edges = {tuple(sorted(e)) for f in faces for e in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2]))}
    tets = [list(t) for t in T]
    parent = list(range(len(T)))
    around = defaultdict(set)
    for k, t in enumerate(tets):
        for a in t:
            around[a].add(k)
    n_next = int(T.max()) + 1
    edges = []
    for k0 in np.where((T < n_surface).all(1))[0]:
        t = tets[k0]
        if not all(a < n_surface for a in t):
            continue  # already split through a neighbour
        inner = [(t[i], t[j]) for i in range(4) for j in range(i + 1, 4) if tuple(sorted((t[i], t[j]))) not in surf_edges]
        if not inner:
            raise ValueError(f"tetrahedron {k0} has all six edges on the surface")
        u, v = inner[0]
        m = n_next; n_next += 1
        edges.append((u, v))
        for k in list(around[u] & around[v]):
            t = tets[k]
            t1 = [m if a == v else a for a in t]  # replacing a vertex by a point of the edge
            t2 = [m if a == u else a for a in t]  # keeps the orientation
            tets[k] = t1
            tets.append(t2); parent.append(parent[k])
            k2 = len(tets) - 1
            around[v].discard(k); around[u].discard(k2)
            for a in t1:
                around[a].add(k)
            for a in t2:
                around[a].add(k2)
    return np.array(tets, dtype=np.int64), np.array(parent), np.array(edges, dtype=np.int64)


def extend(a, edges, axis):
    """Append the mean of the edge ends along the node axis (applied in order: a new node may
    be an end of a later edge)."""
    a = np.moveaxis(np.asarray(a), axis, 0)
    out = np.concatenate([a, np.zeros((len(edges),) + a.shape[1:], dtype=a.dtype)])
    n0 = len(a)
    for k, (u, v) in enumerate(edges):
        out[n0 + k] = 0.5 * (out[u] + out[v])
    return np.ascontiguousarray(np.moveaxis(out, 0, axis))  # C order (the Rust npy reader)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh"); ap.add_argument("out")
    ap.add_argument("--node-npy", nargs="*", default=[], help="path:axis of per-node arrays")
    ap.add_argument("--node-txt", nargs="*", default=[], help="per-node text columns (growth mask)")
    ap.add_argument("--ckpt", nargs="*", default=[], help="src:dst checkpoint directories")
    ap.add_argument("--prep", default=None, help="src:dst adjoint prep root (owner.npy per week)")
    args = ap.parse_args()

    src, dst = Path(args.mesh), Path(args.out)
    meta = json.loads(src.with_suffix(".json").read_text())
    ns = meta["n_surface"]
    X, T, treg, F, freg = read_mesh(src)
    T2, parent, edges = split(T, ns, F)
    X2 = extend(X, edges, 0)
    V = lambda p, t: np.linalg.det(np.stack([p[t[:, k]] - p[t[:, 0]] for k in (1, 2, 3)], axis=2))
    v0, v2 = V(X, T), V(X2, T2)
    assert np.all(np.sign(v2) == np.sign(v0[parent])), "orientation changed"
    assert abs(abs(v2).sum() - abs(v0).sum()) < 1e-9 * abs(v0).sum(), "volume changed"
    left = int((T2 < ns).all(1).sum())
    print(f"{len(edges)} edges split: {len(X)} -> {len(X2)} nodes, {len(T)} -> {len(T2)} tets; all-surface tets left: {left}")
    write_mesh(dst, X2, T2, treg[parent], F, freg)
    meta.update(n_nodes=len(X2), n_tets=len(T2), split_from=src.name, split_edges=len(edges))
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    np.save(dst.with_suffix(".split_edges.npy"), edges)
    np.save(dst.with_suffix(".split_parent.npy"), parent)

    rename = lambda p: Path(p).parent / Path(p).name.replace(src.stem, dst.stem, 1)
    for spec in args.node_npy:
        p, ax = spec.rsplit(":", 1)
        np.save(rename(p), extend(np.load(p), edges, int(ax)))
    for p in args.node_txt:
        np.savetxt(rename(p), extend(np.loadtxt(p), edges, 0), fmt="%.6f")
    for spec in args.ckpt:
        s, d = map(Path, spec.split(":"))
        d.mkdir(parents=True, exist_ok=True)
        for f in ("x.npy", "v.npy"):
            np.save(d / f, extend(np.load(s / f), edges, 0))
        np.save(d / "growth_prev.npy", np.load(s / "growth_prev.npy")[parent])
        shutil.copy(s / "state.json", d / "state.json")
    if args.prep:
        s, d = map(Path, args.prep.split(":"))
        for w in sorted(s.glob("w*")):
            (d / w.name).mkdir(parents=True, exist_ok=True)
            for f in w.iterdir():
                if f.name == "owner.npy":
                    o = np.load(f)
                    np.save(d / w.name / f.name, np.concatenate([o, o[edges[:, 0]]]))
                else:
                    shutil.copy(f, d / w.name / f.name)


if __name__ == "__main__":
    main()
