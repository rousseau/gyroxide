"""Sub-mesh (slab) around a cortical region, for fast local fits with examples/adjoint_week.rs.

The slab keeps the tetrahedra whose nodes lie under the region (nearest surface node in the region)
and less than `--depth` mm below the surface (reference mesh). Only its pial faces are declared as
surface (cortical layer, contact, growth normals); the nodes of the cut faces (sides and bottom)
are fixed (Dirichlet) at the positions of a full-model run, week by week.

Atlas targets are expressed in the frame and at the size of the full model: for every week the
similarity (scale, rotation, translation) best mapping the atlas cortex onto the model cortex is
applied to the atlas, so that the loss sees the folds, not the global growth.

    python3 tools/make_slab.py --region region.npy --out data/slab_cs --weeks 26 27 28 \\
        --full out/remodel_a0/w{w}/eq/x_released.npy

writes <out>/slab.mesh, slab.json, slab.nodal_{a,h,core}.npy, slab.gr.txt, slab_nodes.npy (slab ->
full node), fixed.npy, fixed_pos_w{w}.npy, ckpt_w{w}/ (full run of week w) and
prep/w{w}/ (target, cortex = loss mask, edges, faces, owner, theta0).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent


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


def similarity(a, b):
    """s, R, t minimising |s R a + t - b| (rows)."""
    ca, cb = a.mean(0), b.mean(0)
    u, sv, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    s = (sv * np.diag(d)).sum() / ((a - ca) ** 2).sum()
    return s, r, cb - s * ca @ r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default="data/meshes/w21_left_610k_s.mesh")
    ap.add_argument("--prep", default="data/adjoint_prep_s")
    ap.add_argument("--region", required=True, help="surface vertex indices of the region (npy)")
    ap.add_argument("--depth", type=float, default=5.0, help="slab depth below the surface, mm (reference mesh)")
    ap.add_argument("--loss-margin", type=float, default=1.5, help="loss vertices at least this far from fixed nodes, mm")
    ap.add_argument("--full", required=True, help="full-model positions per week, with {w}")
    ap.add_argument("--weeks", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / args.mesh).with_suffix(".json").read_text())
    ns_full, mm = meta["n_surface"], meta["maxd_mm"]
    X, T, treg, F, freg = read_mesh(ROOT / args.mesh)
    region = np.zeros(ns_full, bool)
    region[np.load(args.region)] = True

    depth, near = cKDTree(X[:ns_full]).query(X)
    keep_node = region[near] & (depth * mm < args.depth)
    tk = keep_node[T].all(1)
    Ts = T[tk]
    used = np.unique(Ts)
    pial_faces = F[np.isin(F, used).all(1)]
    pial = np.unique(pial_faces)
    pial = pial[pial < ns_full]
    pial_faces = pial_faces[np.isin(pial_faces, pial).all(1)]
    interior = np.setdiff1d(used, pial)
    nodes = np.concatenate([pial, interior])          # slab -> full; pial (surface) nodes first
    new = -np.ones(len(X), np.int64)
    new[nodes] = np.arange(len(nodes))
    T2, F2 = new[Ts], new[pial_faces]
    ns = len(pial)

    # cut faces: tetrahedron faces seen once that are not pial faces -> their nodes are fixed
    fk = np.sort(np.concatenate([T2[:, [0, 1, 2]], T2[:, [0, 1, 3]], T2[:, [0, 2, 3]], T2[:, [1, 2, 3]]]), 1)
    uniq, cnt = np.unique(fk, axis=0, return_counts=True)
    boundary = uniq[cnt == 1]
    pial_set = {tuple(f) for f in np.sort(F2, 1)}
    cut = np.array([f for f in boundary if tuple(f) not in pial_set])
    fixed = np.zeros(len(nodes), bool)
    fixed[np.unique(cut)] = True
    np.save(out / "fixed.npy", fixed.astype(np.float64))
    np.save(out / "slab_nodes.npy", nodes)

    with open(out / "slab.mesh", "w") as f:
        f.write(f"{len(nodes)}\n")
        np.savetxt(f, X[nodes], fmt="%.10g")
        f.write(f"{len(T2)}\n")
        np.savetxt(f, np.column_stack([treg[tk], T2 + 1]), fmt="%d")
        f.write(f"{len(F2)}\n")
        np.savetxt(f, np.column_stack([np.ones(len(F2), int), F2 + 1]), fmt="%d")
    m2 = dict(meta, n_surface=int(ns), n_nodes=int(len(nodes)), n_tets=int(len(T2)), slab_of=Path(args.mesh).name,
              slab_depth_mm=args.depth, n_fixed=int(fixed.sum()))
    (out / "slab.json").write_text(json.dumps(m2, indent=2))
    stem = str(ROOT / args.mesh)[:-5]
    for kind in ("a", "h", "core"):
        np.save(out / f"slab.nodal_{kind}.npy", np.ascontiguousarray(np.load(f"{stem}.nodal_{kind}.npy")[:, nodes]))
    np.savetxt(out / "slab.gr.txt", np.loadtxt(f"{stem}.gr.txt")[nodes], fmt="%.6f")

    # loss mask: pial, not fixed, at least loss-margin mm from every fixed node (reference)
    dfix = cKDTree(X[nodes][fixed]).query(X[nodes][:ns])[0] * mm
    cortex_full = np.load(ROOT / args.prep / f"w{args.weeks[0]}" / "cortex.npy")[:ns_full] > 0.5
    loss = (~fixed[:ns]) & (dfix > args.loss_margin) & cortex_full[pial]
    sedges = np.unique(np.sort(np.concatenate([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [0, 2]]]), 1), axis=0)
    owner_full = np.load(ROOT / args.prep / f"w{args.weeks[0]}" / "owner.npy").astype(np.int64)
    pial_pos = cKDTree(X[pial])
    own = new[owner_full[nodes]]
    bad = (own < 0) | (own >= ns)
    own[bad] = pial_pos.query(X[nodes][bad])[1]
    print(f"slab: {len(nodes)} nodes ({100 * len(nodes) / len(X):.1f} %), {len(T2)} tets ({100 * len(T2) / len(T):.1f} %), "
          f"{ns} pial vertices, {fixed.sum()} fixed nodes, {loss.sum()} loss vertices")

    for w in args.weeks:
        x_full = np.load(ROOT / args.full.format(w=w))
        np.save(out / f"fixed_pos_w{w}.npy", np.ascontiguousarray(x_full[nodes]))
        # start of week w: the full-model state of the same week (consistent with the fixed nodes;
        # a fit then moves the free nodes away from the measured-growth solution)
        ck = out / f"ckpt_w{w}"
        ck.mkdir(exist_ok=True)
        np.save(ck / "x.npy", np.ascontiguousarray(x_full[nodes]))
        np.save(ck / "v.npy", np.zeros((len(nodes), 3)))
        np.save(ck / "growth_prev.npy", np.zeros(len(T2)))
        (ck / "state.json").write_text(json.dumps({"step": 0, "t": (w - 21) / 15}))
        y = np.load(ROOT / args.prep / f"w{w}" / "target.npy")[:ns_full]
        s, r, t = similarity(y[cortex_full], x_full[:ns_full][cortex_full])
        yt = s * y @ r + t
        P = out / "prep" / f"w{w}"
        P.mkdir(parents=True, exist_ok=True)
        np.save(P / "target.npy", np.ascontiguousarray(yt[pial]))
        np.save(P / "cortex.npy", loss.astype(np.float64))
        np.save(P / "edges.npy", sedges.astype(np.float64))
        np.save(P / "faces.npy", F2.astype(np.float64))
        np.save(P / "owner.npy", own.astype(np.float64))
        np.save(P / "theta0.npy", np.zeros(ns))
        print(f"SG {w}: atlas -> model similarity scale {s:.3f}")


if __name__ == "__main__":
    main()
