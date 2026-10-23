"""Inputs of whole-hemisphere fits with examples/adjoint_week.rs, laid out like a slab
(tools/make_slab.py) without fixed nodes: start checkpoints of a full-model run and atlas targets
mapped by a similarity onto the model cortex (the loss sees the folds, not the global growth), plus
the depth maps of the field law (tools: --field-law), centred and scaled on the cortex, / 10.

    python3 tools/make_full.py --out data/full --weeks 26 27 28 29 30 \\
        --full out/remodel_a0/w{w}/eq/x_released.npy

writes <out>/ckpt_w{w}/, prep/w{w}/ (target, cortex, edges, faces, owner, theta0) and
law/basis_{d30,d3o}_w{w}_s10.npy, law/basis_one_s10.npy.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_slab import similarity  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def depth_bases(y, faces, cortex, mm):
    """d30 = (y - S^30 y).n smoothed by S^10 and d_fine = (y - S^3 y).n smoothed by S, made orthogonal
    to d30; both centred and scaled on the cortex."""
    ns = len(y)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    s = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a

    def smooth(v, k):
        for _ in range(k):
            v = s @ v
        return v

    def normals(p):
        fn = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
        vn = np.zeros_like(p)
        for k in range(3):
            np.add.at(vn, faces[:, k], fn)
        return vn / np.linalg.norm(vn, axis=1, keepdims=True)

    def depth(k, after):
        ys = smooth(y, k)
        d = smooth(np.einsum("ij,ij->i", y - ys, normals(ys)) * mm, after)
        return (d - d[cortex].mean()) / d[cortex].std()

    # depth < 0 at fundi: normals pointing out of the hemisphere, whatever the face orientation
    out_frac = np.mean(np.einsum("ij,ij->i", normals(y), y - y.mean(0)) > 0)
    if out_frac < 0.5:
        faces = faces[:, ::-1]
    d30, fine = depth(30, 10), depth(3, 1)
    fine -= (fine[cortex] @ d30[cortex]) / (d30[cortex] @ d30[cortex]) * d30
    return d30, fine / fine[cortex].std()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default="data/meshes/w21_left_610k_s.mesh")
    ap.add_argument("--prep", default="data/adjoint_prep_s")
    ap.add_argument("--full", required=True, help="full-model positions per week, with {w}")
    ap.add_argument("--weeks", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "law").mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / args.mesh).with_suffix(".json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    for w in args.weeks:
        src = ROOT / args.prep / f"w{w}"
        x = np.load(ROOT / args.full.format(w=w))
        ck = out / f"ckpt_w{w}"
        ck.mkdir(parents=True, exist_ok=True)
        np.save(ck / "x.npy", np.ascontiguousarray(x))
        np.save(ck / "v.npy", np.zeros_like(x))
        np.save(ck / "growth_prev.npy", np.zeros(meta["n_tets"]))
        (ck / "state.json").write_text(json.dumps({"step": 0, "t": (w - 21) / 15}))
        p = out / "prep" / f"w{w}"
        p.mkdir(parents=True, exist_ok=True)
        for f in ("cortex", "edges", "faces", "owner", "theta0"):
            shutil.copy(src / f"{f}.npy", p / f"{f}.npy")
        cortex = np.load(src / "cortex.npy")[:ns] > 0.5
        y = np.load(src / "target.npy")
        s, r, t = similarity(y[:ns][cortex], x[:ns][cortex])
        np.save(p / "target.npy", np.ascontiguousarray(s * y @ r + t))
        d30, fine = depth_bases(y[:ns], np.load(src / "faces.npy").astype(int), cortex, mm)
        np.save(out / "law" / f"basis_d30_w{w}_s10.npy", d30 / 10)
        np.save(out / "law" / f"basis_d3o_w{w}_s10.npy", fine / 10)
        print(f"SG {w}: atlas -> model similarity scale {s:.3f}")
    np.save(out / "law" / "basis_one_s10.npy", np.full(ns, 0.1))


if __name__ == "__main__":
    main()
