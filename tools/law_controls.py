"""Basis maps of the field-law controls on a slab (adjoint_week --field-law, step 2 of the plan):
depth maps computed on the whole atlas surface of a week (tools/make_full.py depth_bases) and
restricted to the slab, and the same SG 27 maps rotated on the atlas sphere (spin test). Each pair
d30, d3o is centred and scaled on the slab loss vertices, d3o made orthogonal to d30, / 10 like
out/slab_*/law/*_s10.npy.

    python3 tools/law_controls.py cs --weeks 24 26 27 --spins 2

writes out/slab_<name>/law/ctl_<tag>_{d30,d3o}_s10.npy with tag w24, w26, w27 (unrotated reference)
and spin<i>, and prints the correlation of each d30 with the slab's own SG 27 map.
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from make_full import depth_bases  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slab", help="cs or calc")
    ap.add_argument("--weeks", type=int, nargs="+", default=[24, 26, 27])
    ap.add_argument("--spins", type=int, default=2)
    ap.add_argument("--max-corr", type=float, default=0.2, help="largest |r| of a spun map with the unrotated one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = json.loads((ROOT / "data/meshes/w21_left_610k_s.json").read_text())
    ns_full, mm = meta["n_surface"], meta["maxd_mm"]
    slab = ROOT / "data" / f"slab_{args.slab}"
    ns = json.loads((slab / "slab.json").read_text())["n_surface"]
    nodes = np.load(slab / "slab_nodes.npy")[:ns]
    m = np.load(slab / "prep/w27/cortex.npy")[:ns] > 0.5
    out = ROOT / "out" / f"slab_{args.slab}" / "law"
    own = np.load(out / "basis_d30_w27.npy")

    def full_maps(w):
        p = ROOT / "data/adjoint_prep_s" / f"w{w}"
        cortex = np.load(p / "cortex.npy")[:ns_full] > 0.5
        return depth_bases(np.load(p / "target.npy")[:ns_full], np.load(p / "faces.npy").astype(int), cortex, mm)

    def save(tag, d30, d3o):
        a, b = d30[nodes], d3o[nodes]
        a = (a - a[m].mean()) / a[m].std()
        b = (b - b[m].mean()) / b[m].std()
        b -= (b[m] @ a[m]) / (a[m] @ a[m]) * a
        b /= b[m].std()
        np.save(out / f"ctl_{tag}_d30_s10.npy", a / 10)
        np.save(out / f"ctl_{tag}_d3o_s10.npy", b / 10)
        print(f"{args.slab} {tag}: corr of d30 with the slab SG 27 map {np.corrcoef(a[m], own[m])[0, 1]:+.2f}")
        return a

    ref = None
    for w in args.weeks:
        a = save(f"w{w}", *full_maps(w))
        if w == 27:
            ref = a
    d30, d3o = full_maps(27)
    sph = nib.load(ROOT / "data/atlas_dhcp/fetal.week27.left.sphere.surf.gii").darrays[0].data.astype(float)
    sph /= np.linalg.norm(sph, axis=1, keepdims=True)
    tree = cKDTree(sph)
    rng = np.random.default_rng(args.seed)
    i = 0
    while i < args.spins:
        r = Rotation.random(random_state=rng)
        nn = tree.query(sph @ r.as_matrix().T)[1]
        a = d30[nn][nodes]
        if abs(np.corrcoef(a[m], ref[m])[0, 1]) > args.max_corr:
            continue
        i += 1
        print(f"spin{i}: rotation of {np.degrees(r.magnitude()):.0f} deg")
        save(f"spin{i}", d30[nn], d3o[nn])


if __name__ == "__main__":
    main()


def self_maps(name, week=27):
    """d30, d3o of the model's own surface (slab start checkpoint of the week, no atlas
    information), oriented like the atlas maps: tag self<week>."""
    import scipy.sparse as sp
    slab = ROOT / "data" / f"slab_{name}"
    meta = json.loads((slab / "slab.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    p = slab / "prep" / f"w{week}"
    f = np.load(p / "faces.npy").astype(int)
    m = np.load(p / "cortex.npy")[:ns] > 0.5
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    s = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a

    def smooth(v, k):
        for _ in range(k):
            v = s @ v
        return v

    def normals(q):
        fn = np.cross(q[f[:, 1]] - q[f[:, 0]], q[f[:, 2]] - q[f[:, 0]])
        vn = np.zeros_like(q)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        return vn / np.linalg.norm(vn, axis=1, keepdims=True)

    def depth(y, k, after):
        ys = smooth(y, k)
        d = smooth(np.einsum("ij,ij->i", y - ys, normals(ys)) * mm, after)
        return (d - d[m].mean()) / d[m].std()

    out = ROOT / "out" / f"slab_{name}" / "law"
    own = np.load(out / f"basis_d30_w{week}.npy")
    sgn = np.sign(np.corrcoef(depth(np.load(p / "target.npy")[:ns], 30, 10)[m], own[m])[0, 1])
    x = np.load(slab / f"ckpt_w{week}" / "x.npy")[:ns]
    d30, fine = sgn * depth(x, 30, 10), sgn * depth(x, 3, 1)
    fine -= (fine[m] @ d30[m]) / (d30[m] @ d30[m]) * d30
    fine /= fine[m].std()
    np.save(out / f"ctl_self{week}_d30_s10.npy", d30 / 10)
    np.save(out / f"ctl_self{week}_d3o_s10.npy", fine / 10)
    print(f"{name} self{week}: corr of d30 with the atlas SG {week} map {np.corrcoef(d30[m], own[m])[0, 1]:+.2f}")
