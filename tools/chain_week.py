"""Inputs for chaining a slab fit (examples/adjoint_week.rs) from week w to week w+1: the growth
correction fitted at week w is carried to the later knots of the nodal a table, and the start state of
week w+1 is the fitted state of week w moved by the full-model displacement between the two weeks.

    python3 tools/chain_week.py data/slab_cs 27 out/slab_cs/fit27_common075 common075 [prev]

writes <slab>/slab.nodal_a.carry_<name>.npy, <slab>/ckpt_w<w+1>_from_<name>/ and
configs/mu_test/mu2_slab_<region>_carry_<name>.toml (region: slab directory name after "slab_").
With [prev], the fit of week w used the carried table of an earlier link <prev> (several weeks).
data/full (tools/make_full.py) is the whole hemisphere: its table is the mesh one and its config
configs/mu_test/mu2_full.toml.
"""
import shutil
import sys
from pathlib import Path

import numpy as np


def last_x(run):
    ks = [int(p.stem.split("_")[1]) for p in Path(run).glob("x_*.npy") if p.stem.split("_")[1].isdigit()]
    return np.load(Path(run) / f"x_{max(ks)}.npy")


def main():
    slab, w, run, name = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    row = w - 21                                    # one knot per week from SG 21
    full = slab.name == "full"
    table = Path("data/meshes/w21_left_610k_s.nodal_a.npy") if full else slab / "slab.nodal_a.npy"
    prev = sys.argv[5] if len(sys.argv) > 5 else None
    a = np.load(slab / f"slab.nodal_a.carry_{prev}.npy" if prev else table)
    own = np.load(slab / "prep" / f"w{w}" / "owner.npy").astype(int)
    theta = np.load(run / "theta_final.npy")
    na = int(own.max()) + 1                         # surface vertices: theta starts with a
    d = theta[:na][own] - a[row]
    carried = a.copy()
    carried[row:] += d
    np.save(slab / f"slab.nodal_a.carry_{name}.npy", carried)

    ck = slab / f"ckpt_w{w + 1}_from_{name}"
    shutil.rmtree(ck, ignore_errors=True)
    shutil.copytree(slab / f"ckpt_w{w + 1}", ck)
    x = last_x(run) + np.load(slab / f"ckpt_w{w + 1}" / "x.npy") - np.load(slab / f"ckpt_w{w}" / "x.npy")
    np.save(ck / "x.npy", np.ascontiguousarray(x))

    stem = "mu2_full" if full else f"mu2_slab_{slab.name.removeprefix('slab_')}"
    base = Path(f"configs/mu_test/{stem}.toml").read_text()
    old = f'a = "{table}"'
    assert old in base, old
    Path(f"configs/mu_test/{stem}_carry_{name}.toml").write_text(
        base.replace(old, f'a = "{slab}/slab.nodal_a.carry_{name}.npy"'))
    print(f"carried correction: mean {d.mean():+.3f}, max |d| {abs(d).max():.3f}; start {ck}")


if __name__ == "__main__":
    main()
