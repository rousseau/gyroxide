"""Depth maps of the model's own surface for the field law (tools/full_chain.sh with MAPS=self):
d30 and d_fine of a start checkpoint, computed like the atlas maps (tools/make_full.py depth_bases,
outward normals), / 100 like data/full/law/*_s100.npy. No atlas information enters the maps.

    python3 tools/self_maps.py data/full/ckpt_w28_from_lbself28/x.npy 28 data/full/law/basis_self_lbself28
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from make_full import depth_bases  # noqa: E402


def main():
    x, w, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    meta = json.loads((ROOT / "data/meshes/w21_left_610k_s.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    p = ROOT / "data/full/prep" / f"w{w}"
    cortex = np.load(p / "cortex.npy")[:ns] > 0.5
    d30, fine = depth_bases(np.load(x)[:ns], np.load(p / "faces.npy").astype(int), cortex, mm)
    atlas = np.load(ROOT / f"data/full/law/basis_d30_w{w}_s100.npy") * 100
    np.save(f"{out}_d30_s100.npy", d30 / 100)
    np.save(f"{out}_d3o_s100.npy", fine / 100)
    print(f"self maps SG {w}: corr of d30 with the atlas map of SG {w} {np.corrcoef(d30[cortex], atlas[cortex])[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
