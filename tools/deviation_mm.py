#!/usr/bin/env python3
"""Deviation in mm of simulated surfaces from the dHCP atlas, week by week, at several scales.

    python3 tools/deviation_mm.py <run dir> [--weeks 22-36] [--ckpt '{run}/w{w}/checkpoint']

For every week whose checkpoint exists (x.npy) and whose prep dir exists (data/adjoint_prep/w<w>,
tools/adjoint_prep.py): over the cortex vertices, |S^k x_i - S^k y_i| (S: neighbour averaging on the
surface, as in examples/adjoint_week.rs; k = 200 lobes, 50 primary folds, 10 secondary folds,
0 vertices), absolute (common frame), rigid (after the best rotation + translation, no scaling:
the placement of the hemisphere in the atlas frame is not a model property) and shape-only
(centred, rescaled to the atlas rms radius); RMS / p95 in mm, and the volume ratio model / atlas.
Writes <run>/deviation_mm.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surface_metrics import LEVELS, deviation_levels, smoother  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--weeks", default="22-36")
ap.add_argument("--ckpt", default="{run}/w{w}/checkpoint")
args = ap.parse_args()
w0, w1 = map(int, args.weeks.split("-"))
F = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
MM, NS = F["maxd_mm"], F["n_surface"]


def volume(p, tri):
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


out = {}
S = None
print(f"{'SG':>3} {'V/Vatlas':>8} | " + " | ".join(f"k={k:<3} abs  rigid (p95)  shape" for k in LEVELS))
for w in range(w0, w1 + 1):
    ck = Path(args.ckpt.format(run=args.run, w=w))
    prep = ROOT / f"data/adjoint_prep/w{w}"
    if not (ck / "x.npy").exists() or not prep.exists():
        continue
    if S is None:
        S = smoother(np.load(prep / "edges.npy"), NS)
    x = np.load(ck / "x.npy")[:NS]
    y = np.load(prep / "target.npy")
    cx = np.load(prep / "cortex.npy") > 0.5
    tri = np.load(prep / "faces.npy").astype(int)
    row = {"volume_ratio": volume(x, tri) / volume(y, tri)}
    row.update(deviation_levels(x, y, cx, S, MM))
    out[w] = row
    print(f"{w:3d} {row['volume_ratio']:8.3f} | " + " | ".join(
        f"{row[f'k{k}']['absolute_mm']['rms']:5.2f} {row[f'k{k}']['rigid_mm']['rms']:5.2f} ({row[f'k{k}']['rigid_mm']['p95']:5.2f}) "
        f"{row[f'k{k}']['shape_mm']['rms']:5.2f}" for k in LEVELS))
(args.run / "deviation_mm.json").write_text(json.dumps(out, indent=1))
