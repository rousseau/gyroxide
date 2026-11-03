#!/usr/bin/env python3
"""Week-by-week comparison of simulation runs with the dHCP atlas, for the summary deck.

    python3 tools/experiments_mm.py out/experiments_mm.json [run ...]

For each run (RUNS below; surfaces from the VTK files of `gyroxide run --vtk-every 8769`, or
checkpoints), per gestational week SG: rigid deviation (best rotation + translation, no scaling)
in mm over the cortex vertices at the vertex level (k = 0) and at the lobar scale (k = 200
neighbour averagings, as in examples/adjoint_week.rs), folding index FI = A / V^(2/3) of the
pial triangles and volume, both as ratios to the atlas of that week. Missing weeks (run stopped)
are left out.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
F = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
MM, NS = F["maxd_mm"], F["n_surface"]
WEEK = 8769
OUT = ROOT / "out"

RUNS = {
    "tallinen": {"ckpt": "fwd_tallinen/w{w}/checkpoint"},
    "global_fi": {"ckpt": "atlas_fit/w{w}/checkpoint"},
    **{f"scan_{d.name}": {"vtk": f"scan_hmu/{d.name}"} for d in sorted((OUT / "scan_hmu").glob("*")) if d.is_dir()},
    "measured_core_eq": {"ckpt": "eq_thick/w{w}/ckpt"},
    "local_fit": {"ckpt": "chain_eq/w{w}/ckpt", "ckpt27": "chain_eq/w27/ckpt_conv"},
}


def read_vtk_points(path):
    with open(path) as f:
        for line in f:
            if line.startswith("POINTS"):
                n = int(line.split()[1])
                break
        vals = np.fromstring(" ".join(next(f) for _ in range(n)), sep=" ")
    return vals.reshape(-1, 3)[:NS]


def surfaces(spec):
    if "vtk" in spec:
        for p in sorted((OUT / spec["vtk"]).glob("surface_*.vtk")):
            step = p.stem.split("_")[1]
            if step.isdigit() and int(step) % WEEK == 0 and int(step) > 0:
                yield 21 + int(step) // WEEK, lambda p=p: read_vtk_points(p)
    else:
        for w in range(22, 37):
            ck = OUT / (spec.get(f"ckpt{w}") or spec["ckpt"].format(w=w))
            if (ck / "x.npy").exists():
                yield w, lambda ck=ck: np.load(ck / "x.npy")[:NS]


prep = ROOT / "data/adjoint_prep/w22"
e = np.load(prep / "edges.npy").astype(int)
A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(NS, NS))
A = ((A + A.T) > 0).astype(float).tocsr()
S = 0.5 * sp.identity(NS) + 0.5 * sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A
TRI = np.load(prep / "faces.npy").astype(int)


def smooth(v, k):
    for _ in range(k):
        v = S @ v
    return v


def rigid(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt((np.linalg.norm(xc @ r - yc, axis=1) ** 2).mean()) * MM)


def area_volume(p):
    a, b, c = p[TRI[:, 0]], p[TRI[:, 1]], p[TRI[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    vol = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
    return area, vol


only = set(sys.argv[2:])
out = json.loads(Path(sys.argv[1]).read_text()) if only and Path(sys.argv[1]).exists() else {}
for name, spec in RUNS.items():
    if only and name not in only:
        continue
    rows = {}
    for w, load in surfaces(spec):
        pw = ROOT / f"data/adjoint_prep/w{w}"
        if not pw.exists():
            continue
        x, y = load(), np.load(pw / "target.npy")
        if not np.isfinite(x).all():
            continue
        cx = np.load(pw / "cortex.npy") > 0.5
        (ax, vx), (ay, vy) = area_volume(x), area_volume(y)
        rows[w] = {
            "rigid_k0": rigid(x[cx], y[cx]),
            "rigid_k200": rigid(smooth(x, 200)[cx], smooth(y, 200)[cx]),
            "fi_ratio": float((ax / vx ** (2 / 3)) / (ay / vy ** (2 / 3))),
            "fi": float(ax / vx ** (2 / 3)),
            "fi_atlas": float(ay / vy ** (2 / 3)),
            "volume_ratio": float(vx / vy),
        }
    out[name] = rows
    print(name, " ".join(f"{w}:{r['rigid_k0']:.2f}/{r['fi_ratio']:.2f}" for w, r in sorted(rows.items())))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
