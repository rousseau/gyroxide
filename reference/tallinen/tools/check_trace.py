"""Quick sanity check / loader for a tallinen_ref trace directory.

usage: python3 check_trace.py RUN_DIR
"""
import json, sys, csv
from pathlib import Path
import numpy as np

run = Path(sys.argv[1])
man = json.loads((run / "manifest.json").read_text())
print(man["preset"], man["counts"], "dt", man["params"]["dt"])
st = {p.stem: np.load(p) for p in sorted((run / "static").glob("*.npy"))}
for k, v in st.items():
    print(f"  static/{k}: {v.dtype} {v.shape}")
X0, T = st["X0"], st["tets"]
a, b, c = (X0[T[:, i]] - X0[T[:, 0]] for i in (1, 2, 3))
vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
print("  reference volume", vol.sum(), "min tet vol", vol.min())
for d in sorted(run.glob("step_*")):
    f = {p.stem: np.load(p) for p in d.glob("*.npy")}
    info = json.loads((d / "step.json").read_text())
    print(f"  {d.name}: t={info['t']:.6g} Ue={info['Ue']:.6g} |f_total|max={np.abs(f['f_total']).max():.3e} ninv={info['ninv']}")
rows = list(csv.DictReader(open(run / "scalars.csv")))
print("  scalars rows", len(rows), "last", {k: rows[-1][k] for k in ("step", "t", "Area", "Volume", "ninv")})
