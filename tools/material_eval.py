"""Equilibria of a fitted model under other materials (tools/stress_run.sh with MAT=...): rigid
deviation to the atlas cortex, fold depth, curvature correlation, and the elastic volume change J of
the cortex (median, share of tets with J < 0.7).

    python3 tools/material_eval.py 30 out/full/fit30_lbm out/material/c30_1_5 out/material/c30_1_50 ...
(a fit run directory is evaluated at its last iterate, without J)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_gallery import Surf, last_x, load_full, metrics, rigid_to, week_data  # noqa: E402


def main():
    w, runs = int(sys.argv[1]), sys.argv[2:]
    ns, mm = load_full()
    y, m, faces = week_data(w, ns)
    S = Surf(faces, y, mm)
    print(f"{'run':24s} | rigid mm | corr H | depth (atlas) | J cortex p50 | J < 0.7 | J core p50")
    for r in runs:
        stress = (Path(r) / "x_stress.npy").exists()       # else a fit run (out/full/fit*), its last iterate
        x = (np.load(Path(r) / "x_stress.npy") if stress else last_x(r))[:ns]
        q = metrics(S, rigid_to(x, y, m), y, m)
        line = f"{Path(r).name:24s} | {q['mm']:8.3f} | {q['corr']:6.3f} | {q['depth']:5.2f} ({q['depth_atlas']:.2f})"
        if stress:
            j, gm = np.load(Path(r) / "stress_j.npy"), np.load(Path(r) / "stress_gm.npy")
            jc, jw = j[gm > 0.5], j[gm < 0.5]
            line += f" | {np.median(jc):12.3f} | {(jc < 0.7).mean() * 100:5.1f} % | {np.median(jw):10.3f}"
        print(line)


if __name__ == "__main__":
    main()
