"""Identifiability of the field-law coefficients of a slab fit (step 2 of the plan): Jacobian of the
equilibrium surface with respect to each coefficient by central differences (a fixed at its fitted
value), on the loss vertices, in mm per unit coefficient; prints its singular values, the condition
number and the cosines between columns (is the thickening c separable from the core growth k?).

    python3 tools/law_identifiability.py cs out/slab_cs/fit27_lawzero [--h 0.2]

Each perturbed equilibrium starts from the fitted one (adjoint_week --multiscale 0:0, no fit).
Writes <run>/jacobian.npy ([n_coef, n_vertices, 3], mm).
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["c:d30", "c:d3o", "c:1", "k:d30", "k:d3o", "k:1"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slab")
    ap.add_argument("run", type=Path)
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--bin", default="target/mu/adjoint_week_lawk")
    ap.add_argument("--bases", default="basis_d30_w27_s10,basis_d3o_w27_s10,basis_one_s10")
    args = ap.parse_args()
    R, run = args.slab, ROOT / args.run
    D = ROOT / "data" / f"slab_{R}"
    meta = json.loads((D / "slab.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    m = np.load(D / "prep/w27/cortex.npy")[:ns] > 0.5
    fixed = np.load(D / "fixed.npy").astype(bool)[:ns]
    sel = m & ~fixed
    theta = np.load(run / "theta_final.npy")
    na = len(theta) - 6
    ks = [int(p.stem.split("_")[1]) for p in run.glob("x_*.npy") if p.stem.split("_")[1].isdigit()]
    tmp = Path(tempfile.mkdtemp())
    ck = tmp / "ck"
    shutil.copytree(D / "ckpt_w27", ck)
    shutil.copy(run / f"x_{max(ks)}.npy", ck / "x.npy")
    L = ROOT / "out" / f"slab_{R}" / "law"
    bases = ",".join(str(L / f"{b}.npy") for b in args.bases.split(","))

    def equilibrium(th, tag):
        np.save(tmp / "theta.npy", th)
        out = tmp / tag
        cmd = [args.bin, f"configs/mu_test/mu2_slab_{R}.toml", str(D / "slab.mesh"), str(ck), str(D / "prep/w27"), str(out),
               "--absolute", "--multiscale", "0:0", "--theta0-table", "--relax", "0", "--relax0", "0", "--newton", "40",
               "--min-growth", "0", "--growth-band", "0.5", "--fit-fields", "c,k", "--field-band", "50", "--freeze-fields",
               "--dirichlet", str(D / "fixed.npy"), "--dirichlet-pos", str(D / "fixed_pos_w27.npy"),
               "--field-law", bases, "--theta-init", str(tmp / "theta.npy")]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return np.load(out / "x_0.npy")[:ns][sel] * mm

    J = []
    for j in range(6):
        cols = []
        for s in (1, -1):
            th = theta.copy()
            th[na + j] += s * args.h
            cols.append(equilibrium(th, f"{j}_{s}"))
        J.append((cols[0] - cols[1]) / (2 * args.h))
        print(f"{NAMES[j]}: |dx/dalpha| rms {np.sqrt((J[-1] ** 2).sum(1).mean()):.3f} mm per unit", flush=True)
    J = np.array(J)
    np.save(run / "jacobian.npy", J)
    A = J.reshape(6, -1).T / np.sqrt(sel.sum())          # rms-normalised columns: mm
    sv = np.linalg.svd(A, compute_uv=False)
    print("singular values (mm per unit):", np.array2string(sv, precision=3), f"condition {sv[0] / sv[-1]:.0f}")
    nrm = A / np.linalg.norm(A, axis=0)
    cos = nrm.T @ nrm
    print("cosines between columns:")
    print("       " + " ".join(f"{n:>6s}" for n in NAMES))
    for n, row in zip(NAMES, cos):
        print(f"{n:6s} " + " ".join(f"{v:+.2f}" for v in row))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
