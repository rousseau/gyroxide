"""Universal scaling law of cortical folding (Mota & Herculano-Houzel 2015, Science 349:74):
T^(1/2) A_t = k A_e^(5/4), with T the mean cortical thickness, A_t the total (pial) area and A_e the
exposed area (convex hull). Evaluated on the dHCP atlas and on each model step, week by week:
slope of log10(T^(1/2) A_t) against log10(A_e), and offset log10 k at slope 5/4.

Atlas: A_t, A_e from the pial surface (data/adjoint_prep_s/w<w>/target.npy, unscaled; SG 21: the mesh surface), T the
area-weighted mean of the measured pial-white distance (mesh nodal_h table). Models: A_t, A_e from
the outer surface, T = volume of the cortical layer (tetrahedra weighted by the model's cortical weight) /
outer area of the cortex, calibrated on SG 21 (the estimator on the SG 21 mesh, 1.32 mm, against the
atlas thickness, 1.43 mm: x 1.08 for every model). Areas are those of the cortex vertices only (medial wall excluded) for A_t and T; A_e is
the hull area of the whole hemisphere surface, the same way for the atlas and the models.

    python3 tools/mota_law.py out/mota_law.json [--plot out/mota_law.png]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parent.parent
MESHES = {n: ROOT / f"data/meshes/{n}.mesh" for n in ("w21_left_610k_s", "w21_left_610k")}
WEEKS = range(21, 37)


def last_x(run):
    p = ROOT / run
    if p.suffix == ".npy":
        return np.load(p)
    if (p / "x_stress.npy").exists():
        return np.load(p / "x_stress.npy")
    ks = [int(q.stem.split("_")[1]) for q in p.glob("x_*.npy") if q.stem.split("_")[1].isdigit()]
    return np.load(p / f"x_{max(ks)}.npy")


# Model steps: label -> week -> state (positions in Tallinen's frame).
def _fit(w):
    if w == 27:
        return "out/full/fit27_lb40"
    return f"out/full/fit{w}_lbm" if w <= 31 else f"out/full/fit{w}_lbmf"


STEPS = {
    "0 · Tallinen 2016": {w: f"out/fwd_tallinen_s/w{w}/checkpoint/x.npy" for w in range(22, 28)},
    "2 · croissance mesurée (cœur)": {w: f"out/remodel_a0/w{w}/eq/x_released.npy" for w in range(22, 37)},
    "2 · idem, matériau réaliste": {w: f"out/real/ref/w{w}/eq/x_released.npy" for w in range(22, 37)},
    "3 · tangentielle ajustée": {w: f"out/chain_eq/w{w}/ckpt/x.npy" for w in range(23, 27)},
    "5 · loi + cœur lobaire": {w: _fit(w) for w in range(27, 37)},
    "5 · idem, matériau réaliste": {w: (f"out/material/fit{w}_1_50" if w != 27 else "out/real/fit27_lbm")
                                     for w in range(27, 37)},
}


class Mesh:
    """Tetrahedra and their cortical weight, as in the model (crates/gyroxide-model/src/growth.rs):
    g = 1 / (1 + exp(10 (d / H - 1))), d the reference distance of the centroid to the nearest
    surface vertex, H the SG 21 thickness there."""

    def __init__(self, path, ns, h21):
        tok = path.read_text().split()
        n = int(tok[0])
        self.x0 = np.array(tok[1:1 + 3 * n], dtype=float).reshape(n, 3)
        nt = int(tok[1 + 3 * n])
        self.tets = np.array(tok[2 + 3 * n:2 + 3 * n + 5 * nt], dtype=np.int64).reshape(nt, 5)[:, 1:] - 1
        d, i = cKDTree(self.x0[:ns]).query(self.x0[self.tets].mean(1))
        self.g = 1 / (1 + np.exp(np.clip(10 * (d / h21[i] - 1), -50, 50)))


def n_nodes(path):
    with open(path) as f:
        return int(f.readline().split()[0])


def tri_areas(q, f):
    return 0.5 * np.linalg.norm(np.cross(q[f[:, 1]] - q[f[:, 0]], q[f[:, 2]] - q[f[:, 0]]), axis=1)


def vertex_areas(q, f, n):
    a = tri_areas(q, f) / 3
    va = np.zeros(n)
    for k in range(3):
        np.add.at(va, f[:, k], a)
    return va


def tet_volumes(x, t):
    e = np.stack([x[t[:, k]] - x[t[:, 0]] for k in (1, 2, 3)], 1)
    return np.abs(np.linalg.det(e)) / 6


def quantities(q, faces, cortex, mm):
    """A_t (cortex), A_e (hull of the whole surface), in mm^2."""
    va = vertex_areas(q, faces, len(q)) * mm ** 2
    hull = ConvexHull(q * mm)
    return va[cortex].sum(), hull.area, va


def fit(log_ae, log_y):
    slope, icpt = np.polyfit(log_ae, log_y, 1)
    return slope, float(np.mean(log_y - 1.25 * log_ae))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--plot")
    args = ap.parse_args()
    meta = json.loads((ROOT / "data/meshes/w21_left_610k_s.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    faces = np.load(ROOT / "data/adjoint_prep_s/w22/faces.npy").astype(int)
    cortex = np.load(ROOT / "data/adjoint_prep_s/w22/cortex.npy")[:ns] > 0.5
    H = np.load(ROOT / "data/meshes/w21_left_610k_s.nodal_h.npy")[:, :ns] * mm
    h21 = H[0] / mm                 # Tallinen units, as the mesh
    meshes = {"w21_left_610k_s": Mesh(MESHES["w21_left_610k_s"], ns, h21)}

    res = {"atlas": {}}
    for w in WEEKS:     # SG 21: the mesh surface itself (built from the atlas pial surface)
        y = (meshes["w21_left_610k_s"].x0[:ns] if w == 21
             else np.load(ROOT / f"data/adjoint_prep_s/w{w}/target.npy")[:ns])
        at, ae, va = quantities(y, faces, cortex, mm)
        t = float((H[w - 21] * va)[cortex].sum() / va[cortex].sum())
        res["atlas"][w] = {"A_t": at, "A_e": ae, "T": t}

    def model_T(x, mesh, va):
        v = (tet_volumes(x * mm, mesh.tets) * mesh.g).sum()
        return v / va[cortex].sum()

    m = meshes["w21_left_610k_s"]
    _, _, va = quantities(m.x0[:ns], faces, cortex, mm)
    t21 = model_T(m.x0, m, va)
    calib = res["atlas"][21]["T"] / t21
    res["check_sg21"] = {"T_volume_estimator": t21, "T_atlas": res["atlas"][21]["T"], "calibration": calib}

    for label, runs in STEPS.items():
        res[label] = {}
        for w, run in runs.items():
            try:
                x = last_x(run)
            except (FileNotFoundError, ValueError):
                continue
            if not np.isfinite(x).all():
                print(f"{label}, SG {w}: non-finite positions ({run}), skipped")
                continue
            key = next(k for k, p in MESHES.items() if n_nodes(p) == len(x))
            if key not in meshes:
                meshes[key] = Mesh(MESHES[key], ns, h21)
            at, ae, va = quantities(x[:ns], faces, cortex, mm)
            res[label][w] = {"A_t": at, "A_e": ae, "T": calib * model_T(x, meshes[key], va)}


    print(f"SG 21 thickness: layer volume / area {res['check_sg21']['T_volume_estimator']:.3f} mm, "
          f"atlas {res['check_sg21']['T_atlas']:.3f} mm")
    summary = {}
    for label, d in res.items():
        if label == "check_sg21" or len(d) < 3:
            continue
        ws = sorted(d)
        lae = np.log10([d[w]["A_e"] for w in ws])
        ly = np.log10([d[w]["A_t"] * np.sqrt(d[w]["T"]) for w in ws])
        s, k = fit(lae, ly)
        summary[label] = {"weeks": [ws[0], ws[-1]], "slope": s, "log10_k": k}
        print(f"{label:34s} SG {ws[0]}-{ws[-1]}: slope {s:.3f}, log10 k {k:+.3f}")
    # per week offset from the atlas line: log10(T^1/2 A_t) model - atlas law at the model's A_e
    sa, ka = np.polyfit(np.log10([res['atlas'][w]['A_e'] for w in WEEKS]),
                        np.log10([res['atlas'][w]['A_t'] * np.sqrt(res['atlas'][w]['T']) for w in WEEKS]), 1)
    for label, d in res.items():
        if label in ("atlas", "check_sg21"):
            continue
        for w in d:
            d[w]["offset_vs_atlas"] = float(np.log10(d[w]["A_t"] * np.sqrt(d[w]["T"]))
                                            - (sa * np.log10(d[w]["A_e"]) + ka))
    print("offset from the atlas line (log10 units; +0.01 = +2.3 %):")
    for label, d in res.items():
        if label in ("atlas", "check_sg21"):
            continue
        print(f"  {label:32s} " + " ".join(f"{w}:{d[w]['offset_vs_atlas']:+.3f}" for w in sorted(d)))
    print("ratios to the atlas, A_t / A_e / T, at SG 26, 30, 36:")
    for label, d in res.items():
        if label in ("atlas", "check_sg21"):
            continue
        cells = []
        for w in (26, 30, 36):
            if w in d:
                a = res["atlas"][w]
                cells.append(f"{w}: {d[w]['A_t'] / a['A_t']:.3f} / {d[w]['A_e'] / a['A_e']:.3f} / {d[w]['T'] / a['T']:.3f}")
        print(f"  {label:32s} " + "   ".join(cells))
    Path(ROOT / args.out).write_text(json.dumps({"weeks": res, "fits": summary}, indent=1, default=float))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
        for label, d in res.items():
            if label == "check_sg21":
                continue
            ws = sorted(d)
            ax.plot([np.log10(d[w]["A_e"]) for w in ws], [np.log10(d[w]["A_t"] * np.sqrt(d[w]["T"])) for w in ws],
                    "k-o" if label == "atlas" else "-", ms=3, lw=2.5 if label == "atlas" else 1.5, label=label)
        ax.set_xlabel("log10 A_e (mm²)"); ax.set_ylabel("log10 T^½ A_t")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT / args.plot)


if __name__ == "__main__":
    main()
