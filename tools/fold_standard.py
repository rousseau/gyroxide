"""Standard, phase-insensitive folding metrics of whole-hemisphere fits against the dHCP atlas
(step 3 of the plan): gyrification index A / A(convex hull), band power of the mean curvature at
dyadic scales (a heat-diffusion analogue of the SpAnGy bands: H filtered at scale s minus H filtered
at 2s, s = 1, 2, 4, 8, 16 mm), and the similarity of the shape-index distributions (histogram
intersection, 1 = identical). Curvatures on the cortex, cotangent Laplacian, angle-deficit Gaussian
curvature, both lightly smoothed (S^2) the same way for the model and the atlas.

    python3 tools/fold_standard.py out/fold_standard.json [label=week:run ...]

Without runs, evaluates the chains of tools/full_figure.py at SG 27-30. A run is a fit directory
(last x_<k>.npy) or a positions .npy. Prints model / atlas ratios.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from full_figure import CHAINS, WEEKS, state  # noqa: E402
from lobar_basis import cotan_laplacian  # noqa: E402

SCALES = [1, 2, 4, 8, 16, 32]   # mm; band k between SCALES[k] and SCALES[k + 1]


class Surface:
    def __init__(self, faces, cortex):
        self.faces, self.cortex, n = faces, cortex, len(cortex)
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        a = ((a + a.T) > 0).astype(float)
        self.s = 0.5 * sp.identity(n) + 0.5 * sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a

    def metrics(self, x):
        """x in mm. Returns GI, band powers, shape index."""
        f = self.faces
        lap, mass = cotan_laplacian(x, f)
        m = mass.diagonal()
        fn = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
        vn = np.zeros_like(x)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True)
        if np.mean(np.einsum("ij,ij->i", vn, x - x.mean(0)) > 0) < 0.5:
            vn = -vn
        h = 0.5 * np.einsum("ij,ij->i", (lap @ x) / m[:, None], vn)      # > 0 on crests
        ang = np.zeros(len(x))
        for k in range(3):
            u = x[f[:, (k + 1) % 3]] - x[f[:, k]]
            v = x[f[:, (k + 2) % 3]] - x[f[:, k]]
            c = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
            np.add.at(ang, f[:, k], np.arccos(np.clip(c, -1, 1)))
        kg = (2 * np.pi - ang) / m
        for _ in range(2):
            h, kg = self.s @ h, self.s @ kg
        c = self.cortex
        area = 0.5 * np.linalg.norm(fn, axis=1).sum()
        gi = area / ConvexHull(x).area
        bands, prev = [], None
        for sc in SCALES:
            t = sc ** 2 / 2 / 4          # heat time of scale sc, in 4 implicit sub-steps
            lu = splu((mass + t * lap).tocsc())
            u = h.copy()
            for _ in range(4):
                u = lu.solve(m * u)
            if prev is not None:
                b = prev - u
                bands.append(float(np.sum(m[c] * b[c] ** 2) / m[c].sum()))
            prev = u
        disc = np.sqrt(np.maximum(h ** 2 - kg, 1e-12))
        si = 2 / np.pi * np.arctan(h / disc)
        hist = np.histogram(si[c], bins=50, range=(-1, 1), weights=m[c])[0]
        return {"gi": float(gi), "bands": bands, "si_hist": (hist / hist.sum()).tolist()}


def compare(mod, atl):
    return {
        "gi_ratio": mod["gi"] / atl["gi"],
        "band_ratio": [a / b for a, b in zip(mod["bands"], atl["bands"])],
        "si_similarity": float(np.minimum(mod["si_hist"], atl["si_hist"]).sum()),
    }


def main():
    out = Path(sys.argv[1])
    meta = json.loads((ROOT / "data/meshes/w21_left_610k_s.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    runs = {}
    for a in sys.argv[2:]:
        label, spec = a.split("=", 1)
        w, r = spec.split(":", 1)
        runs.setdefault(label, {})[int(w)] = r
    if not runs:
        runs = {label: {w: run(w) for w in WEEKS} for label, (_, _, run) in CHAINS.items()}
    res, atlas = {}, {}
    weeks = sorted({w for r in runs.values() for w in r})
    for w in weeks:
        p = ROOT / "data/full/prep" / f"w{w}"
        surf = Surface(np.load(p / "faces.npy").astype(int), np.load(p / "cortex.npy")[:ns] > 0.5)
        atlas[w] = surf.metrics(np.load(p / "target.npy")[:ns] * mm)
        for label, r in runs.items():
            if w in r:
                res.setdefault(label, {})[w] = compare(surf.metrics(state(ROOT / r[w])[:ns] * mm), atlas[w])
    out.write_text(json.dumps({"scales_mm": SCALES, "atlas": atlas, "runs": res}, indent=1))
    print("bands (mm):", " ".join(f"{a}-{b}" for a, b in zip(SCALES, SCALES[1:])))
    for w in weeks:
        print(f"SG {w} atlas: GI {atlas[w]['gi']:.3f}, band power " + " ".join(f"{v:.2e}" for v in atlas[w]["bands"]))
        for label in res:
            if w in res[label]:
                c = res[label][w]
                print(f"  {label:20s} GI x{c['gi_ratio']:.3f}  bands x " + " ".join(f"{v:.2f}" for v in c["band_ratio"])
                      + f"  SI similarity {c['si_similarity']:.3f}")


if __name__ == "__main__":
    main()
