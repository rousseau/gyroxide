"""Area budget of the outer surface: is the atlas area prescribed by a(t) realized by the model?
Per cortex vertex, three areas at week w:
  atlas      the vertex area of the atlas pial surface (data/adjoint_prep_s/w<w>/target.npy);
  prescribed the SG 21 area grown by the model's tangential growth, (1 + g_m a)^2 per surface triangle,
             g_m and a those of the tetrahedron owning the triangle (crates/gyroxide-model/src/growth.rs:
             a the mean of its 4 nodes, g_m = gr / (1 + exp(10 (d / H21 - 1))), d its centroid's distance
             to the surface);
  realized   the vertex area of the model's outer surface.
prescribed / atlas measures the growth law (cortical weight, mask, averaging over the tetrahedron);
realized / prescribed the elastic tangential stretch of the surface (< 1: compression).

    python3 tools/area_budget.py [out/area_budget.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mota_law import MESHES, ROOT, tri_areas, vertex_areas  # noqa: E402

RUNS = {
    "croissance mesurée (cœur)": "out/remodel_a0/w{w}/eq/x_released.npy",
    "idem, matériau réaliste": "out/real/ref/w{w}/eq/x_released.npy",
}


def main():
    meta = json.loads((ROOT / "data/meshes/w21_left_610k_s.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    tok = MESHES["w21_left_610k_s"].read_text().split()
    n = int(tok[0])
    x0 = np.array(tok[1:1 + 3 * n], dtype=float).reshape(n, 3)
    nt = int(tok[1 + 3 * n])
    tets = np.array(tok[2 + 3 * n:2 + 3 * n + 5 * nt], dtype=np.int64).reshape(nt, 5)[:, 1:] - 1
    faces = np.load(ROOT / "data/adjoint_prep_s/w22/faces.npy").astype(np.int64)
    cortex = np.load(ROOT / "data/adjoint_prep_s/w22/cortex.npy")[:ns] > 0.5
    A = np.load(ROOT / "data/meshes/w21_left_610k_s.nodal_a.npy")
    h21 = np.load(ROOT / "data/meshes/w21_left_610k_s.nodal_h.npy")[0]
    gr = np.loadtxt(ROOT / "data/meshes/w21_left_610k_s.gr.txt")

    # owning tetrahedron of each surface triangle
    key = lambda t: (t[:, 0] * n + t[:, 1]) * n + t[:, 2]
    tf = np.concatenate([np.sort(tets[:, c], 1) for c in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])])
    tid = np.tile(np.arange(nt), 4)
    order = np.argsort(key(tf))
    kf = key(np.sort(faces, 1))
    pos = np.searchsorted(key(tf)[order], kf)
    owner = tid[order][pos]
    assert (key(tf)[order][pos] == kf).all()
    d, i = cKDTree(x0[:ns]).query(x0[tets[owner]].mean(1))
    g = gr[tets[owner]].mean(1) / (1 + np.exp(np.clip(10 * (d / h21[i] - 1), -50, 50)))

    a21 = tri_areas(x0[:ns], faces)
    res = {}
    print("cortex totals, ratios: prescribed/atlas, realized/prescribed, realized/atlas; "
          "realized/prescribed p10 p50 p90 per vertex")
    for w in range(22, 37):
        y = np.load(ROOT / f"data/adjoint_prep_s/w{w}/target.npy")[:ns]
        va_atlas = vertex_areas(y, faces, ns)
        ta = A[w - 21][tets[owner]].mean(1)
        presc = np.zeros(ns)
        for k in range(3):
            np.add.at(presc, faces[:, k], a21 * (1 + g * ta) ** 2 / 3)
        row = {"atlas": va_atlas[cortex].sum() * mm ** 2, "prescribed": presc[cortex].sum() * mm ** 2}
        for label, run in RUNS.items():
            p = ROOT / run.format(w=w)
            if not p.exists():
                continue
            va = vertex_areas(np.load(p)[:ns], faces, ns)
            r = va[cortex] / presc[cortex]
            row[label] = {"realized": va[cortex].sum() * mm ** 2,
                          "stretch_p10_p50_p90": np.percentile(r, [10, 50, 90]).tolist()}
        res[w] = row
        cells = [f"SG {w}: presc/atlas {row['prescribed'] / row['atlas']:.3f}"]
        for label in RUNS:
            if label in row:
                q = row[label]
                cells.append(f"{label[:10]} real/presc {q['realized'] / row['prescribed']:.3f} "
                             f"real/atlas {q['realized'] / row['atlas']:.3f} "
                             f"[{' '.join(f'{v:.2f}' for v in q['stretch_p10_p50_p90'])}]")
        print(" | ".join(cells))
    out = sys.argv[1] if len(sys.argv) > 1 else "out/area_budget.json"
    (ROOT / out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
