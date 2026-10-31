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
Reference runs: a from the measured table; fitted runs (step 5): a = the fitted theta of the owner
surface vertex (<run>/theta_full.npy, as in adjoint_week).

    python3 tools/area_budget.py [out/area_budget.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mota_law import MESHES, ROOT, STEPS, last_x, tri_areas, vertex_areas  # noqa: E402

RUNS = {     # label -> (week -> run), fitted (a from the run) or not (a from the measured table)
    "croissance mesurée (cœur)": (STEPS["2 · croissance mesurée (cœur)"], False),
    "idem, matériau réaliste": (STEPS["2 · idem, matériau réaliste"], False),
    "ajusté (loi + cœur lobaire)": (STEPS["5 · loi + cœur lobaire"], True),
    "ajusté, matériau réaliste": (STEPS["5 · idem, matériau réaliste"], True),
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
    owner_v = np.load(ROOT / "data/adjoint_prep_s/w22/owner.npy").astype(np.int64)

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
        def prescribed(a_node):
            ta = a_node[tets[owner]].mean(1)
            v = np.zeros(ns)
            for k in range(3):
                np.add.at(v, faces[:, k], a21 * (1 + g * ta) ** 2 / 3)
            return v

        presc = prescribed(A[w - 21])
        row = {"atlas": va_atlas[cortex].sum() * mm ** 2, "prescribed": presc[cortex].sum() * mm ** 2}
        for label, (runs, fitted) in RUNS.items():
            if w not in runs or not (ROOT / runs[w]).exists():
                continue
            pr = prescribed(np.load(ROOT / runs[w] / "theta_full.npy")[:ns][owner_v]) if fitted else presc
            va = vertex_areas(last_x(runs[w])[:ns], faces, ns)
            r = va[cortex] / pr[cortex]
            row[label] = {"prescribed": pr[cortex].sum() * mm ** 2, "realized": va[cortex].sum() * mm ** 2,
                          "stretch_p10_p50_p90": np.percentile(r, [10, 50, 90]).tolist()}
        res[w] = row
        cells = [f"SG {w}"]
        for label in RUNS:
            if label in row:
                q = row[label]
                cells.append(f"{label[:12]} presc/atlas {q['prescribed'] / row['atlas']:.3f} "
                             f"real/presc {q['realized'] / q['prescribed']:.3f} "
                             f"real/atlas {q['realized'] / row['atlas']:.3f} "
                             f"[{' '.join(f'{v:.2f}' for v in q['stretch_p10_p50_p90'])}]")
        print(" | ".join(cells))
    out = sys.argv[1] if len(sys.argv) > 1 else "out/area_budget.json"
    (ROOT / out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
