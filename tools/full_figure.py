"""Figure of the whole-hemisphere chains (tools/full_chain.sh): week by week, deviation from the
atlas (mm, rigid alignment), mean angle between face normals, correlation of the mean curvature and
fold depth on the cortex, for the reference chain (measured growth), a alone, and the field law
refitted or frozen; below, lateral and medial views at the last week, coloured by mean curvature.

    python3 tools/full_figure.py out/full_chains.png
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

WEEKS = [27, 28, 29, 30]
INK, INK2, MUTED, GRID, SURF = "#1f1f1e", "#52524c", "#8a897f", "#e4e3dc", "#fcfcfb"
CHAINS = {   # label: (colour, style, run per week)
    "Croissance mesurée": ("#8a897f", "--", lambda w: f"out/remodel_a0/w{w}/eq/x_released.npy"),
    "a seul": ("#2a78d6", "-", lambda w: "out/full/fit27_combo_band" if w == 27 else f"out/full/fit{w}_a"),
    "Loi réajustée": ("#eb6834", "-", lambda w: "out/full/fit27_lawfit100b" if w == 27 else f"out/full/fit{w}_law"),
    "Loi figée": ("#1baf7a", "-", lambda w: "out/full/fit27_lawfit100b" if w == 27 else f"out/full/fit{w}_lawfix"),
}


def state(path):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    ks = [int(q.stem.split("_")[1]) for q in p.glob("x_*.npy") if q.stem.split("_")[1].isdigit()]
    return np.load(p / f"x_{max(ks)}.npy")


def main():
    out = sys.argv[1]
    meta = json.loads(Path("data/meshes/w21_left_610k_s.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    faces = np.load("data/full/prep/w27/faces.npy").astype(int)
    y27 = np.load("data/full/prep/w27/target.npy")[:ns]
    fn27 = np.cross(y27[faces[:, 1]] - y27[faces[:, 0]], y27[faces[:, 2]] - y27[faces[:, 0]])
    if np.mean(np.einsum("ij,ij->i", fn27, y27[faces].mean(1) - y27.mean(0)) > 0) < 0.5:
        faces = faces[:, ::-1]          # outward normals: mean curvature < 0 at fundi
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    deg = np.asarray(a.sum(1)).ravel()
    s = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / deg) @ a
    lap = sp.identity(ns) - sp.diags(1 / deg) @ a

    def vnormals(q):
        fn = np.cross(q[faces[:, 1]] - q[faces[:, 0]], q[faces[:, 2]] - q[faces[:, 0]])
        vn = np.zeros_like(q)
        for k in range(3):
            np.add.at(vn, faces[:, k], fn)
        return vn / np.linalg.norm(vn, axis=1, keepdims=True)

    def fnormals(q):
        fn = np.cross(q[faces[:, 1]] - q[faces[:, 0]], q[faces[:, 2]] - q[faces[:, 0]])
        return fn / np.linalg.norm(fn, axis=1, keepdims=True)

    def depth(q):
        qs = q
        for _ in range(30):
            qs = s @ qs
        return np.einsum("ij,ij->i", q - qs, vnormals(qs)) * mm

    def curv(q):
        return np.einsum("ij,ij->i", lap @ q, vnormals(q))

    def rigid(x, y, m):
        ca, cb = x[m].mean(0), y[m].mean(0)
        u, _, vt = np.linalg.svd((x[m] - ca).T @ (y[m] - cb))
        d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
        return (x - ca) @ (u @ d @ vt) + cb

    metrics = {k: {"mm": [], "ang": [], "corr": [], "depth": []} for k in CHAINS}
    atlas_depth, last = [], {}
    for w in WEEKS:
        p = Path("data/full/prep") / f"w{w}"
        y = np.load(p / "target.npy")[:ns]
        m = np.load(p / "cortex.npy")[:ns] > 0.5
        fm = m[faces].all(1)
        dy, hy, ny = depth(y), curv(y), fnormals(y)
        atlas_depth.append(np.percentile(dy[m], 95) - np.percentile(dy[m], 5))
        for k, (_, _, run) in CHAINS.items():
            x = rigid(state(run(w))[:ns], y, m)
            dx = depth(x)
            mt = metrics[k]
            mt["mm"].append(np.sqrt((np.linalg.norm(x[m] - y[m], axis=1) ** 2).mean()) * mm)
            mt["ang"].append(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fnormals(x)[fm], ny[fm]), -1, 1))).mean())
            mt["corr"].append(np.corrcoef(curv(x)[m], hy[m])[0, 1])
            mt["depth"].append(np.percentile(dx[m], 95) - np.percentile(dx[m], 5))
            if w == WEEKS[-1]:
                last[k] = x
        if w == WEEKS[-1]:
            last["Atlas"] = y
    Path(out).with_suffix(".json").write_text(json.dumps({"weeks": WEEKS, "atlas_depth": atlas_depth,
                                                          "metrics": metrics}, indent=1))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig = plt.figure(figsize=(15, 10.5), dpi=130, facecolor=SURF)
    panels = [("mm", "Écart à l'atlas (mm, cortex)", "%.2f"), ("ang", "Angle des normales (°)", "%.1f"),
              ("corr", "Corrélation de la courbure", "%.2f"), ("depth", "Profondeur des plis (mm)", "%.2f")]
    for i, (key, title, fmt) in enumerate(panels):
        ax = fig.add_axes([0.05 + i * 0.24, 0.64, 0.19, 0.27], facecolor=SURF)
        if key == "depth":
            ax.plot(WEEKS, atlas_depth, color=INK, lw=2.5, label="Atlas")
            ax.annotate("Atlas", (WEEKS[-1], atlas_depth[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
        for k, (col, ls, _) in CHAINS.items():
            v = metrics[k][key]
            ax.plot(WEEKS, v, color=col, ls=ls, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label=k)
        for k in ("a seul", "Loi figée"):
            ax.annotate(fmt % metrics[k][key][-1], (WEEKS[-1], metrics[k][key][-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK2, fontsize=8)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks(WEEKS, [f"SG {w}" for w in WEEKS])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_xlim(WEEKS[0] - 0.3, WEEKS[-1] + 0.6)
        for sname in ("top", "right"):
            ax.spines[sname].set_visible(False)
    handles, labels = fig.axes[3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.985))

    # SG 30 surfaces, lateral and medial, coloured by mean curvature
    order = ["Atlas", "Croissance mesurée", "a seul", "Loi figée"]
    hy = curv(last["Atlas"])
    lim = np.percentile(np.abs(hy), 98)
    cmap = plt.get_cmap("RdBu")
    for j, k in enumerate(order):
        x = last[k]
        h = curv(x)
        v = np.column_stack([x[:, 0], -x[:, 1], -x[:, 2]])
        v -= last["Atlas"].mean(0) * np.array([1, -1, -1])
        pts = v[faces]
        nn = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
        light = np.array([-0.35, 0.85, 0.45]) / np.linalg.norm([-0.35, 0.85, 0.45])
        col = cmap(0.5 + 0.5 * np.clip(-h[faces].mean(1) / lim, -1, 1))[:, :3]
        fc = col * (0.6 + 0.4 * np.clip(np.abs(nn @ light), 0, 1))[:, None]
        for r, az in enumerate((90, -90)):
            ax = fig.add_axes([0.005 + j * 0.25, 0.30 - r * 0.27, 0.24, 0.27], projection="3d", facecolor=SURF)
            ax.add_collection3d(Poly3DCollection(pts, facecolors=fc, edgecolors=fc, linewidths=0.05))
            rr = 0.7 * np.abs(v).max()
            ax.set_xlim(-rr, rr); ax.set_ylim(-rr, rr); ax.set_zlim(-rr * .65, rr * .65)
            ax.view_init(elev=0, azim=az); ax.set_box_aspect((1, 1, .65)); ax.set_axis_off()
            if r == 0:
                fig.text(0.125 + j * 0.25, 0.575, f"{k} — SG {WEEKS[-1]}", ha="center", fontsize=11, color=INK)
    fig.text(0.5, 0.012, "Surfaces colorées par la courbure moyenne (bleu : fonds de sillon, rouge : crêtes), "
             "même échelle que l'atlas ; en haut vue latérale, en bas vue médiale",
             ha="center", color=INK2, fontsize=9)
    fig.savefig(out, facecolor=SURF)
    print("saved", out)


if __name__ == "__main__":
    main()
