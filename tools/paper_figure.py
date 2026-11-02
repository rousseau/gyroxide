"""ISBI 2027 figure: folding from SG 21 to SG 36, atlas vs models, true scale, lateral view.

    python3 tools/paper_figure.py ~/Exp/isbi_2027_gyroxide/figures/folding.png

Column 1: the shared initial state (atlas SG 21, also every model's rest shape). Then one column per
week, rows: dHCP atlas, original growth law (Tallinen et al. 2016, dynamic run out/atlas_w21_tab),
measured growth with core growth at equilibrium (out/remodel_a0), fitted growth (inverse problem,
out/paper/fit*_lbm from tools/paper_chain.sh, else the earlier chain out/full/fit*_lbm[f]). Surfaces rigidly aligned on the atlas, coloured by mean curvature (one scale
per week, from the atlas), same frame and scale everywhere.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from model_gallery import (ACCENT, BG, BLUE, CURV, INK, INK2, MUTED, Surf, curv_colors, frame_box,  # noqa: E402
                           last_x, load_full, metrics, rigid_to)
from raster import render  # noqa: E402

WEEKS = (26, 28, 30, 33, 36)
TAB = Path("out/atlas_w21_tab")


def atlas_xyz(w, ns):
    """Atlas pial surface of week w in the simulation frame (same vertices for every week)."""
    p = Path("data/full/prep") / f"w{w}" / "target.npy"
    if p.exists():
        return np.load(p)[:ns]
    import json
    import nibabel as nib
    fr = json.loads(Path("data/meshes/w21_left_610k.json").read_text())
    q = nib.load(f"data/atlas_dhcp/fetal.week{w}.left.pial.surf.gii").darrays[0].data.astype(float)
    c = np.array(fr["cog_mm"])
    return np.column_stack([-(q[:, 1] - c[1]), q[:, 0] - c[0], -(q[:, 2] - c[2])]) / fr["maxd_mm"]


def tallinen(w, ns):
    f = TAB / f"surface_{(w - 21) * 8769:07d}.vtk"
    if not f.exists():
        return None
    L = f.read_text().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    x = np.loadtxt(L[i + 1:i + 1 + ns])
    return x if np.isfinite(x).all() else None


def fitted(w):
    """Fit of the paper chain (tools/paper_chain.sh) once finished, else the earlier lobar chain."""
    run = Path(f"out/paper/fit{w}_lbm")
    if (run / "theta_final.npy").exists():
        return run
    old = Path({27: "out/full/fit27_lb40"}.get(w, f"out/full/fit{w}_lbm" if w <= 31 else f"out/full/fit{w}_lbmf"))
    return old if list(old.glob("x_*.npy")) else None


ROWS = [("dHCP\natlas", INK, lambda w, ns: None),
        ("Original\ngrowth law", MUTED, tallinen),
        ("Measured\ngrowth", BLUE, lambda w, ns: last_x(f"out/remodel_a0/w{w}/eq/x_released.npy")[:ns]),
        ("Estimated\ngrowth", ACCENT, lambda w, ns: last_x(fitted(w))[:ns] if fitted(w) else None)]


def main():
    out = sys.argv[1]
    ns, mm = load_full()
    faces = np.load("data/full/prep/w30/faces.npy").astype(int)
    m = np.load("data/full/prep/w30/cortex.npy")[:ns] > 0.5
    y36 = atlas_xyz(36, ns)
    box = frame_box(y36 - y36.mean(0), "lateral", 0.02)        # one frame, true scale
    res = 500
    nr, nc = len(ROWS), len(WEEKS) + 1
    # ISBI two-column page: full text width 7.16 in, type set at the final size (7 pt labels)
    aspect = (box[3] - box[2]) / (box[1] - box[0])
    w_in = 7.16
    h_in = nr * (0.97 * w_in / nc) * aspect * 1.0 + 0.3
    fig = plt.figure(figsize=(w_in, h_in), dpi=300, facecolor=BG)
    gs = fig.add_gridspec(nr, nc, hspace=0.02, wspace=0.02, left=0.045, right=0.998, top=1 - 0.17 / h_in,
                          bottom=0.02)

    y21 = atlas_xyz(21, ns)
    S = Surf(faces, y21, mm)
    ax = fig.add_subplot(gs[1:3, 0])
    hmax21 = np.percentile(np.abs(S.curv(y21)[m]), 88)
    ax.imshow(render(y21 - y21.mean(0), S.f, curv_colors(S, y21, hmax21), "lateral", res, box)); ax.axis("off")
    ax.set_title("GA 21 w\ninitial state", fontsize=7, color=INK)
    # 10 mm scale bar (true scale: every panel shares this frame) and the curvature colour scale
    W = res * 2
    L = 10 / mm / (box[1] - box[0]) * W
    H = W * aspect
    ax.plot([0.1 * W, 0.1 * W + L], [0.96 * H, 0.96 * H], color=INK, lw=1.2)
    ax.text(0.1 * W + L + 0.03 * W, 0.96 * H, "10 mm", ha="left", va="center", fontsize=6, color=INK)
    b = fig.add_subplot(gs[0, 0]).get_position()      # empty cell left of the atlas row
    fig.axes[-1].axis("off")
    cax = fig.add_axes([b.x0 + 0.15 * b.width, b.y0 + 0.6 * b.height, 0.7 * b.width, 0.012])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CURV, aspect="auto")
    cax.set_xticks([0, 255], ["sulcus", "gyrus"], fontsize=6, color=INK2)
    cax.set_yticks([])
    cax.tick_params(length=0, pad=1.5)
    for s_ in cax.spines.values():
        s_.set_visible(False)
    cax.set_title("mean curvature", fontsize=6, color=INK2, pad=2)

    for j, w in enumerate(WEEKS, start=1):
        y = atlas_xyz(w, ns)
        S = Surf(faces, y, mm)
        hmax = np.percentile(np.abs(S.curv(y)[m]), 88)
        c = y.mean(0)
        for i, (name, col, fn) in enumerate(ROWS):
            ax = fig.add_subplot(gs[i, j]); ax.axis("off")
            if j == 1:
                b = ax.get_position()
                fig.text(0.02, (b.y0 + b.y1) / 2, name, rotation=90, ha="center", va="center", fontsize=7, color=col, linespacing=1.0)
            if i == 0:
                ax.set_title(f"GA {w} w", fontsize=7, color=INK, pad=2)
            x = fn(w, ns)
            if i > 0 and x is None:
                # original law: only the first missing week is annotated (the run stops there)
                txt = ("simulation diverged\n(GA 26.5 w)" if j == 2 else "") if i == 1 else "running"
                ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=6, color=MUTED, transform=ax.transAxes)
                continue
            x = y if x is None else (rigid_to(x, y, m) if np.isfinite(x).all() else None)
            ax.imshow(render(x - c, S.f, curv_colors(S, x, hmax), "lateral", res, box))
            if i > 0:
                mt = metrics(S, x, y, m)
                ax.text(0.98, 0.02, f"{mt['mm']:.2f} mm", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=5.5, color=INK2)
    fig.savefig(out, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
