"""Visual comparison of the fitted models (tools/raster.py renderer), surfaces coloured by mean
curvature (blue: sulcal fundi, cream: gyral crowns) or by the deviation from the atlas.

    python3 tools/model_gallery.py steps out/gallery/steps_sg30.png     # steps 2 to 5 at SG 30
    python3 tools/model_gallery.py chain out/gallery/chain.png          # SG 28 to 36
    python3 tools/model_gallery.py losses out/gallery/losses.png        # central sulcus, SG 27
    python3 tools/model_gallery.py fields out/gallery/fields.png        # central sulcus, SG 27
    python3 tools/model_gallery.py secondary out/gallery/secondary.png  # frontal region, SG 33
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from raster import camera, render  # noqa: E402

INK, INK2, MUTED, BG = "#1B2430", "#4A5361", "#6B7280", "#FCFCFB"
ACCENT, BLUE = "#C2571A", "#2A5DA8"
CURV = LinearSegmentedColormap.from_list("curv", ["#2F4F7A", "#7E95B3", "#E6DDCF", "#F6F0E6"])
ERR = LinearSegmentedColormap.from_list("err", ["#F2EEE6", "#F0C9A0", "#E08A4E", "#B6491B", "#6E2410"])
LINE = ["#2A5DA8", "#C2571A", "#3B8C6E", "#8E5BB5", "#B8860B", "#C0392B"]
ERR_MAX = 3.0


def last_x(run):
    p = Path(run)
    if p.suffix == ".npy":
        return np.load(p)
    ks = [int(q.stem.split("_")[1]) for q in p.glob("x_*.npy") if q.stem.split("_")[1].isdigit()]
    return np.load(p / f"x_{max(ks)}.npy")


class Surf:
    """Surface operators on one triangulation (the atlas one, shared by every week and model)."""

    def __init__(self, faces, ref, mm):
        self.mm = mm
        n = len(ref)
        f = faces
        fn = np.cross(ref[f[:, 1]] - ref[f[:, 0]], ref[f[:, 2]] - ref[f[:, 0]])
        if np.mean(np.einsum("ij,ij->i", fn, ref[f].mean(1) - ref.mean(0)) > 0) < 0.5:
            f = f[:, ::-1]                  # outward normals
        self.f = f
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        a = ((a + a.T) > 0).astype(float)
        deg = np.asarray(a.sum(1)).ravel()
        self.s = 0.5 * sp.identity(n) + 0.5 * sp.diags(1 / deg) @ a
        self.lap = sp.identity(n) - sp.diags(1 / deg) @ a

    def normals(self, q):
        f = self.f
        fn = np.cross(q[f[:, 1]] - q[f[:, 0]], q[f[:, 2]] - q[f[:, 0]])
        vn = np.zeros_like(q)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)

    def curv(self, q):
        h = np.einsum("ij,ij->i", self.lap @ q, self.normals(q))
        return self.s @ (self.s @ h)

    def depth(self, q):
        qs = q
        for _ in range(30):
            qs = self.s @ qs
        return np.einsum("ij,ij->i", q - qs, self.normals(qs)) * self.mm


def rigid_to(x, y, m):
    ca, cb = x[m].mean(0), y[m].mean(0)
    u, _, vt = np.linalg.svd((x[m] - ca).T @ (y[m] - cb))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    return (x - ca) @ (u @ d @ vt) + cb


def metrics(S, x, y, m):
    d = np.linalg.norm(x - y, axis=1) * S.mm
    dx, dy = S.depth(x), S.depth(y)
    return {"mm": np.sqrt((d[m] ** 2).mean()), "corr": np.corrcoef(S.curv(x)[m], S.curv(y)[m])[0, 1],
            "depth": np.percentile(dx[m], 95) - np.percentile(dx[m], 5),
            "depth_atlas": np.percentile(dy[m], 95) - np.percentile(dy[m], 5), "err": d}


def curv_colors(S, q, hmax):
    return CURV(0.5 + 0.5 * np.clip(S.curv(q) / hmax, -1, 1))[:, :3]


def frame_box(points, view, pad=0.03):
    p = points @ camera(view).T
    lo, hi = p[:, :2].min(0), p[:, :2].max(0)
    r = pad * (hi - lo).max()
    return (lo[0] - r, hi[0] + r, lo[1] - r, hi[1] + r)


def label(ax, title, sub=None, color=INK):
    ax.set_title(title, loc="left", fontsize=13, color=color, pad=4)
    if sub:
        ax.text(0.0, -0.02, sub, transform=ax.transAxes, va="top", ha="left", fontsize=11, color=INK2)


def colorbars(fig, y=0.035, hmax_label=True):
    for i, (cm, lo, hi, txt) in enumerate([(CURV, "sillon", "crête", "Courbure moyenne"),
                                            (ERR, "0", f"{ERR_MAX:.0f} mm", "Écart à l’atlas")]):
        ax = fig.add_axes([0.18 + i * 0.4, y, 0.22, 0.012])
        ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cm, aspect="auto")
        ax.set_xticks([0, 255], [lo, hi], fontsize=10, color=INK2)
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(txt, fontsize=10, color=INK2, pad=3)


def fmt(v, nd=2):
    return f"{v:.{nd}f}".replace(".", ",")


# ------------------------------------------------------------------ whole hemisphere

def load_full():
    meta = json.loads(Path("data/meshes/w21_left_610k_s.json").read_text())
    return meta["n_surface"], meta["maxd_mm"]


def week_data(w, ns):
    p = Path("data/full/prep") / f"w{w}"
    return np.load(p / "target.npy")[:ns], np.load(p / "cortex.npy")[:ns] > 0.5, np.load(p / "faces.npy").astype(int)


def fig_steps(out, w=30):
    ns, mm = load_full()
    y, m, faces = week_data(w, ns)
    S = Surf(faces, y, mm)
    cols = [("Atlas dHCP", None),
            ("2 · croissance du cœur", f"out/remodel_a0/w{w}/eq/x_released.npy"),
            ("3 · + tangentielle ajustée", f"out/full/fit{w}_a"),
            ("4 · + croissance radiale", f"out/full/fit{w}_law"),
            ("5 · + cœur lobaire", f"out/full/fit{w}_lbm")]
    hmax = np.percentile(np.abs(S.curv(y)[m]), 88)
    box = frame_box(y, "lateral")
    # zoom: perisylvian and central region, middle of the lateral view
    zx0, zx1, zy0, zy1 = box
    cx, cy = zx0 + 0.42 * (zx1 - zx0), zy0 + 0.55 * (zy1 - zy0)
    half = 0.2 * (zx1 - zx0)
    zoom = (cx - half, cx + half, cy - 0.62 * half, cy + 0.62 * half)
    fig = plt.figure(figsize=(25, 13.2), dpi=110, facecolor=BG)
    gs = fig.add_gridspec(3, len(cols), height_ratios=[1, 1, 1], hspace=0.16, wspace=0.03,
                          left=0.01, right=0.99, top=0.95, bottom=0.08)
    for j, (name, run) in enumerate(cols):
        x = y if run is None else rigid_to(last_x(run)[:ns], y, m)
        mt = metrics(S, x, y, m)
        cc = curv_colors(S, x, hmax)
        ax = fig.add_subplot(gs[0, j]); ax.imshow(render(x, S.f, cc, "lateral", 900, box)); ax.axis("off")
        sub = (f"profondeur des plis {fmt(mt['depth'])} mm" if run is None else
               f"{fmt(mt['mm'])} mm · corr. courbure {fmt(mt['corr'])} · profondeur {fmt(mt['depth'])} mm")
        label(ax, name, sub, INK if run is None else (BLUE if j == 1 else ACCENT))
        ax = fig.add_subplot(gs[1, j]); ax.imshow(render(x, S.f, cc, "lateral", 900, zoom)); ax.axis("off")
        ax = fig.add_subplot(gs[2, j])
        if run is None:
            ax.imshow(render(x, S.f, cc, "medial", 900, frame_box(y, "medial"))); ax.axis("off")
            label(ax, "Atlas, vue médiale")
        else:
            ec = ERR(np.clip(mt["err"] / ERR_MAX, 0, 1))[:, :3]
            ax.imshow(render(x, S.f, ec, "lateral", 900, box)); ax.axis("off")
            label(ax, "Écart à l’atlas", f"95e centile {fmt(np.percentile(mt['err'][m], 95))} mm")
    fig.text(0.01, 0.985, f"SG {w} · vue latérale, zoom sur les régions sylvienne et centrale, écart au sommet",
             fontsize=14, color=INK2, va="top")
    colorbars(fig)
    fig.savefig(out, facecolor=BG)
    print("saved", out)


def fig_chain(out, weeks=(28, 30, 32, 34, 36)):
    ns, mm = load_full()
    rows = [("Atlas dHCP", lambda w: None),
            ("2 · croissance du cœur", lambda w: f"out/remodel_a0/w{w}/eq/x_released.npy"),
            ("5 · modèle enrichi", lambda w: f"out/full/fit{w}_lbm" if w <= 31 else f"out/full/fit{w}_lbmf")]
    fig = plt.figure(figsize=(26, 15.5), dpi=105, facecolor=BG)
    gs = fig.add_gridspec(len(rows) + 1, len(weeks), hspace=0.2, wspace=0.03, left=0.07, right=0.995, top=0.95, bottom=0.07)
    for j, w in enumerate(weeks):
        y, m, faces = week_data(w, ns)
        S = Surf(faces, y, mm)
        hmax = np.percentile(np.abs(S.curv(y)[m]), 88)
        box = frame_box(last_x(f"out/remodel_a0/w{weeks[-1]}/eq/x_released.npy")[:ns] - y.mean(0) + y.mean(0), "lateral")
        box = frame_box(np.load(Path("data/full/prep") / f"w{weeks[-1]}" / "target.npy")[:ns], "lateral", 0.02)
        for i, (name, run) in enumerate(rows):
            r = run(w)
            x = y if r is None else rigid_to(last_x(r)[:ns], y, m)
            mt = metrics(S, x, y, m)
            ax = fig.add_subplot(gs[i, j]); ax.imshow(render(x, S.f, curv_colors(S, x, hmax), "lateral", 800, box)); ax.axis("off")
            sub = (f"profondeur {fmt(mt['depth'])} mm" if r is None else
                   f"{fmt(mt['mm'])} mm · profondeur {fmt(mt['depth'])} mm")
            label(ax, f"SG {w}" if i == 0 else "", sub)
            if j == 0:
                ax.text(-0.02, 0.5, name, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=14,
                        color=INK if i == 0 else (BLUE if i == 1 else ACCENT))
            if i == len(rows) - 1:
                ax = fig.add_subplot(gs[len(rows), j])
                ax.imshow(render(x, S.f, ERR(np.clip(mt["err"] / ERR_MAX, 0, 1))[:, :3], "lateral", 800, box)); ax.axis("off")
                label(ax, "", f"écart, 95e centile {fmt(np.percentile(mt['err'][m], 95))} mm")
                if j == 0:
                    ax.text(-0.02, 0.5, "5 · écart", transform=ax.transAxes, rotation=90, ha="right", va="center",
                            fontsize=14, color=ACCENT)
    colorbars(fig, 0.03)
    fig.savefig(out, facecolor=BG)
    print("saved", out)


# ------------------------------------------------------------------ regions (slabs)

def fig_slab(out, slab, w, runs, title, ncol=None):
    slab = Path(slab)
    meta = json.loads((slab / "slab.json").read_text())
    ns, mm = meta["n_surface"], meta["maxd_mm"]
    P = slab / "prep" / f"w{w}"
    y = np.load(P / "target.npy")[:ns]
    m = np.load(P / "cortex.npy")[:ns] > 0.5
    S = Surf(np.load(P / "faces.npy").astype(int), y, mm)
    first = next(r for _, r in runs)
    items = [("Atlas dHCP", y), ("Départ", np.load(Path(first) / "x_0.npy")[:ns])]
    items += [(k, last_x(r)[:ns]) for k, r in runs]
    ncol = ncol or len(items)
    # view along the mean outward normal of the region; sulcus axis (most concave vertices) vertical
    nz = S.normals(y)[m].mean(0); nz /= np.linalg.norm(nz)
    h = S.curv(y)
    fund = m & (h < np.percentile(h[m], 15))
    c = y[m].mean(0)
    _, _, vt = np.linalg.svd(y[fund] - y[fund].mean(0))
    ay = vt[0] - (vt[0] @ nz) * nz; ay /= np.linalg.norm(ay)
    ax_ = np.cross(ay, nz)
    view = (-nz, ay)
    hmax = np.percentile(np.abs(h[m]), 88)
    q = (y[m] - c) @ camera(view).T
    lo, hi = q[:, :2].min(0), q[:, :2].max(0)
    box = (lo[0] + c @ camera(view)[0], hi[0] + c @ camera(view)[0], lo[1] + c @ camera(view)[1], hi[1] + c @ camera(view)[1])
    nrow = int(np.ceil(len(items) / ncol))
    fig = plt.figure(figsize=(3.6 * ncol, 5.0 * nrow + 5.2), dpi=110, facecolor=BG)
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[5.0] * nrow + [5.2], hspace=0.22, wspace=0.04,
                          left=0.02, right=0.98, top=0.93, bottom=0.08)
    for k, (name, x) in enumerate(items):
        if k:
            x = rigid_to(x, y, m)
            items[k] = (name, x)
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        ax.imshow(render(x, S.f, curv_colors(S, x, hmax), view, 700, box)); ax.axis("off")
        mt = metrics(S, x, y, m)
        sub = (f"profondeur {fmt(mt['depth'])} mm" if k == 0 else
               f"{fmt(mt['mm'])} mm · profondeur {fmt(mt['depth'])} mm")
        label(ax, name, sub, INK if k == 0 else (MUTED if k == 1 else ACCENT))
    # cross-section through the region centre, across the sulcus axis
    ax = fig.add_subplot(gs[nrow, :])
    for k, (name, x) in enumerate(items):
        v = (x - c) @ np.stack([ax_, ay, nz]).T * mm
        F = S.f[m[S.f].all(1)]
        a = v[F]
        sg = a[:, :, 1]
        cut = (sg.min(1) < 0) & (sg.max(1) > 0)
        segs = []
        for tri in a[cut]:
            s_ = tri[:, 1]
            pts = [tri[i0] + s_[i0] / (s_[i0] - s_[i1]) * (tri[i1] - tri[i0])
                   for i0, i1 in ((0, 1), (1, 2), (2, 0)) if (s_[i0] > 0) != (s_[i1] > 0)]
            if len(pts) == 2:
                segs.append(pts)
        style = (dict(color=INK, lw=3.2) if k == 0 else dict(color=MUTED, lw=2, ls="--") if k == 1
                 else dict(color=LINE[(k - 2) % len(LINE)], lw=2))
        for p0, p1 in segs:
            ax.plot([p0[0], p1[0]], [p0[2], p1[2]], **style)
        ax.plot([], [], label=name, **style)
    ax.set_aspect("equal")
    ax.set_xlabel("mm, en travers du sillon", fontsize=11, color=INK2)
    ax.set_ylabel("mm, normale", fontsize=11, color=INK2)
    ax.tick_params(colors=MUTED)
    for s_ in ax.spines.values():
        s_.set_color("#D0D4D9")
    ax.legend(frameon=False, fontsize=11, ncol=min(len(items), 4), loc="upper left", bbox_to_anchor=(0, -0.18))
    ax.set_title("Coupe au centre de la région", loc="left", fontsize=13, color=INK)
    fig.text(0.02, 0.975, title, fontsize=15, color=INK2, va="top")
    fig.savefig(out, facecolor=BG, bbox_inches="tight")
    print("saved", out)


def main():
    what, out = sys.argv[1], sys.argv[2]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if what == "steps":
        fig_steps(out, int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    elif what == "chain":
        fig_chain(out)
    elif what == "losses":
        fig_slab(out, "data/slab_cs", 27, [
            ("Sommets", "out/slab_cs/fit27_vertex"), ("Normales", "out/slab_cs/fit27_normal"),
            ("Laplacien", "out/slab_cs/fit27_lap"), ("Sommets + normales", "out/slab_cs/fit27_vn"),
            ("Composite (retenue)", "out/slab_cs/fit27_combo")],
            "Sillon central, SG 27 · même croissance ajustée (a), fonctions de coût différentes")
    elif what == "fields":
        fig_slab(out, "data/slab_cs", 27, [
            ("a seul (étape 3)", "out/slab_cs/fit27_combo_band"), ("a + épaississement c", "out/slab_cs/fit27_band_c"),
            ("a + cœur k", "out/slab_cs/fit27_band_k"), ("a + c + k libres", "out/slab_cs/fit27_band_ck"),
            ("Loi de profondeur (étape 4)", "out/slab_cs/fit27_lawfit")],
            "Sillon central, SG 27 · ce que chaque champ de croissance ajoute")
    elif what == "secondary":
        fig_slab(out, "data/slab_front10", 33, [
            ("a seul", "out/slab_front10/fit33_aonly"), ("Loi de profondeur", "out/slab_front10/fit33_law"),
            ("a, c, k libres", "out/slab_front10/fit33_free_b1e-3")],
            "Région frontale, SG 33 (10 mm de profondeur) · plis secondaires")


if __name__ == "__main__":
    main()
