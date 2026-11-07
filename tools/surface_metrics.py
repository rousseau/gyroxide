"""Opérateurs de surface et écart rigide, partagés par model_gallery.py, deviation_mm.py et scorecard.py.

Code déplacé tel quel depuis tools/model_gallery.py (Surf, rigid_to, metrics) et tools/deviation_mm.py
(boucle sur les niveaux de lissage) pour être importable sans matplotlib ni effet de bord ; les définitions
sont inchangées. Dépendances : numpy, scipy.
"""
import numpy as np
import scipy.sparse as sp


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


# ------------------------------------------------------------------ écart par niveau de lissage

LEVELS = [200, 50, 10, 0]


def edges_of(faces):
    """Arêtes uniques (i < j) d'un maillage triangulaire (identiques à edges.npy de tools/adjoint_prep.py)."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def smoother(edges, n):
    """Moyenne sur les voisins : S = (I + D^-1 A) / 2 (S^k dans deviation_mm.py et adjoint_week)."""
    e = np.asarray(edges).astype(int)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(float).tocsr()
    return 0.5 * sp.identity(n) + 0.5 * sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A


def stats(d):
    return {"rms": float(np.sqrt((d ** 2).mean())), "p95": float(np.percentile(d, 95))}


def deviation_levels(x, y, cx, S, mm, levels=LEVELS):
    """|S^k x_i - S^k y_i| sur les sommets de cortex cx, pour chaque k de levels : écart absolu (repère
    commun), rigide (meilleure rotation + translation, sans échelle, recalculée sur les surfaces
    lissées) et de forme seule (centré, mis à l'échelle du rayon rms de y), en mm (x, y en unités de
    1/mm). Renvoie {"k<k>": {"absolute_mm", "rigid_mm", "shape_mm"}}, chacun {rms, p95}."""
    row = {}
    sx, sy = x.copy(), y.copy()
    done = 0
    for k in sorted(levels):
        for _ in range(k - done):
            sx, sy = S @ sx, S @ sy
        done = k
        a = np.linalg.norm(sx[cx] - sy[cx], axis=1) * mm
        xc, yc = sx[cx] - sx[cx].mean(0), sy[cx] - sy[cx].mean(0)
        rx, ry = np.sqrt((xc ** 2).sum(1).mean()), np.sqrt((yc ** 2).sum(1).mean())
        s = np.linalg.norm(xc / rx - yc / ry, axis=1) * ry * mm
        u, _, vt = np.linalg.svd(xc.T @ yc)
        dfix = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
        r = u @ dfix @ vt  # xc @ r ~ yc
        g = np.linalg.norm(xc @ r - yc, axis=1) * mm
        row[f"k{k}"] = {"absolute_mm": stats(a), "rigid_mm": stats(g), "shape_mm": stats(s)}
    return row


def folding_index(x, f):
    """Indice de plissement A / V^(2/3) d'une surface fermée (colonne « folding index » de la Table 1 de
    l'article ISBI, tools/paper_table.py). Distinct de l'IG A / aire de l'enveloppe convexe (fold_standard.py)."""
    a, b, c = x[f[:, 0]], x[f[:, 1]], x[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    vol = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6
    return area / vol ** (2 / 3)
