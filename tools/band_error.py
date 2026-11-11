#!/usr/bin/env python3
"""Erreur de courbure par bande d'échelle (P05, mesure commune) : pour chaque bande dyadique de longueur
d'onde (analogue par diffusion de la chaleur des bandes SpAnGy, comme tools/fold_standard.py : H filtrée à
l'échelle s moins H filtrée à 2s, s = 1, 2, 4, 8, 16 mm), sur l'hémisphère et la cortex :

  - rapport de puissance  P_modèle / P_atlas  (déjà dans fold_standard.py) ;
  - erreur relative       rms(B(H_modèle − H_atlas)) / rms(B(H_atlas)) ;
  - corrélation           corr(B(H_modèle), B(H_atlas))  (phase : un pli de bonne amplitude mal placé a
                          rapport 1, corrélation 0 et erreur √2).

Courbure moyenne, Laplacien des cotangentes, lissage S^2 : la définition de fold_standard.py, inchangée.
La numérotation B4 / B5 / B6 de SpAnGy n'est pas reprise (sa définition en fréquences propres n'est pas
retrouvée ici) : les bandes sont nommées par leur longueur d'onde (8–16 et 4–8 mm = plis secondaires).

    python3 tools/band_error.py out/E12/bandes.json étiquette=semaine:run_ou_x.npy ...
    python3 tools/band_error.py out/E12/bandes.json --scorecard out/scorecard/t1_r0857 estimated 28 30 33 36

--scorecard <dossier> <bras> <semaines> : lit les entrées (run + positions) de <dossier>/<bras>_<SG>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from lobar_basis import cotan_laplacian  # noqa: E402

SCALES = [1, 2, 4, 8, 16, 32]
META = json.loads((ROOT / "data/meshes/w21_left_610k_s.json").read_text())
NS, MM = META["n_surface"], META["maxd_mm"]


class Surface:
    def __init__(self, faces, cortex):
        self.faces, self.cortex = faces, cortex
        n = len(cortex)
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        a = ((a + a.T) > 0).astype(float)
        self.s = 0.5 * sp.identity(n) + 0.5 * sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a

    def curvature(self, x):
        """Courbure moyenne (positive sur les crêtes), lissée S^2 : celle de fold_standard.Surface.metrics."""
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
        h = 0.5 * np.einsum("ij,ij->i", (lap @ x) / m[:, None], vn)
        for _ in range(2):
            h = self.s @ h
        return h, lap, mass

    def mass(self, x):
        return cotan_laplacian(x, self.faces)[1].diagonal()

    def filters(self, x):
        """Opérateurs de diffusion de la chaleur aux échelles SCALES sur la géométrie x (mm)."""
        lap, mass = cotan_laplacian(x, self.faces)
        return [splu((mass + sc ** 2 / 2 / 4 * lap).tocsc()) for sc in SCALES], mass.diagonal()


def bands(lus, m, h):
    out, prev = [], None
    for lu in lus:
        u = h.copy()
        for _ in range(4):
            u = lu.solve(m * u)
        if prev is not None:
            out.append(prev - u)
        prev = u
    return out


def band_errors(surf, xm, xa):
    """xm, xa en mm. Chaque courbure est filtrée sur sa propre géométrie (définition de fold_standard.py : le
    modèle et l'atlas n'ont pas la même aire) ; les champs de bande se comparent sommet à sommet (même
    triangulation). Poids : masse de l'atlas."""
    ha, _, _ = surf.curvature(xa)
    hm, _, _ = surf.curvature(xm)
    lus, m = surf.filters(xa)
    lum, _ = surf.filters(xm)
    c = surf.cortex
    mm_ = surf.mass(xm)
    ba, bm = bands(lus, m, ha), bands(lum, mm_, hm)

    def rms(v, w=m):
        return float(np.sqrt(np.sum(w[c] * v[c] ** 2) / w[c].sum()))

    res = []
    for k, (a, b) in enumerate(zip(ba, bm)):
        r = rms(b, mm_) / rms(a)
        cor = float(np.sum(m[c] * a[c] * b[c]) / np.sqrt(np.sum(m[c] * a[c] ** 2) * np.sum(m[c] * b[c] ** 2)))
        res.append({"bande_mm": f"{SCALES[k]}-{SCALES[k + 1]}", "rms_atlas": rms(a), "rms_modele": rms(b, mm_),
                    "rapport_puissance": rms(b, mm_) ** 2 / rms(a) ** 2, "erreur_rel": rms(b - a) / rms(a), "corr": cor,
                    # erreur² / puissance de l'atlas = (1 − r)² [amplitude] + 2 r (1 − corr) [motif], r = ‖B_m‖/‖B_a‖
                    "part_amplitude": (1 - r) ** 2, "part_motif": 2 * r * (1 - cor)})
    return res


def state(path):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    ks = [int(q.stem.split("_")[1]) for q in p.glob("x_*.npy") if q.stem.split("_")[1].isdigit()]
    return np.load(p / f"x_{max(ks)}.npy")


def main():
    out, args = Path(sys.argv[1]), sys.argv[2:]
    runs = {}
    if args and args[0] == "--scorecard":
        d, arm, weeks = Path(args[1]), args[2], [int(w) for w in args[3:]]
        for w in weeks:
            j = json.loads((d / f"{arm}_{w}.json").read_text())["inputs"]
            runs.setdefault(f"{d.name}/{arm}", {})[w] = j["run"] if j["run"].endswith(".npy") else str(Path(j["run"]) / j["positions"])
    else:
        for a in args:
            label, spec = a.split("=", 1)
            w, r = spec.split(":", 1)
            runs.setdefault(label, {})[int(w)] = r
    res = json.loads(out.read_text()) if out.exists() else {}
    for w in sorted({w for r in runs.values() for w in r}):
        p = ROOT / "data/full/prep" / f"w{w}"
        surf = Surface(np.load(p / "faces.npy").astype(int), np.load(p / "cortex.npy")[:NS] > 0.5)
        xa = np.load(p / "target.npy")[:NS] * MM
        for label, r in runs.items():
            if w in r:
                e = band_errors(surf, state(ROOT / r[w])[:NS] * MM, xa)
                res.setdefault(label, {})[str(w)] = e
                print(f"SG {w} {label}")
                for b in e:
                    print(f"  {b['bande_mm']:>6} mm  puissance x{b['rapport_puissance']:.2f}  erreur {b['erreur_rel']:.2f}  corr {b['corr']:+.2f}"
                          f"  erreur² = amplitude {b['part_amplitude']:.3f} + motif {b['part_motif']:.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
