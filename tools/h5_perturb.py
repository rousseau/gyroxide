#!/usr/bin/env python3
"""Perturbation lisse aléatoire d'un état ajusté (P05, hypothèse 5 : instabilités secondaires).

  python3 tools/h5_perturb.py growth <eps> <graine> <fields_in.npy> <fields_out.npy> [--smooth 10]
      a_v -> a_v (1 + eps s_v), s : bruit gaussien lissé par S^m (voisinage sur la surface), de moyenne nulle et de
      moyenne quadratique 1 ; les coefficients de la loi (les 46 derniers) ne changent pas.
  python3 tools/h5_perturb.py geom <sigma_mm> <graine> <ckpt_in> <ckpt_out> [--smooth 10]
      positions x des nœuds -> x + sigma s (3 composantes indépendantes, bruit lissé de même nature, chaque nœud prend
      la valeur de son sommet de surface propriétaire) ; v et le reste du point de reprise sont copiés tels quels.

Le lissage S^m (m = 10 par défaut) est celui du niveau k = 10 de la perte multi-échelle : longueur d'onde des plis
secondaires (4 à 8 mm).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "data/meshes/w21_left_610k_s.json").read_text())
NS, MM = META["n_surface"], META["maxd_mm"]
PREP = ROOT / "data/adjoint_prep_s/w36"


def smoother():
    f = np.load(PREP / "faces.npy").astype(int)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(NS, NS)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    return 0.5 * sp.identity(NS) + 0.5 * sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a


def noise(rng, m, shape=()):
    s = smoother()
    z = rng.standard_normal((NS,) + shape)
    for _ in range(m):
        z = s @ z
    z -= z.mean(0)
    return z / np.sqrt((z**2).mean(0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["growth", "geom"])
    p.add_argument("amp", type=float)
    p.add_argument("seed", type=int)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--smooth", type=int, default=10)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.kind == "growth":
        t = np.load(a.src).copy()
        s = noise(rng, a.smooth)
        t[:NS] = t[:NS] * (1 + a.amp * s)
        np.save(a.dst, t)
        print(f"croissance : eps {a.amp}, a moyen {t[:NS].mean():.4f}, ecart {np.abs(t[:NS] - np.load(a.src)[:NS]).max():.4f} au plus")
    else:
        src, dst = Path(a.src), Path(a.dst)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        x = np.load(src / "x.npy")
        owner = np.load(PREP / "owner.npy").astype(int)
        assert len(owner) == len(x), (len(owner), len(x))
        s = noise(rng, a.smooth, (3,))
        x = x + (a.amp / MM) * s[owner]
        np.save(dst / "x.npy", x)
        print(f"géométrie : sigma {a.amp} mm ({a.amp / MM:.2e} unités), {len(x)} nœuds")


if __name__ == "__main__":
    main()
