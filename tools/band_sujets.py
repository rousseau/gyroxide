#!/usr/bin/env python3
"""P05, hypothèse 1 (cible) : contenu par bande de la courbure moyenne de surfaces pial individuelles (hémisphère
gauche, jeu surfoxide-dataset, multibounti43) contre l'atlas fœtal dHCP au même âge gestationnel.

Même définition que tools/band_error.py (courbure moyenne lissée, bandes dyadiques de longueur d'onde par diffusion
de la chaleur, 1 à 32 mm, pondération par la masse de la surface), mais sans correspondance sommet à sommet : seule
la puissance de chaque bande et la profondeur de plis (p95 − p5 de la profondeur le long des normales de la surface
lissée, comme tools/full_eval.py) se comparent. Les maillages n'ont pas la même résolution (sujets : arête 0,6 mm,
atlas : 0,84 mm) : les deux lissages dépendants du maillage (S² de la courbure, 30 passes de la profondeur) sont
rapportés à l'arête de l'atlas (nombre de passes × (arête de l'atlas au même âge / arête du sujet)²) ; les valeurs « natives »
(S² et 30 passes, sans correction) sont données à côté.

Zone de mesure : les sommets où l'écart pial-substance blanche dépasse 0,6 mm (règle de tools/atlas_to_mesh.py
pour la paroi médiale), pour les sujets comme pour l'atlas.

    python3 tools/band_sujets.py sorties.json liste.tsv dossier_surfaces atlas_dossier

liste.tsv : colonnes `fichier` (<sujet>_<séance>, sans suffixe) et `ga_weeks`. Les noms de sujets restent dans
sorties.json, hors git (out/E48/) ; ce script n'en contient aucun.
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from lobar_basis import cotan_laplacian  # noqa: E402

SCALES = [1, 2, 4, 8, 16, 32]
SEUIL_PAROI_MM = 0.6


def charge(f):
    g = nib.load(str(f))
    return np.asarray(g.agg_data("pointset"), float), np.asarray(g.agg_data("triangle"), int)


def subdivise(x, f):
    """Subdivision 1 → 4 par les milieux d'arêtes (même surface linéaire par morceaux, arête divisée par 2)."""
    n = len(x)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    u, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    m = len(f)
    m01, m12, m20 = n + inv[:m], n + inv[m:2 * m], n + inv[2 * m:]
    xs = np.concatenate([x, 0.5 * (x[u[:, 0]] + x[u[:, 1]])])
    fs = np.concatenate([
        np.stack([f[:, 0], m01, m20], 1), np.stack([f[:, 1], m12, m01], 1),
        np.stack([f[:, 2], m20, m12], 1), np.stack([m01, m12, m20], 1)])
    return xs, fs, u


def lisseur(f, n):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    return 0.5 * sp.identity(n) + 0.5 * sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a


def normales(x, f):
    fn = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    vn = np.zeros_like(x)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.linalg.norm(vn, axis=1, keepdims=True)


def mesure(x, f, zone, passes_courbure, passes_profondeur):
    """Puissance (rms, mm⁻¹) par bande de la courbure moyenne et profondeur de plis, sur la zone."""
    n = len(x)
    s = lisseur(f, n)
    lap, mass = cotan_laplacian(x, f)
    m = mass.diagonal()
    vn = normales(x, f)
    if np.mean(np.einsum("ij,ij->i", vn, x - x.mean(0)) > 0) < 0.5:
        vn = -vn
    h = 0.5 * np.einsum("ij,ij->i", (lap @ x) / m[:, None], vn)
    for _ in range(passes_courbure):
        h = s @ h
    lus = [splu((mass + sc ** 2 / 2 / 4 * lap).tocsc()) for sc in SCALES]
    prev, bandes = None, []
    for lu in lus:
        u = h.copy()
        for _ in range(4):
            u = lu.solve(m * u)
        if prev is not None:
            bandes.append(prev - u)
        prev = u

    def rms(v):
        return float(np.sqrt(np.sum(m[zone] * v[zone] ** 2) / m[zone].sum()))

    qs = x
    for _ in range(passes_profondeur):
        qs = s @ qs
    d = np.einsum("ij,ij->i", x - qs, normales(qs, f))
    prof = float(np.percentile(d[zone], 95) - np.percentile(d[zone], 5))
    return {"rms": [rms(b) for b in bandes], "profondeur_mm": prof, "aire_zone_mm2": float(m[zone].sum())}


def arete(x, f):
    return float(np.linalg.norm(x[f[:, 0]] - x[f[:, 1]], axis=1).mean())


def zone_de(p, w):
    return np.linalg.norm(p - w, axis=1) >= SEUIL_PAROI_MM


def main():
    sortie, liste, dossier, atlas = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    # atlas : une mesure par semaine, native (S², 30 passes) ; rapportée à l'arête de l'atlas = native
    at = {}
    for w in range(30, 37):
        p, f = charge(atlas / f"fetal.week{w}.left.pial.surf.gii")
        bl, _ = charge(atlas / f"fetal.week{w}.left.wm.surf.gii")
        z = zone_de(p, bl)
        at[w] = {"arete_mm": arete(p, f), "zone_sommets": int(z.sum()), **mesure(p, f, z, 2, 30)}
        print(f"atlas SG {w}: arête {at[w]['arete_mm']:.2f} mm, zone {z.sum()} sommets, rms {np.round(at[w]['rms'], 4).tolist()}, profondeur {at[w]['profondeur_mm']:.2f}")
    e_at = np.mean([v["arete_mm"] for v in at.values()])
    # contrôle de méthode : l'atlas du SG 36 subdivisé (arête / 2), mesuré natif puis avec les passes rapportées à l'arête de l'atlas
    p, f = charge(atlas / "fetal.week36.left.pial.surf.gii")
    bl, _ = charge(atlas / "fetal.week36.left.wm.surf.gii")
    z = zone_de(p, bl)
    ps, fs, _ = subdivise(p, f)
    xw, _, _ = subdivise(bl, f)
    zs = zone_de(ps, xw)
    r = (arete(p, f) / arete(ps, fs)) ** 2
    ctrl = {"natif": mesure(ps, fs, zs, 2, 30), "rapporte": mesure(ps, fs, zs, int(round(2 * r)), int(round(30 * r))),
            "passes_rapportees": [int(round(2 * r)), int(round(30 * r))]}
    print(f"contrôle (atlas SG 36 subdivisé, arête {arete(ps, fs):.2f} mm) : natif rms {np.round(ctrl['natif']['rms'], 4).tolist()} prof {ctrl['natif']['profondeur_mm']:.2f}; "
          f"rapporté {ctrl['passes_rapportees']} passes rms {np.round(ctrl['rapporte']['rms'], 4).tolist()} prof {ctrl['rapporte']['profondeur_mm']:.2f}")
    sujets = []
    lignes = [l.rstrip("\n").split("\t") for l in liste.read_text().splitlines()[1:]]
    for nom, ga in lignes:
        p, f = charge(dossier / f"{nom}_hemi-L_pial.surf.gii")
        bl, _ = charge(dossier / f"{nom}_hemi-L_white.surf.gii")
        z = zone_de(p, bl)
        e = arete(p, f)
        e_ref = float(np.interp(float(ga), list(at), [v["arete_mm"] for v in at.values()]))  # arête de l'atlas à l'âge du sujet
        r = (e_ref / e) ** 2
        nat = mesure(p, f, z, 2, 30)
        eq = mesure(p, f, z, int(round(2 * r)), int(round(30 * r)))
        sujets.append({"sujet": nom, "ga_weeks": float(ga), "arete_mm": e, "zone_sommets": int(z.sum()), "natif": nat, "rapporte": eq})
        print(f"{nom[:9]}… SG {float(ga):.2f} arête {e:.2f} : puissance 4-8 (rapportée) {eq['rms'][2] ** 2:.3e}, prof {eq['profondeur_mm']:.2f}")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps({"atlas": at, "controle_subdivision_atlas_36": ctrl, "arete_atlas_moyenne_mm": e_at, "sujets": sujets}, indent=1))


if __name__ == "__main__":
    main()
