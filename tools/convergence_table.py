#!/usr/bin/env python3
"""Convergence en maillage (P03, tâche 3) : lit les scorecard/measured_<SG>.json des trois maillages (g, m, f) et écrit, en Markdown,
les valeurs de chaque métrique, l'écart relatif entre maillages voisins et l'extrapolation de Richardson.

Usage : tools/convergence_table.py [--dir out/E38] [--weeks 26 28 30] [--ref out/scorecard/sens_r0857] [--tets g=N m=N f=N]

Écart relatif entre deux maillages : |a − b| / |b| (b : le plus fin des deux) ; pour les écarts rigides en mm, la valeur est la RMS elle-même.
Richardson : pas h = N_tets^(−1/3) ; avec trois valeurs, l'ordre observé p résout (Mg − Mm) / (Mm − Mf) = (hg^p − hm^p) / (hm^p − hf^p)
(bissection sur p dans ]0, 8[, seulement si la suite est monotone) ; limite M∞ = Mf − (Mm − Mf) hf^p / (hm^p − hf^p).
"""
import argparse
import json
import math
import sys
from pathlib import Path

METRIQUES = [
    ("RMS S200 (mm)", lambda d: d["metrics"]["dev_rigid_mm"]["s200"]["rms"]),
    ("RMS S50 (mm)", lambda d: d["metrics"]["dev_rigid_mm"]["s50"]["rms"]),
    ("RMS sommets (mm)", lambda d: d["metrics"]["dev_rigid_mm"]["vertex"]["rms"]),
    ("corrélation de courbure", lambda d: d["metrics"]["curvature_corr"]),
    ("profondeur des plis / atlas", lambda d: d["metrics"]["fold_depth_mm"]["ratio"]),
    ("aire / atlas", lambda d: d["metrics"]["area_mm2"]["ratio"]),
    ("IG / atlas", lambda d: d["metrics"]["gi"]["ratio"]),
    ("indice de plissement / atlas", lambda d: d["metrics"]["folding_index"]["ratio"]),
    ("épaisseur / atlas", lambda d: d["metrics"]["thickness_mm"]["ratio"]),
    ("puissance de courbure 4-8 mm / atlas", lambda d: d["metrics"]["curvature_bands"]["ratio"][2]),
]
GLOBALES = {0, 1, 2, 3, 4, 5, 6, 7, 8}  # les neuf premières sont les métriques globales du critère ; la puissance par bande est rapportée à part


def ordre(mg, mm, mf, hg, hm, hf):
    """Ordre observé p, ou None si la suite n'est pas monotone ou si l'équation n'a pas de racine dans ]0, 8[."""
    if not ((mg - mm) * (mm - mf) > 0):
        return None
    cible = (mg - mm) / (mm - mf)

    def f(p):
        return (hg**p - hm**p) / (hm**p - hf**p) - cible

    lo, hi = 1e-3, 8.0
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/E38"))
    ap.add_argument("--weeks", type=int, nargs="+", default=[26, 28, 30])
    ap.add_argument("--ref", type=Path, default=None, help="dossier de scorecards de la chaîne de référence (measured_<SG>.json)")
    ap.add_argument("--tets", nargs=3, default=["g=320563", "m=632276", "f=1617269"])
    a = ap.parse_args()
    n = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.tets}
    h = {k: v ** (-1 / 3) for k, v in n.items()}
    print(f"Pas h = N^(−1/3) : g {h['g']:.5f}, m {h['m']:.5f}, f {h['f']:.5f} ; rapports h_g/h_m = {h['g']/h['m']:.3f}, h_m/h_f = {h['m']/h['f']:.3f}.\n")
    pire = {}
    for w in a.weeks:
        d = {m: json.load(open(a.dir / m / "scorecard" / f"measured_{w}.json")) for m in "gmf"}
        ref = json.load(open(a.ref / f"measured_{w}.json")) if a.ref and (a.ref / f"measured_{w}.json").exists() else None
        print(f"### SG {w}\n")
        print("| métrique | g (0,32 M) | m (0,63 M) | f (1,62 M) | écart g/m | écart m/f | ordre p | limite (Richardson) | écart f/limite |" + (" réf. 610 k | écart m/réf. |" if ref else ""))
        print("|---|---|---|---|---|---|---|---|---|" + ("---|---|" if ref else ""))
        for i, (nom, fn) in enumerate(METRIQUES):
            mg, mm, mf = (fn(d[m]) for m in "gmf")
            egm, emf = abs(mg - mm) / abs(mm), abs(mm - mf) / abs(mf)
            p = ordre(mg, mm, mf, h["g"], h["m"], h["f"])
            if p is not None:
                lim = mf - (mm - mf) * h["f"] ** p / (h["m"] ** p - h["f"] ** p)
                s_p, s_l, s_e = f"{p:.2f}", f"{lim:.4f}", f"{100*abs(mf-lim)/abs(lim):.2f} %"
            else:
                s_p, s_l, s_e = "non monotone", "—", "—"
            ligne = f"| {nom} | {mg:.4f} | {mm:.4f} | {mf:.4f} | {100*egm:.2f} % | {100*emf:.2f} % | {s_p} | {s_l} | {s_e} |"
            if ref:
                r = fn(ref)
                ligne += f" {r:.4f} | {100*abs(mm-r)/abs(r):.2f} % |"
            print(ligne)
            if i in GLOBALES:
                pire[w] = max(pire.get(w, (0, "")), (emf, nom))
        print()
    print("Plus grand écart m/f d'une métrique globale, par semaine : " + " ; ".join(f"SG {w} : {100*v[0]:.2f} % ({v[1]})" for w, v in pire.items()) + ".")


if __name__ == "__main__":
    sys.exit(main())
