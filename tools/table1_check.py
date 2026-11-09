#!/usr/bin/env python3
"""Compare les scorecard.json rejoués de la Table 1 (ISBI) à son table_metrics.json (tools/paper_table.py).

    python3 tools/table1_check.py out/scorecard/t1 ~/Exp/isbi_2027_gyroxide/table_metrics.json [--tol 0.01]

Les scorecards portent le label `measured`, `estimated` ou `original` et la semaine ; table_metrics.json
nomme les lignes « Measured\\ngrowth », « Estimated\\ngrowth », « Original\\ngrowth law » et « persistence ».
Pour chaque cellule : valeur rejouée, valeur de l'article, écart relatif. Code de retour 1 si un écart
dépasse --tol (défaut 1 %) ; les écarts sur les valeurs inférieures à --floor (défaut 0,05, RMS en mm)
sont jugés en absolu, pour ne pas gonfler les écarts de grandeurs presque nulles.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard_table import collect  # noqa: E402

# (clé de table_metrics.json, colonne de scorecard.flatten, rapport à l'atlas ?)
CELLS = [("rms", "dev_vertex_rms_mm"), ("p95", "dev_vertex_p95_mm"), ("corr", "curvature_corr"),
         ("depth", "fold_depth_mm.model"), ("fi", "folding_index.ratio")]
ROWS = {"measured": "Measured\ngrowth", "estimated": "Estimated\ngrowth", "original": "Original\ngrowth law"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scorecards")
    ap.add_argument("table_metrics")
    ap.add_argument("--tol", type=float, default=0.01, help="écart relatif toléré (défaut 1 %%)")
    ap.add_argument("--floor", type=float, default=0.05, help="sous cette valeur, écart absolu")
    args = ap.parse_args(argv)
    rows = collect([args.scorecards])
    ref = json.loads(Path(args.table_metrics).read_text())
    worst, bad, n = 0.0, [], 0
    print(f"{'ligne':<11}{'SG':>3}  {'grandeur':<6}{'rejoué':>9}{'article':>9}{'écart':>9}")
    for label, name in ROWS.items():
        for w in sorted(rows.get(label, {})):
            r = ref.get(name, {}).get(str(w))
            if r is None:
                continue
            for key, col in CELLS:
                a, b = rows[label][w].get(col), r.get(key)
                if a is None or b is None:
                    continue
                d = abs(a - b) if abs(b) < args.floor else abs(a / b - 1)
                n += 1
                worst = max(worst, d)
                flag = "" if d <= args.tol else "  <-- hors tolérance"
                if flag:
                    bad.append((label, w, key, a, b, d))
                print(f"{label:<11}{w:>3}  {key:<6}{a:>9.3f}{b:>9.3f}{100 * d:>8.2f}%{flag}")
    if "persistence" in rows:
        for w in sorted(rows["persistence"]):
            r = ref.get("persistence", {}).get(str(w))
            if r is None:
                continue
            for key, col in CELLS[:2]:
                a, b = rows["persistence"][w].get(col), r.get(key)
                d = abs(a / b - 1)
                n += 1
                worst = max(worst, d)
                flag = "" if d <= args.tol else "  <-- hors tolérance"
                if flag:
                    bad.append(("persistence", w, key, a, b, d))
                print(f"{'persistence':<11}{w:>3}  {key:<6}{a:>9.3f}{b:>9.3f}{100 * d:>8.2f}%{flag}")
    print(f"\n{n} cellules, écart maximal {100 * worst:.2f} %, {len(bad)} hors tolérance ({100 * args.tol:.1f} %)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
