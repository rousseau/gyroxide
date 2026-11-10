#!/usr/bin/env python3
"""Table 1 (croissance mesurée) de plusieurs bras de sensibilité côte à côte (P02, tâche 3).

    python3 tools/sensibilite_table.py label=dossier [label=dossier ...] [--label measured] [--weeks 26 28 30 33 36]

Chaque dossier contient `<label>_<SG>.json` (sorties de `gyroxide eval`). Écrit, par semaine, l'écart rigide RMS et p95
(mm), la corrélation de courbure, la profondeur de sillon (mm) et l'indice de plissement relatif, et l'écart relatif de
chaque bras au premier (la référence).
"""
import argparse
import json
from pathlib import Path

COLS = [("RMS", lambda m: m["dev_rigid_mm"]["vertex"]["rms"]), ("p95", lambda m: m["dev_rigid_mm"]["vertex"]["p95"]),
        ("corr", lambda m: m["curvature_corr"]), ("prof", lambda m: m["fold_depth_mm"]["model"]),
        ("IG", lambda m: m["folding_index"]["ratio"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--label", default="measured")
    ap.add_argument("--weeks", type=int, nargs="+", default=[26, 28, 30, 33, 36])
    a = ap.parse_args()
    arms = [x.split("=", 1) for x in a.arms]
    print(f"{'SG':>3} {'bras':<10}" + "".join(f"{n:>9}{'':>8}" for n, _ in COLS))
    for w in a.weeks:
        ref = None
        for name, d in arms:
            m = json.loads((Path(d) / f"{a.label}_{w}.json").read_text())["metrics"]
            v = [f(m) for _, f in COLS]
            ref = ref or v
            cells = "".join(f"{x:>9.3f}{(100 * (x / r - 1) if x is not r else 0):>+7.1f}%" for x, r in zip(v, ref))
            print(f"{w:>3} {name:<10}{cells}")


if __name__ == "__main__":
    main()
