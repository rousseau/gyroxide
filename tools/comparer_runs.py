#!/usr/bin/env python3
"""E40 : compare deux exécutions de adjoint_week (dossiers `<sortie>/fit`) : fichiers x_<k>.npy et theta_<k>.npy au bit près, et ligne par
ligne les pertes imprimées dans fit.log.

Usage : comparer_runs.py <sortie A> <sortie B>
"""
import re
import sys
from pathlib import Path

import numpy as np


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    premier = None
    for nom in sorted({p.name for p in (a / "fit").glob("*.npy")} & {p.name for p in (b / "fit").glob("*.npy")},
                      key=lambda s: (re.sub(r"\d+", "", s), int(re.sub(r"\D", "", s) or 0))):
        xa, xb = np.load(a / "fit" / nom), np.load(b / "fit" / nom)
        identiques = xa.shape == xb.shape and xa.tobytes() == xb.tobytes()
        ecart = float(np.abs(xa.astype(np.float64) - xb.astype(np.float64)).max()) if xa.shape == xb.shape else float("nan")
        print(f"{nom:14s} {'identique au bit' if identiques else f'DIFFÈRE, écart max {ecart:.3e}'}")
        if not identiques and premier is None:
            premier = nom
    la = [l for l in (a / "fit.log").read_text().splitlines() if re.match(r"iter \d+: shape loss", l)]
    lb = [l for l in (b / "fit.log").read_text().splitlines() if re.match(r"iter \d+: shape loss", l)]
    for x, y in zip(la, lb):
        mx = re.match(r"(iter \d+): shape loss (\S+)", x)
        my = re.match(r"(iter \d+): shape loss (\S+)", y)
        print(f"{mx.group(1)} perte A {mx.group(2)} B {my.group(2)} {'=' if mx.group(2) == my.group(2) else '≠'}")
    print("premier fichier différent :", premier)


if __name__ == "__main__":
    main()
