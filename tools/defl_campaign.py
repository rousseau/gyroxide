#!/usr/bin/env python3
"""P05, hypothèse 5, déflation : tentatives successives au SG 36 (tools/defl_run.sh), chacune déflatant tous les
équilibres distincts trouvés avant elle. Reprend où elle s'est arrêtée (une tentative terminée a son journal).

    python3 tools/defl_campaign.py            # écrit out/E47/campagne.tsv

Équilibre « trouvé » : |F| final < 5e-5. « Distinct » : écart quadratique moyen des nœuds, au mouvement rigide près,
d'au moins 0,1 mm à chacun des précédents (bruit numérique de deux convergences du même état : 0,01 à 0,05 mm, E12).
Les tentatives qui ne convergent pas ou retournent des éléments sont consignées telles quelles.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from defl_dist import dist  # noqa: E402

D = "data/full_paper_r0857"
BASE = "out/E12/h5/base/x_0.npy"
# (étiquette, point de reprise, relaxation avant le Newton, p, sigma, options supplémentaires)
RUNS = [
    ("t0_w35", f"{D}/ckpt_w35_from_lbm35", 3000, None, None, ""),  # témoin sans déflation (E47, essai 1)
    ("d1", f"{D}/ckpt_w36_from_lbm36", 3000, 2, 1, ""),
    ("d2", f"{D}/ckpt_w35_from_lbm35", 3000, 2, 1, ""),
    ("d3", f"{D}/ckpt_w34_from_lbm34", 3000, 2, 1, ""),
    ("d4", f"{D}/ckpt_w33_from_lbm33", 3000, 2, 1, ""),
    ("d5", f"{D}/ckpt_w36", 3000, 2, 1, ""),
    ("d6", f"{D}/ckpt_w36_from_lbm36", 3000, 3, 0.1, ""),
    ("d7", f"{D}/ckpt_w35_from_lbm35", 3000, 2, 0.1, ""),
    ("d8", f"{D}/ckpt_w36_from_lbm36", 0, 2, 1, ""),
    ("d9", f"{D}/ckpt_w35_from_lbm35", 0, 2, 1, ""),
]


def parse(log):
    t = Path(log).read_text() if Path(log).exists() else ""
    m = re.search(r"^start: shape loss ([\d.e+-]+).*\|F\| ([\d.e+-]+) -> ([\d.e+-]+)", t, re.M)
    inv = re.search(r"^inverted elements at the fitted equilibrium: (\d+)", t, re.M)
    nw = len(re.findall(r"^newton \d+", t, re.M))
    return (float(m[1]), float(m[3]), int(inv[1])) if m and inv else None


def main():
    os.chdir(ROOT)
    known = [BASE]  # équilibres distincts, dans l'ordre
    rows = ["étiquette\tdépart\trelax0\tp\tsigma\tdéflatés\tperte\t|F| final\tretournés\tétat\tdist_min_mm\tplus_proche"]
    for tag, ckpt, relax0, p, sg, extra in RUNS:
        log = f"out/E47/{tag}.log"
        x = f"out/E47/{tag}/x_0.npy"
        done = parse(log) is not None and Path(x).exists()
        if not done:
            env = dict(os.environ, RELAX0=str(relax0), NEWTON="100", NEWTON_DEBUG="1")
            args = ["sh", "tools/defl_run.sh", tag, ckpt, ",".join(known) if p is not None else ""]
            if p is not None:
                args += ["--defl-p", str(p), "--defl-sigma", str(sg)]
            args += extra.split()
            r = subprocess.run(args, env=env, capture_output=True, text=True)
            print(tag, "code", r.returncode, flush=True)
        res = parse(log)
        if res is None or not Path(x).exists():
            rows.append(f"{tag}\t{ckpt}\t{relax0}\t{p}\t{sg}\t{len(known) if p is not None else 0}\t\t\t\techec (pas de sortie)\t\t")
            continue
        loss, f, inv = res
        xs = np.load(x)
        ds = [(dist(xs, np.load(k))[0], k) for k in known]
        dmin, near = min(ds)
        conv = f < 5e-5
        distinct = conv and dmin >= 0.1
        etat = ("distinct" if distinct else "déjà connu") if conv else "non convergé"
        if inv > 0:
            etat += f" ({inv} éléments retournés)"
        rows.append(f"{tag}\t{Path(ckpt).name}\t{relax0}\t{p}\t{sg}\t{len(known) if p is not None else 0}\t{loss:.4f}\t{f:.2e}\t{inv}\t{etat}\t{dmin:.4f}\t{Path(near).parent.name}")
        if distinct and inv == 0:
            known.append(x)
        Path("out/E47/campagne.tsv").write_text("\n".join(rows) + "\n")
    Path("out/E47/campagne.tsv").write_text("\n".join(rows) + "\n")
    Path("out/E47/equilibres.txt").write_text("\n".join(known) + "\n")


if __name__ == "__main__":
    main()
