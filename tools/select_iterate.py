#!/usr/bin/env python3
"""Sélection de l'itéré d'un ajustement (P02, tâche 2) : relit les journaux de `adjoint_week` et compare, semaine par
semaine, les itérés que `--keep-best --select-by rigid|loss|last` rendrait.

    python3 tools/select_iterate.py out/paper/fit{22..36}_lbm.log [--csv out.csv]

- `rigid` : l'itéré d'écart rigide minimal au niveau le plus fin (k = 0), c'est aussi la mesure d'évaluation ;
- `loss` : l'itéré de coût L minimal (perte de forme + régularisation) parmi ceux du dernier niveau, y compris son
  départ ; le coût change quand un niveau s'ajoute, d'où cette restriction ;
- `last` : le dernier itéré accepté.

Tout vient du journal (« iter k: shape loss … reg … », « iter k mm k=  0: … rigid rms … », « level n … »). C'est le
choix après coup, sans rejouer la chaîne : en chaîne, la semaine suivante partirait de l'itéré choisi, ce que cet
outil ne simule pas. L'itéré 0 est l'état de départ (équilibre de la semaine avant ajustement).
"""
import argparse
import re
import sys
from pathlib import Path

RE_LEVEL = re.compile(r"^level (\d+) \(smoothing \d+\): loss ([-+0-9.e]+), reg ([-+0-9.e]+)")
RE_ITER = re.compile(r"^iter (\d+): shape loss ([-+0-9.e]+), reg ([-+0-9.e]+),")
RE_MM = re.compile(r"^(start|iter (\d+)) mm k=\s*0: .*?\| rigid rms ([0-9.]+)")


def parse(text):
    """Itérés d'un journal : liste de dicts {k, loss, rigid, level}, avec k = 0 pour le départ.

    Le coût d'un itéré est celui de la ligne `iter k` ; celui du départ d'un niveau est celui de la ligne
    `level n` (coût du nouvel ensemble de niveaux, au dernier itéré accepté).
    """
    its = {}
    level, last_k = 0, 0
    for line in text.splitlines():
        m = RE_LEVEL.match(line)
        if m:
            level = int(m.group(1))
            # départ du niveau : l'état du dernier itéré accepté, avec le coût du nouveau niveau
            its.setdefault(last_k, {"k": last_k})["level_start"] = (level, float(m.group(2)) + float(m.group(3)))
            continue
        m = RE_ITER.match(line)
        if m:
            k = int(m.group(1))
            d = its.setdefault(k, {"k": k})
            d["loss"], d["level"] = float(m.group(2)) + float(m.group(3)), level
            last_k = k
            continue
        m = RE_MM.match(line)
        if m:
            k = 0 if m.group(1) == "start" else int(m.group(2))
            its.setdefault(k, {"k": k})["rigid"] = float(m.group(3))
    return its


def choose(its):
    """Les trois sélections : {rigid, loss, last} -> dict de l'itéré."""
    done = [d for d in (its[k] for k in sorted(its)) if "rigid" in d]
    if not done:
        raise ValueError("aucun itéré dans le journal")
    last_level = max((d.get("level", 0) for d in done), default=0)
    # candidats de `loss` : ceux du dernier niveau, plus l'itéré dont c'est le départ (coût du niveau)
    cands = []
    for d in done:
        if d.get("level", 0) == last_level and "loss" in d:
            cands.append({**d})
        ls = d.get("level_start")
        if ls and ls[0] == last_level:
            cands.append({**d, "loss": ls[1], "level": last_level})
    if not cands:  # un seul niveau sans coût de départ : tous les itérés comptent, le départ n'a pas de coût
        cands = [d for d in done if "loss" in d] or done
    return {
        "rigid": min(done, key=lambda d: d["rigid"]),
        "loss": min(cands, key=lambda d: d.get("loss", float("inf"))),
        "last": done[-1],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--csv", help="écrit aussi le tableau en CSV")
    args = ap.parse_args(argv)
    rows = []
    print(f"{'journal':<24}{'rigid : k':>10}{'mm':>8}{'loss : k':>10}{'mm':>8}{'last : k':>10}{'mm':>8}")
    for p in args.logs:
        ch = choose(parse(Path(p).read_text()))
        name = Path(p).name.replace(".log", "")
        r = [name] + [v for s in ("rigid", "loss", "last") for v in (ch[s]["k"], ch[s]["rigid"])]
        rows.append(r)
        print(f"{name:<24}{r[1]:>10}{r[2]:>8.3f}{r[3]:>10}{r[4]:>8.3f}{r[5]:>10}{r[6]:>8.3f}")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("journal,rigid_k,rigid_mm,loss_k,loss_mm,last_k,last_mm\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
