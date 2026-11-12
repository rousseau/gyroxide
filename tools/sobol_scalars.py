"""Indices de Sobol de la perte à l'équilibre sur trois scalaires du modèle (E46, P04 tâche 2).

Facteurs : rapport de cisaillement mu_w / mu_g, module de compressibilité K, échelle de l'épaisseur H (décalage
uniforme delta de la correction d'épaississement c, que `--thick-mult` rend multiplicative : H -> H exp(delta)).
Les quatre scalaires du prompt comptent beta, poids de la régularisation des arêtes : il n'agit sur la perte que par
les champs ajustés (nul au départ, où les champs sont ceux de la croissance mesurée), il n'entre donc pas ici.

    python3 -I tools/sobol_scalars.py design  <dossier> <mu_w/mu_g> <K> [--n 32] [--largeur 0.3] [--dc 0.15]
    # lance ensuite : adjoint_week ... --newton-kick 5 --eval-points <dossier>/points.tsv  (écrit points.out)
    python3 -I tools/sobol_scalars.py analyse <dossier> [--boot 500]

Plan de Saltelli (A, B, AB_1..AB_3 ; 5 N points, suite de Sobol brouillée) ; estimateurs de Saltelli 2010 pour
l'indice du premier ordre et de Jansen pour l'indice total ; intervalles par bootstrap sur les N lignes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

NOMS = ["mu_w/mu_g", "K", "H (delta de c)"]


def plan(n, seed=0):
    from scipy.stats import qmc

    u = qmc.Sobol(d=6, scramble=True, seed=seed).random(n)
    a, b = u[:, :3], u[:, 3:]
    ab = []
    for i in range(3):
        m = a.copy()
        m[:, i] = b[:, i]
        ab.append(m)
    return a, b, ab


def indices(fa, fb, fab):
    """Premier ordre (Saltelli 2010) et total (Jansen) pour chaque facteur ; fab : liste de trois vecteurs."""
    v = np.var(np.concatenate([fa, fb]), ddof=1)
    s1 = [float(np.mean(fb * (f - fa)) / v) for f in fab]
    st = [float(0.5 * np.mean((fa - f) ** 2) / v) for f in fab]
    return s1, st


def design(args):
    out = Path(args.dossier)
    out.mkdir(parents=True, exist_ok=True)
    a, b, ab = plan(args.n)
    bas = np.array([args.ratio, args.k, 0.0])
    demi = np.array([args.largeur * args.ratio, args.largeur * args.k, args.dc])
    pts = [bas - demi + 2 * demi * m for m in [a, b] + ab]
    pts = np.vstack(pts)
    np.savetxt(out / "points.tsv", pts, fmt="%.10g", header="mu_w/mu_g K delta_c")
    (out / "plan.json").write_text(json.dumps({"n": args.n, "base": bas.tolist(), "demi_largeur": demi.tolist()}))
    print(f"{len(pts)} points dans {out / 'points.tsv'}")


def analyse(args):
    out = Path(args.dossier)
    plan_ = json.loads((out / "plan.json").read_text())
    n = plan_["n"]
    f = np.loadtxt(out / "points.out")
    if f.shape[0] < 5 * n:
        sys.exit(f"{f.shape[0]} points évalués sur {5 * n}")
    res = f[:, 1]
    f = f[:, 0]
    bons = np.isfinite(f) & (res < args.residu_max)
    fa, fb = f[:n], f[n : 2 * n]
    fab = [f[(2 + i) * n : (3 + i) * n] for i in range(3)]
    ok = bons[:n] & bons[n : 2 * n]
    for i in range(3):
        ok &= bons[(2 + i) * n : (3 + i) * n]
    fa, fb, fab = fa[ok], fb[ok], [x[ok] for x in fab]
    s1, st = indices(fa, fb, fab)
    rng = np.random.default_rng(0)
    m = len(fa)
    boot = []
    for _ in range(args.boot):
        idx = rng.integers(0, m, m)
        boot.append(indices(fa[idx], fb[idx], [x[idx] for x in fab]))
    s1b = np.array([b[0] for b in boot])
    stb = np.array([b[1] for b in boot])
    rapport = {
        "n_utiles": int(m),
        "n_plan": int(n),
        "moyenne": float(np.mean(np.concatenate([fa, fb]))),
        "ecart_type": float(np.std(np.concatenate([fa, fb]), ddof=1)),
        "facteurs": NOMS,
        "S1": s1,
        "S1_ic95": np.percentile(s1b, [2.5, 97.5], axis=0).T.tolist(),
        "ST": st,
        "ST_ic95": np.percentile(stb, [2.5, 97.5], axis=0).T.tolist(),
    }
    (out / "sobol.json").write_text(json.dumps(rapport, indent=1))
    print(f"{m} lignes utiles sur {n} ; perte {rapport['moyenne']:.4f} ± {rapport['ecart_type']:.4f}")
    for i, nom in enumerate(NOMS):
        print(
            f"{nom:16s} S1 {s1[i]:+.3f} [{rapport['S1_ic95'][i][0]:+.3f}, {rapport['S1_ic95'][i][1]:+.3f}]"
            f"  ST {st[i]:.3f} [{rapport['ST_ic95'][i][0]:.3f}, {rapport['ST_ic95'][i][1]:.3f}]"
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design")
    d.add_argument("dossier")
    d.add_argument("ratio", type=float)
    d.add_argument("k", type=float)
    d.add_argument("--n", type=int, default=32)
    d.add_argument("--largeur", type=float, default=0.3, help="demi-largeur relative de mu_w/mu_g et de K")
    d.add_argument("--dc", type=float, default=0.15, help="demi-largeur de delta de c (H entre exp(-dc) et exp(dc))")
    a = sub.add_parser("analyse")
    a.add_argument("dossier")
    a.add_argument("--boot", type=int, default=500)
    a.add_argument("--residu-max", type=float, default=1e-3, help="les points dont le Newton n'a pas convergé sont écartés")
    args = ap.parse_args()
    design(args) if args.cmd == "design" else analyse(args)


if __name__ == "__main__":
    main()
