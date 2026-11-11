#!/usr/bin/env python3
"""Vérité synthétique de P04 (tâche 1) : une croissance connue sur l'hémisphère, son équilibre, une cible
lissée comme l'atlas, puis l'erreur de récupération par échelle (modes de Laplace-Beltrami).

Étapes (une semaine, par défaut SG 29 : état de départ ckpt_w29_from_lbm29, croissance mesurée SG 29) :

    python3 tools/synthetic_truth.py truth  <dossier>            # theta_true.npy : (a, c, k) par sommet
    adjoint_week … --theta-init <dossier>/theta_true.npy --fit-fields c,k --stress   # équilibre -> x_stress.npy
    python3 tools/synthetic_truth.py target <dossier>            # prep/ de la cible lissée (target.npy)
    adjoint_week … <dossier>/prep …                              # inversion (protocole habituel)
    python3 tools/synthetic_truth.py analyse <dossier> <sortie d'inversion> [--law]   # erreur par échelle

Vérité = champs de la chaîne de référence (fit29_lbm : a par sommet, c et k développés de la loi) + une perturbation
lisse connue (bruit blanc lissé par S^200 et S^30, de l'ordre de 30 % de l'écart type du champ). Les champs c et k
partent de zéro dans l'inversion, a de la croissance mesurée (table de la config).
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
META = json.loads((DATA / "meshes/w21_left_610k_s.json").read_text())
NS, MM = META["n_surface"], META["maxd_mm"]
WEEK = 29
RUN = ROOT / "out/paper_r0857/fit29_lbm"
PREP = DATA / "full_paper_r0857/prep" / f"w{WEEK}"
TABLE = DATA / f"full_paper_r0857/slab.nodal_a.carry_lbm{WEEK}.npy"
# bandes de longueur d'onde (mm) : lobes, plis primaires, plis secondaires, détails
BANDS = [(40, 1e9, "lobes > 40 mm"), (15, 40, "primaires 15-40 mm"), (6, 15, "secondaires 6-15 mm"), (0, 6, "< 6 mm")]


def neighbours():
    e = np.load(PREP / "edges.npy").astype(int)
    n = len(np.load(PREP / "cortex.npy"))
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr()
    a.data[:] = 1.0
    return a


def smoother(adj):
    """S = 0,5 I + 0,5 moyenne des voisins (celui de adjoint_week)."""
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return 0.5 * sp.identity(adj.shape[0]) + 0.5 * sp.diags(1 / deg) @ adj


def power(s, v, k):
    for _ in range(k):
        v = s @ v
    return v


def law_fields():
    """c et k par sommet de la loi du fit29_lbm (mêmes cartes que full_chain.sh LAW=1)."""
    L = DATA / "full/law"
    theta = np.load(RUN / "theta_final.npy")
    base = [np.load(L / f"basis_{n}_w{WEEK - 1}_s100.npy") for n in ("d30", "d3o")] + [np.load(L / "basis_one_s100.npy")]
    lb = [np.load(p) for p in sorted((L / "lb40").glob("basis_lb*_s100.npy"))]
    nb_c, nb_k = len(base), len(base) + len(lb)
    assert len(theta) == NS + nb_c + nb_k, (len(theta), NS, nb_c, nb_k)
    ac, ak = theta[NS:NS + nb_c], theta[NS + nb_c:]
    c = sum(a * f for a, f in zip(ac, base))
    k = sum(a * f for a, f in zip(ak, base + lb))
    return theta[:NS], c, k


def start_a():
    return np.load(TABLE)[WEEK - 21, :NS]


def truth(out, seed=1, rel=0.3):
    a, c, k = law_fields()
    a0 = start_a()
    s = smoother(neighbours())
    rng = np.random.default_rng(seed)
    fields = {"a": a, "c": c, "k": k}
    ref = {"a": a - a0, "c": c, "k": k}
    pert = {}
    for name in fields:
        z = (power(s, rng.standard_normal(NS), 200) / 1.0, power(s, rng.standard_normal(NS), 30))
        z = [v / v.std() for v in z]
        pert[name] = rel * ref[name].std() * (z[0] + z[1]) / np.sqrt(2)
    t = {n: fields[n] + pert[n] for n in fields}
    t["a"] = np.maximum(t["a"], 0.0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "theta_true.npy", np.concatenate([t["a"], t["c"], t["k"]]))
    np.savez(out / "fields.npz", a0=a0, **{f"{n}_chain": fields[n] for n in fields}, **{f"{n}_true": t[n] for n in fields})
    for n in fields:
        print(f"{n}: chaîne std {ref[n].std():.4f} (moy {ref[n].mean():+.4f}), perturbation std {pert[n].std():.4f}, "
              f"vérité min {t[n].min():+.3f} max {t[n].max():+.3f}")
    print("écrit", out / "theta_true.npy", "; lancer adjoint_week --stress avec ce theta")


def rough(x, s):
    """Écart type de |x - S x| (mm) : rugosité à l'échelle de l'arête."""
    d = x - s @ x
    return float(np.sqrt((d ** 2).sum(axis=1).mean()) * MM)


def target(out, k=None):
    out = Path(out)
    s = smoother(neighbours())
    x = np.load(out / "x_stress.npy")[:NS]
    ref = rough(np.load(PREP / "target.npy"), s)
    table = [(j, rough(power(s, x, j), s)) for j in range(0, 31)]
    if k is None:   # plus petit k dont la rugosité tombe sous celle de l'atlas
        k = next((j for j, r in table if r <= ref), 30)
    print(f"rugosité |x - Sx| : atlas {ref * 1e3:.2f} µm ; modèle S^0 {table[0][1] * 1e3:.2f}, S^{k} {table[k][1] * 1e3:.2f} µm -> k = {k}")
    prep = out / "prep"
    prep.mkdir(exist_ok=True)
    for f in PREP.glob("*.npy"):
        if f.name != "target.npy":
            (prep / f.name).unlink(missing_ok=True)
            (prep / f.name).symlink_to(f.resolve())
    np.save(prep / "target.npy", np.ascontiguousarray(power(s, x, k)))
    json.dump({"smoothing_k": k, "atlas_roughness_um": ref * 1e3}, open(out / "target.json", "w"))
    print("écrit", prep)


def lb_modes(x, faces, n_modes=900):
    """Modes de Laplace-Beltrami (cotangentes, masse diagonale) : valeurs propres (mm^-2) et vecteurs."""
    x = x * MM
    n = len(x)
    i, j, kk = faces[:, 0], faces[:, 1], faces[:, 2]
    rows, cols, vals = [], [], []
    area = np.zeros(n)
    for a, b, c in ((i, j, kk), (j, kk, i), (kk, i, j)):    # angle en c, arête (a, b)
        u, v = x[a] - x[c], x[b] - x[c]
        cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-30)
        rows += [a, b]
        cols += [b, a]
        vals += [-0.5 * cot, -0.5 * cot]
    for a, b, c in ((i, j, kk),):
        ar = 0.5 * np.linalg.norm(np.cross(x[b] - x[a], x[c] - x[a]), axis=1)
        for v in (a, b, c):
            np.add.at(area, v, ar / 3)
    w = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    lap = sp.diags(-np.asarray(w.sum(axis=1)).ravel()) + w
    mass = sp.diags(area)
    mu, phi = sla.eigsh(lap.tocsc(), k=n_modes, M=mass.tocsc(), sigma=-1e-4, which="LM")
    o = np.argsort(mu)
    return mu[o], phi[:, o], area


def analyse(out, run, law=False):
    out, run = Path(out), Path(run)
    f = np.load(out / "fields.npz")
    xs = np.load(out / "x_stress.npy")[:NS]
    faces = np.load(PREP / "faces.npy").astype(int)
    cache = out / "lb_modes.npz"
    if cache.exists():
        z = np.load(cache)
        mu, phi, area = z["mu"], z["phi"], z["area"]
    else:
        mu, phi, area = lb_modes(xs, faces)
        np.savez(cache, mu=mu, phi=phi, area=area)
    lam = 2 * np.pi / np.sqrt(np.maximum(mu, 1e-12))
    theta = np.load(run / "theta_final.npy")
    if law:
        theta = expand_law(theta, run)
    fit = {"a": theta[:NS] - f["a0"], "c": theta[NS:2 * NS], "k": theta[2 * NS:3 * NS]}
    tru = {"a": f["a_true"] - f["a0"], "c": f["c_true"], "k": f["k_true"]}
    tru["c+k"], fit["c+k"] = tru["c"] + tru["k"], fit["c"] + fit["k"]

    def coef(v):
        return phi.T @ (area * v)

    def mean_w(v):
        return (area * v).sum() / area.sum()

    rows = []
    print(f"modes : {len(mu)} (longueur d'onde du dernier : {lam[-1]:.1f} mm)")
    print(f"{'champ':5} {'bande':22} {'modes':>6} {'||vérité||':>10} {'corr':>6} {'rapport':>8} {'erreur rel.':>11}")
    for name in ("a", "c", "k", "c+k"):
        ct, cf = coef(tru[name]), coef(fit[name])
        for lo, hi, label in BANDS:
            m = (lam >= lo) & (lam < hi) & (mu > 1e-6)
            if not m.any():
                continue
            nt, nf = np.linalg.norm(ct[m]), np.linalg.norm(cf[m])
            corr = float(ct[m] @ cf[m] / max(nt * nf, 1e-30))
            err = float(np.linalg.norm(ct[m] - cf[m]) / max(nt, 1e-30))
            rows.append(dict(champ=name, bande=label, modes=int(m.sum()), norme_vraie=float(nt), corr=corr,
                             rapport=float(nf / max(nt, 1e-30)), erreur_rel=err))
            print(f"{name:5} {label:22} {int(m.sum()):6d} {nt:10.4f} {corr:6.2f} {nf / max(nt, 1e-30):8.2f} {err:11.2f}")
    json.dump(rows, open(out / f"recuperation_{run.name}.json", "w"), indent=1)


def expand_law(theta, run):
    """theta de l'inversion avec loi -> (a, c, k) par sommet (mêmes cartes que truth)."""
    L = DATA / "full/law"
    base = [np.load(L / f"basis_{n}_w{WEEK - 1}_s100.npy") for n in ("d30", "d3o")] + [np.load(L / "basis_one_s100.npy")]
    lb = [np.load(p) for p in sorted((L / "lb40").glob("basis_lb*_s100.npy"))]
    nb_c = len(base)
    c = sum(a * f for a, f in zip(theta[NS:NS + nb_c], base))
    k = sum(a * f for a, f in zip(theta[NS + nb_c:], base + lb))
    return np.concatenate([theta[:NS], c, k])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "truth":
        truth(sys.argv[2])
    elif cmd == "target":
        target(sys.argv[2])
    elif cmd == "analyse":
        analyse(sys.argv[2], sys.argv[3], "--law" in sys.argv)
    else:
        sys.exit(__doc__)
