#!/usr/bin/env python3
"""Approximation de Laplace de P04 (tâche 3) : écart type a posteriori des champs (a, c, k) par échelle.

    python3 tools/laplace_posterior.py <sortie hessvec> <dossier synthétique> <theta_final.npy> [--prep <dossier prep>]

<sortie hessvec> : dossier de `adjoint_week --hessvec lanczos` (hv_base.json, lanczos_alpha/beta/V.npy).
<dossier synthétique> : celui de synthetic_truth.py (fields.npz, lb_modes.npz) ; theta_final.npy : le point ajusté.

Modèle. L'objectif ajusté est L_données(theta) + R(theta), R = (beta/2) sum_arêtes (theta_i - theta_j)^2 par bloc. Avec un bruit gaussien
de variance sigma^2 par coordonnée, N L / sigma^2 est l'opposé de la log-vraisemblance, donc a posteriori
Gamma = tau (H_m + P)^-1, tau = sigma^2 / N, H_m = hessienne de la perte de données (Lanczos, rang K), P = beta * Laplacien du graphe
+ mu I par bloc (mu très petit : seulement pour rendre le mode constant propre ; les modes constants sont exclus des bandes).
tau est étalonné sur le résidu de l'ajustement : sigma^2 = 2 L / 3 (L = perte de données au point ajusté), tau = 2 L / (3 N).
Blocs : `hv_base.json` porte `blocks` (blocs de θ libres : 0 = a, 1 = c, 2 = k ; `--hv-blocks` de adjoint_week) ; les autres sont figés
(hessienne conditionnelle : l'écart type est celui de c, k sachant a, il sous-estime l'incertitude réelle si a est mal identifié).
Woodbury : Gamma = tau [P^-1 - Q S Q^T], Q = P^-1 W, S = (Lambda^-1 + W^T Q)^-1, (Lambda, W) = valeurs et vecteurs de Ritz positifs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_truth import BANDS, NS, PREP  # noqa: E402


def load_lanczos(out):
    al = np.load(out / "lanczos_alpha.npy")
    be = np.load(out / "lanczos_beta.npy")
    V = np.load(out / "lanczos_V.npy")          # m x n
    return al, be, V


def ritz(al, be, mm):
    T = np.diag(al[:mm]) + np.diag(be[:mm - 1], 1) + np.diag(be[:mm - 1], -1)
    lam, s = np.linalg.eigh(T)
    o = np.argsort(-lam)
    return lam[o], s[:, o]


def woodbury(pinv, W, lam, na, rel_cut=1e-6):
    """Gamma / tau = P^-1 - Q S Q^T pour H_m = W diag(lam) W^T (valeurs de Ritz positives) et P bloc-diagonal (même `pinv` par bloc).

    W : (nk na) x K, blocs de lignes dans l'ordre des blocs retenus. Rend (Q, S, lam retenus, W retenus)."""
    keep = (lam > 0) & (lam > rel_cut * lam[0])
    L_, W_ = lam[keep], W[:, keep]
    nk = W_.shape[0] // na
    Q = np.vstack([pinv(W_[b * na:(b + 1) * na]) for b in range(nk)])
    S = np.linalg.inv(np.diag(1 / L_) + W_.T @ Q)
    return Q, 0.5 * (S + S.T), L_, W_


def main():
    out, syn, thf = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    prep = Path(sys.argv[sys.argv.index("--prep") + 1]) if "--prep" in sys.argv else PREP
    base = json.loads((out / "hv_base.json").read_text())
    na, beta = base["na"], base["beta"]
    assert na == NS
    tau_scale = float(sys.argv[sys.argv.index("--tau-scale") + 1]) if "--tau-scale" in sys.argv else 1.0
    tau = 2 * base["loss"] / (3 * base["n_cortex"]) * tau_scale
    mu_p = 1e-2 * beta

    al, be, V = load_lanczos(out)
    m = len(al)
    n = V.shape[1]
    nb = n // na
    kept = base.get("blocks", list(range(nb)))
    names = {0: "a", 1: "c", 2: "k"}

    lam, s = ritz(al, be, m)
    lam10, _ = ritz(al, be, max(m - 10, 2))
    resid = np.abs(be[m - 1] * s[-1]) if len(be) >= m else np.full(m, np.nan)
    W = (s.T @ V).T                               # n x m, colonnes orthonormées (Ritz)
    W = np.vstack([W[b * na:(b + 1) * na] for b in kept])      # blocs retenus seulement
    print(f"Lanczos : {m} pas, n = {n} ({nb} blocs de {na}), blocs libres {[names[b] for b in kept]} ; tau = {tau:.3e} (perte {base['loss']:.4e}, N {base['n_cortex']})")
    pos = lam > 0
    print(f"valeurs de Ritz : {int(pos.sum())} positives, {int((~pos).sum())} négatives ; 10 plus grandes {np.array2string(lam[:10], precision=3)}")
    print(f"  plus négative {lam.min():.3e} ; plus grande aux pas {m} / {m - 10} : {lam[0]:.4e} / {lam10[0]:.4e}")
    if not np.isnan(resid).all():
        conv = int(np.sum(resid[:50] < 1e-2 * np.abs(lam[:50])))
        print(f"  Ritz convergés (résidu < 1 % de la valeur) parmi les 50 premiers : {conv}")

    e = np.load(prep / "edges.npy").astype(int)
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(na, na))
    adj = (adj + adj.T).tocsr()
    adj.data[:] = 1.0
    lap = sp.diags(np.asarray(adj.sum(axis=1)).ravel()) - adj
    P = (beta * lap + mu_p * sp.identity(na)).tocsc()
    lu = sla.splu(P)

    def pinv(x):
        return lu.solve(np.asarray(x, dtype=float))

    Q, S, L_, W_ = woodbury(pinv, W, lam, na)
    K = len(L_)

    z = np.load(syn / "lb_modes.npz")
    mu, phi, area = z["mu"], z["phi"], z["area"]
    lamb = 2 * np.pi / np.sqrt(np.maximum(mu, 1e-12))
    f = np.load(syn / "fields.npz")
    theta = np.load(thf)
    fit = {"a": theta[:na] - f["a0"], "c": theta[na:2 * na], "k": theta[2 * na:3 * na]}
    tru = {"a": f["a_true"] - f["a0"], "c": f["c_true"], "k": f["k_true"]}

    def coef(v):
        return phi.T @ (area * v)

    allc = {"a": (1, 0, 0), "c": (0, 1, 0), "k": (0, 0, 1), "c+k": (0, 1, 1), "c-k": (0, 1, -1)}
    combos = {k: v for k, v in allc.items() if all(v[b] == 0 for b in range(3) if b not in kept)}
    Pphi = pinv(phi)                                   # na x modes : P^-1 phi_j
    rows = []
    print(f"\n{'champ':5} {'bande':22} {'modes':>5} {'écart type':>10} {'||champ||':>9} {'écart/champ':>11} {'erreur vraie':>12} {'écart/erreur':>12} {'part prior retirée':>18}")
    for name, cf_ in combos.items():
        eta = np.concatenate([cf_[b] * phi for b in kept], axis=0)   # (nk na) x modes, bloc par bloc
        # variance a posteriori de eta_j^T theta, et a priori (prior seul : tau eta^T P^-1 eta)
        pe = np.concatenate([cf_[b] * Pphi for b in kept], axis=0)
        v_prior = tau * np.einsum("ij,ij->j", eta, pe)
        qe = Q.T @ eta                                   # K x modes
        v_post = v_prior - tau * np.einsum("ij,ij->j", qe, S @ qe)
        # champ estimé et vrai dans ces coordonnées
        if name in ("a", "c", "k"):
            fv, tv = fit[name], tru[name]
        else:
            sg = 1 if name == "c+k" else -1
            fv, tv = fit["c"] + sg * fit["k"], tru["c"] + sg * tru["k"]
        cfv, ctv = coef(fv), coef(tv)
        for lo, hi, label in BANDS:
            sel = (lamb >= lo) & (lamb < hi) & (mu > 1e-6)
            if not sel.any():
                continue
            sd = float(np.sqrt(max(v_post[sel].sum(), 0)))
            sd_prior = float(np.sqrt(max(v_prior[sel].sum(), 0)))
            nf = float(np.linalg.norm(cfv[sel]))
            err = float(np.linalg.norm(ctv[sel] - cfv[sel]))
            retire = 1 - float(v_post[sel].sum() / v_prior[sel].sum())
            rows.append(dict(champ=name, bande=label, modes=int(sel.sum()), ecart_type=sd, ecart_type_prior=sd_prior, norme_champ=nf,
                             ecart_sur_champ=sd / max(nf, 1e-30), erreur_vraie=err, ecart_sur_erreur=sd / max(err, 1e-30),
                             part_prior_retiree=retire))
            print(f"{name:5} {label:22} {int(sel.sum()):5d} {sd:10.4f} {nf:9.4f} {sd / max(nf, 1e-30):11.3f} {err:12.4f} {sd / max(err, 1e-30):12.3f} {retire:18.3f}")

    # écart type par sommet (constante retirée) : diag(Pi Gamma Pi)
    diag_pinv = np.empty(na)
    for i0 in range(0, na, 1000):
        i1 = min(na, i0 + 1000)
        cols = np.zeros((na, i1 - i0))
        cols[np.arange(i0, i1), np.arange(i1 - i0)] = 1.0
        diag_pinv[i0:i1] = np.einsum("ij,ij->j", cols, pinv(cols))
    sdv = {}
    one = np.ones(na)
    for ib, b in enumerate(kept):
        name = names[b]
        Qb = Q[ib * na:(ib + 1) * na]
        qS = Qb @ S
        g_diag = diag_pinv - np.einsum("ij,ij->i", qS, Qb)
        g1 = pinv(one) - qS @ (Qb.T @ one)
        g11 = float(one @ g1)
        var = tau * (g_diag - 2 * g1 / na + g11 / na ** 2)
        sdv[name] = np.sqrt(np.maximum(var, 0))
        print(f"écart type par sommet, {name} : médiane {np.median(sdv[name]):.4f}, p95 {np.percentile(sdv[name], 95):.4f}, "
              f"rms du champ ajusté {np.sqrt(np.mean((fit[name] - fit[name].mean()) ** 2)):.4f}")
    np.savez(out / "sd_par_sommet.npz", **sdv)
    json.dump(dict(tau=tau, mu_prior=mu_p, K=int(K), blocks=kept, ritz_top=lam[:20].tolist(), n_negatives=int((~pos).sum()), rows=rows),
              open(out / "laplace.json", "w"), indent=1)


if __name__ == "__main__":
    main()
