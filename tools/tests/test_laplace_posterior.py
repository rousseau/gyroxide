"""Test de tools/laplace_posterior.py (noyau : Lanczos -> valeurs de Ritz -> Woodbury) sur un petit cas dont la réponse est connue :
deux blocs sur une grille 8 x 8, hessienne de rang 6 (donc retrouvée exactement par Lanczos en 10 pas), a priori beta * Laplacien + mu I
par bloc ; la covariance a posteriori de Woodbury est comparée à tau * inv(H + P) calculée en dense.
Aucune donnée, rien sur disque.

    python3 -m unittest discover -s tools/tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import laplace_posterior as lp  # noqa: E402

NX = 8
NA = NX * NX
BETA = 0.7
MU = 1e-2 * BETA
TAU = 0.05


def grid_laplacian():
    idx = np.arange(NA).reshape(NX, NX)
    rows, cols = [], []
    for i in range(NX):
        for j in range(NX):
            if i + 1 < NX:
                rows.append(idx[i, j]), cols.append(idx[i + 1, j])
            if j + 1 < NX:
                rows.append(idx[i, j]), cols.append(idx[i, j + 1])
    a = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(NA, NA))
    a = (a + a.T).tocsr()
    return (sp.diags(np.asarray(a.sum(axis=1)).ravel()) - a).toarray()


def lanczos(h, q0, steps):
    """Lanczos à réorthogonalisation complète (même schéma que `--hessvec lanczos`)."""
    q = q0 / np.linalg.norm(q0)
    vs, al, be = [], [], []
    for j in range(steps):
        w = h @ q
        al.append(q @ w)
        vs.append(q.copy())
        for _ in range(2):
            for v in vs:
                w -= (v @ w) * v
        b = np.linalg.norm(w)
        if j + 1 == steps or b < 1e-10 * max(abs(al[-1]), 1e-300):
            break
        be.append(b)
        q = w / b
    return np.array(al), np.array(be), np.array(vs)


class TestLaplacePosterior(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.n = 2 * NA
        u = np.linalg.qr(rng.standard_normal((self.n, 6)))[0]
        self.lam_true = np.array([40.0, 25.0, 12.0, 6.0, 3.0, 1.0])
        self.h = (u * self.lam_true) @ u.T
        lap = grid_laplacian()
        self.p1 = BETA * lap + MU * np.eye(NA)
        self.pfull = np.block([[self.p1, np.zeros((NA, NA))], [np.zeros((NA, NA)), self.p1]])
        self.gamma = TAU * np.linalg.inv(self.h + self.pfull)
        self.p1_inv = np.linalg.inv(self.p1)

    def test_ritz_retrouve_les_valeurs_propres(self):
        al, be, v = lanczos(self.h, np.random.default_rng(4).standard_normal(self.n), 10)
        lam, _ = lp.ritz(al, be, len(al))
        np.testing.assert_allclose(lam[:6], self.lam_true, rtol=1e-8)

    def test_woodbury_egale_inverse_dense(self):
        al, be, v = lanczos(self.h, np.random.default_rng(4).standard_normal(self.n), 10)
        lam, s = lp.ritz(al, be, len(al))
        w = (s.T @ v).T
        q, sm, l_, w_ = lp.woodbury(lambda x: self.p1_inv @ x, w, lam, NA)
        self.assertEqual(len(l_), 6)
        # variance a posteriori de toutes les coordonnées : diag de tau (P^-1 - Q S Q^T)
        diag = TAU * (np.concatenate([np.diag(self.p1_inv)] * 2) - np.einsum("ij,jk,ik->i", q, sm, q))
        np.testing.assert_allclose(diag, np.diag(self.gamma), rtol=1e-8)
        # une forme linéaire quelconque eta^T theta
        eta = np.random.default_rng(5).standard_normal(self.n)
        v_post = TAU * (eta[:NA] @ self.p1_inv @ eta[:NA] + eta[NA:] @ self.p1_inv @ eta[NA:]) - TAU * (q.T @ eta) @ sm @ (q.T @ eta)
        np.testing.assert_allclose(v_post, eta @ self.gamma @ eta, rtol=1e-8)

    def test_un_seul_bloc_libre_est_la_hessienne_conditionnelle(self):
        """Bloc 0 seul libre (bloc 1 figé) : la covariance est tau inv(H_00 + P), avec H_00 le bloc diagonal de H."""
        h00 = self.h[:NA, :NA]
        al, be, v = lanczos(h00, np.random.default_rng(6).standard_normal(NA), NA)
        lam, s = lp.ritz(al, be, len(al))
        w = (s.T @ v).T
        q, sm, l_, w_ = lp.woodbury(lambda x: self.p1_inv @ x, w, lam, NA)
        gamma0 = TAU * np.linalg.inv(h00 + self.p1)
        diag = TAU * (np.diag(self.p1_inv) - np.einsum("ij,jk,ik->i", q, sm, q))
        np.testing.assert_allclose(diag, np.diag(gamma0), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
