"""Test de tools/sobol_scalars.py sur la fonction d'Ishigami (indices connus), sans modèle ni donnée.

    python3 -m unittest discover -s tools/tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sobol_scalars as sb  # noqa: E402


def ishigami(u):
    x = (u - 0.5) * 2 * np.pi
    return np.sin(x[:, 0]) + 7 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])


class TestSobol(unittest.TestCase):
    def test_ishigami(self):
        a, b, ab = sb.plan(4096)
        s1, st = sb.indices(ishigami(a), ishigami(b), [ishigami(m) for m in ab])
        # valeurs analytiques (a = 7, b = 0,1) : S1 = 0,3139 ; 0,4424 ; 0 -- ST = 0,5576 ; 0,4424 ; 0,2437
        for got, want in zip(s1, [0.3139, 0.4424, 0.0]):
            self.assertAlmostEqual(got, want, delta=0.04)
        for got, want in zip(st, [0.5576, 0.4424, 0.2437]):
            self.assertAlmostEqual(got, want, delta=0.04)

    def test_plan(self):
        a, b, ab = sb.plan(16)
        self.assertEqual(a.shape, (16, 3))
        for i in range(3):
            same = np.isclose(ab[i], a).all(axis=0)
            self.assertFalse(same[i])
            self.assertTrue(np.delete(same, i).all())
            self.assertTrue(np.allclose(ab[i][:, i], b[:, i]))


if __name__ == "__main__":
    unittest.main()
