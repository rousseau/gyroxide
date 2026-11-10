"""Test de tools/select_iterate.py sur un journal synthétique à deux niveaux (aucune donnée, rien sur disque).

    python3 -m unittest discover -s tools/tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import select_iterate as si  # noqa: E402

LOG = """start mm k=  0: absolute rms 2.0 p95 3.0 | rigid rms 1.800 p50 1.0 p95 2.0 max 5.0 | shape rms 1.0 p95 2.0
level 0 (smoothing 10): loss 3.000000e0, reg 1.000e-2
iter 1: shape loss 2.900000e0, reg 1.000e-2, step 1, CG 300 (res 1.0e-4), |theta - theta0| max 0.0013 rms 0.0001, 42s
iter 1 mm k=  0: absolute rms 2.0 p95 3.0 | rigid rms 1.790 p50 1.0 p95 2.0 max 5.0 | shape rms 1.0 p95 2.0
iter 2: shape loss 2.800000e0, reg 1.000e-2, step 1, CG 300 (res 1.0e-4), |theta - theta0| max 0.0013 rms 0.0001, 42s
iter 2 mm k=  0: absolute rms 2.0 p95 3.0 | rigid rms 1.700 p50 1.0 p95 2.0 max 5.0 | shape rms 1.0 p95 2.0
level 1 (smoothing 0): loss 4.000000e0, reg 1.000e-2
iter 3: shape loss 3.900000e0, reg 1.000e-2, step 1, CG 300 (res 1.0e-4), |theta - theta0| max 0.0013 rms 0.0001, 42s
iter 3 mm k=  0: absolute rms 2.0 p95 3.0 | rigid rms 1.750 p50 1.0 p95 2.0 max 5.0 | shape rms 1.0 p95 2.0
iter 4: step 1.000e0 rejected (loss 4.2 vs 3.9)
iter 4: shape loss 3.850000e0, reg 1.000e-2, step 0.5, CG 300 (res 1.0e-4), |theta - theta0| max 0.0013 rms 0.0001, 42s
iter 4 mm k=  0: absolute rms 2.0 p95 3.0 | rigid rms 1.760 p50 1.0 p95 2.0 max 5.0 | shape rms 1.0 p95 2.0
"""


class SelectIterate(unittest.TestCase):
    def test_trois_selections(self):
        ch = si.choose(si.parse(LOG))
        # écart rigide minimal : l'itéré 2, fin du premier niveau
        self.assertEqual((ch["rigid"]["k"], ch["rigid"]["rigid"]), (2, 1.7))
        # coût minimal du dernier niveau (départ à 4,01, puis 3,91 et 3,86) : l'itéré 4
        self.assertEqual((ch["loss"]["k"], ch["loss"]["rigid"]), (4, 1.76))
        self.assertEqual(ch["last"]["k"], 4)

    def test_depart_du_niveau_compte(self):
        # si aucun itéré du dernier niveau ne baisse le coût, l'itéré de départ du niveau est rendu
        log = LOG.replace("shape loss 3.900000e0", "shape loss 4.900000e0").replace("shape loss 3.850000e0", "shape loss 4.800000e0")
        ch = si.choose(si.parse(log))
        self.assertEqual(ch["loss"]["k"], 2)

    def test_journal_vide(self):
        with self.assertRaises(ValueError):
            si.choose(si.parse("rien\n"))


if __name__ == "__main__":
    unittest.main()
