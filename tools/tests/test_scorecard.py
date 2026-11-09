"""Tests de tools/scorecard.py et scorecard_table.py sur un petit cas synthétique (sphère de 642 sommets à plis
de forme connue, coque de tétraèdres pour l'épaisseur) : aucune donnée de sujet, rien hors du dossier temporaire.

    python3 -m unittest discover -s tools/tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scorecard as sc  # noqa: E402
import scorecard_table as table  # noqa: E402

R_MM = 30.0


def icosphere(level=3):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f, dtype=np.int64)


def folded(v, amp, k=3.0):
    """Sphère de R_MM mm dont le rayon est modulé par amp * sin(kx) sin(ky) sin(kz) (plis de forme connue)."""
    return R_MM * v * (1 + amp * np.sin(k * v[:, 0]) * np.sin(k * v[:, 1]) * np.sin(k * v[:, 2]))[:, None]


def rotation(angle=0.7):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def write_gii(path, pts, faces):
    import nibabel as nib
    img = nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(pts.astype(np.float32), intent="NIFTI_INTENT_POINTSET", datatype="float32"),
        nib.gifti.GiftiDataArray(faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE", datatype="int32")])
    nib.save(img, str(path))


class Toy:
    """Atlas synthétique (semaines 21 à 24, plis de plus en plus marqués) et maillage de l'atlas : coque de
    tétraèdres entre la surface et une sphère intérieure à 0,9 R (épaisseur 3 mm)."""
    AMP = {21: 0.0, 22: 0.02, 23: 0.04, 24: 0.06}
    COG = np.array([5.0, -3.0, 2.0])

    def __init__(self, d):
        d = Path(d)
        self.atlas, self.meshes = d / "atlas", d / "meshes"
        self.atlas.mkdir()
        self.meshes.mkdir()
        self.v, self.f = icosphere(3)
        self.n = len(self.v)
        for w, a in self.AMP.items():
            write_gii(self.atlas / f"fetal.week{w}.left.pial.surf.gii", folded(self.v, a) + self.COG, self.f)
        self.mm = R_MM
        c = self.COG
        json.dump({"n_surface": self.n, "n_nodes": 2 * self.n, "maxd_mm": self.mm, "cog_mm": c.tolist()},
                  open(self.meshes / "toy.json", "w"))
        np.savetxt(self.meshes / "toy.gr.txt", np.concatenate([np.ones(2 * self.n)]))
        np.save(self.meshes / "toy.nodal_h.npy", np.full((16, 2 * self.n), 3.0 / self.mm))
        ms = sc.MeshSet("toy", self.meshes)
        self.surf21 = ms.sim_mm(folded(self.v, 0.0) + self.COG) / self.mm          # repère de calcul, normalisé
        self.x0 = np.vstack([self.surf21, 0.9 * self.surf21])
        tets = []
        for a, b, c_ in self.f:
            a2, b2, c2 = a + self.n, b + self.n, c_ + self.n
            tets += [(a, b, c_, a2), (b, c_, a2, b2), (c_, a2, b2, c2)]
        with open(self.meshes / "toy.mesh", "w") as fh:
            fh.write(f"{2 * self.n}\n" + "\n".join(" ".join(repr(float(c)) for c in p) for p in self.x0) + f"\n{len(tets)}\n")
            fh.write("\n".join("1 " + " ".join(str(i + 1) for i in t) for t in tets) + "\n")
        self.tets = np.array(tets)

    def sim_surface(self, amp):
        """Surface de repère de calcul normalisée, plis d'amplitude amp."""
        return sc.MeshSet("toy", self.meshes).sim_mm(folded(self.v, amp) + self.COG) / self.mm

    def nodes(self, amp):
        s = self.sim_surface(amp)
        return np.vstack([s, 0.9 * s])


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.v, self.f = icosphere(3)
        self.cortex = np.ones(len(self.v), bool)
        self.cortex[:40] = False                               # un « mur médian »

    def card(self, x, y):
        return sc.scorecard(x, y, self.f, self.cortex)

    def test_identical_surface(self):
        y = folded(self.v, 0.05)
        m = self.card(y, y)
        for lv in m["dev_rigid_mm"].values():
            self.assertLess(lv["rms"], 1e-9)
        self.assertAlmostEqual(m["curvature_corr"], 1.0, places=12)
        for k in ("fold_depth_mm", "area_mm2", "gi", "folding_index"):
            self.assertAlmostEqual(m[k]["ratio"], 1.0, places=12)
        self.assertTrue(all(abs(r - 1) < 1e-9 for r in m["curvature_bands"]["ratio"]))

    def test_rigid_motion_is_invisible(self):
        y = folded(self.v, 0.05)
        x = folded(self.v, 0.03)
        a = sc.flatten(self.card(x, y))
        b = sc.flatten(self.card(x @ rotation() + np.array([10.0, -4.0, 7.0]), y))
        for k in a:
            if a[k] is not None:
                self.assertAlmostEqual(a[k], b[k], delta=1e-7 * max(1.0, abs(a[k])), msg=k)

    def test_known_folds(self):
        y = folded(self.v, 0.06)
        weak, strong = self.card(folded(self.v, 0.02), y), self.card(folded(self.v, 0.05), y)
        self.assertGreater(weak["dev_rigid_mm"]["vertex"]["rms"], strong["dev_rigid_mm"]["vertex"]["rms"])
        self.assertGreater(strong["curvature_corr"], weak["curvature_corr"])
        self.assertLess(weak["fold_depth_mm"]["ratio"], strong["fold_depth_mm"]["ratio"])
        # 30 mm de rayon, pli de 0,06 : l'écart rigide reste de l'ordre de l'amplitude (1,8 mm), pas au-delà
        self.assertLess(weak["dev_rigid_mm"]["vertex"]["rms"], 0.06 * R_MM)

    def test_other_triangulation_gives_intrinsic_metrics_only(self):
        v2, f2 = icosphere(2)
        m = sc.scorecard(30 * v2, folded(self.v, 0.04), f2, np.ones(len(v2), bool), atlas_faces=self.f,
                         atlas_cortex=np.ones(len(self.v), bool))
        self.assertIsNone(m["dev_rigid_mm"])
        self.assertIsNone(m["curvature_corr"])
        self.assertTrue(np.isfinite(m["gi"]["ratio"]) and np.isfinite(m["area_mm2"]["ratio"]))

    def test_deterministic(self):
        x, y = folded(self.v, 0.03), folded(self.v, 0.06)
        self.assertEqual(json.dumps(sc.clean(self.card(x, y))), json.dumps(sc.clean(self.card(x, y))))


class RunTest(unittest.TestCase):
    """Chaîne complète : run (positions normalisées) + atlas GIfTI + maillage → scorecard.json → tableau."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.toy = Toy(self.d)
        self.common = ["--atlas-dir", str(self.toy.atlas), "--meshes-dir", str(self.toy.meshes), "--mesh-name", "toy"]

    def tearDown(self):
        self.tmp.cleanup()

    def run_dir(self, name, amp, week):
        r = self.d / name
        (r / "checkpoint").mkdir(parents=True)
        np.save(r / "checkpoint/x.npy", self.toy.nodes(amp))
        (r / "checkpoint/state.json").write_text(json.dumps({"t": (week - 21) / 15}))
        (r / "run.log").write_text("  step  t  Uk  Ue  area  volume  dx  dy  ninv\n"
                                   "  100  0.2  1.5e-6  2.5e-2  5.1  0.8  1.0  1.0  0  1.0s\n")
        return r

    def test_run_scorecard(self):
        r = self.run_dir("run", 0.03, 23)
        self.assertEqual(sc.main([str(r), *self.common]), 0)                # semaine lue dans state.json
        card = json.loads((r / "scorecard.json").read_text())
        self.assertEqual(card["week"], 23)
        self.assertEqual(card["inputs"]["mask"], "cortex")
        m = card["metrics"]
        self.assertGreater(m["dev_rigid_mm"]["vertex"]["rms"], 0.1)
        self.assertEqual(m["diagnostics"]["n_inverted"], 0)
        self.assertEqual(m["diagnostics"]["residual"]["kinetic_energy"], 1.5e-6)
        # épaisseur : couche de 3 mm de la coque, étalonnée sur SG 21 ; de l'atlas : 3 mm
        self.assertAlmostEqual(m["thickness_mm"]["atlas"], 3.0, places=9)
        self.assertGreater(m["thickness_mm"]["ratio"], 0.9)
        self.assertLess(m["thickness_mm"]["ratio"], 1.1)

    def test_thickness_is_calibrated_on_sg21(self):
        r = self.run_dir("rest", 0.0, 21)
        sc.main([str(r), *self.common])
        m = json.loads((r / "scorecard.json").read_text())["metrics"]
        self.assertAlmostEqual(m["thickness_mm"]["ratio"], 1.0, places=9)
        self.assertLess(m["dev_rigid_mm"]["vertex"]["rms"], 1e-4)           # float32 du GIfTI

    def test_inverted_elements_are_counted(self):
        r = self.run_dir("inv", 0.02, 22)
        x = np.load(r / "checkpoint/x.npy")
        x[self.toy.n + 5] = 1.6 * x[5]                  # un nœud intérieur passé de l'autre côté de la surface
        np.save(r / "checkpoint/x.npy", x)
        sc.main([str(r), *self.common])
        n = json.loads((r / "scorecard.json").read_text())["metrics"]["diagnostics"]["n_inverted"]
        self.assertGreaterEqual(n, 1)

    def test_two_runs_identical_and_compare(self):
        r = self.run_dir("run", 0.03, 23)
        a, b = self.d / "a.json", self.d / "b.json"
        for o in (a, b):
            sc.main([str(r), "--out", str(o), *self.common])
        self.assertEqual(a.read_text(), b.read_text())
        self.assertEqual(sc.main(["--compare", str(a), str(b)]), 0)
        card = json.loads(b.read_text())
        card["metrics"]["area_mm2"]["ratio"] *= 1 + 1e-4
        b.write_text(json.dumps(card))
        self.assertEqual(sc.main(["--compare", str(a), str(b)]), 1)
        self.assertEqual(sc.main(["--compare", str(a), str(b), "--tol", "1e-3"]), 0)

    def test_persistence_and_table(self):
        out = self.d / "pers"
        self.assertEqual(sc.main(["--persistence", "--weeks", "22-24", "--out", str(out), *self.common]), 0)
        files = sorted(out.glob("persistence_sg*.json"))
        self.assertEqual([f.name for f in files], [f"persistence_sg{w}.json" for w in (22, 23, 24)])
        devs = [json.loads(f.read_text())["metrics"]["dev_rigid_mm"]["vertex"]["rms"] for f in files]
        self.assertTrue(all(0 < d < 0.02 * R_MM * 1.5 for d in devs), devs)
        for w, amp in ((22, 0.02), (23, 0.04), (24, 0.06)):
            r = self.run_dir(f"w{w}", amp * 0.8, w)
            sc.main([str(r), "--label", "modele", *self.common])
        prefix = self.d / "tab/tableau"
        self.assertEqual(table.main([str(self.d), "--out", str(prefix)]), 0)
        md = Path(f"{prefix}.md").read_text()
        self.assertIn("| modele |", md)
        self.assertIn("atlas w−1 (persistance)", md)
        rows = Path(f"{prefix}.csv").read_text().strip().splitlines()
        self.assertEqual(len(rows), 1 + 3 + 3)                     # en-tête, 3 semaines de modèle, 3 de persistance
        self.assertGreater(Path(f"{prefix}.png").stat().st_size, 10_000)

    def test_atlas_npy_target(self):
        """--atlas-npy : la cible d'un .npy normalisé (atlas mis à l'échelle de la Table 1) à la place du GIfTI."""
        for w in (21, 22):
            (self.d / f"t{w}").mkdir()
            np.save(self.d / f"t{w}/target.npy", self.toy.nodes(self.toy.AMP[w]))
        r = self.run_dir("run", 0.035, 22)                 # ≠ atlas de la semaine 22 (0,02) : écarts non nuls
        a, b = self.d / "gii.json", self.d / "npy.json"
        sc.main([str(r), "--out", str(a), *self.common])
        sc.main([str(r), "--out", str(b), "--atlas-npy", str(self.d / "t{w}/target.npy"), *self.common])
        # le GIfTI est en float32 (~2e-6 mm sur 30 mm), le .npy en float64
        self.assertEqual(sc.main(["--compare", str(a), str(b), "--tol", "1e-3"]), 0)
        self.assertGreater(json.loads(b.read_text())["metrics"]["dev_rigid_mm"]["vertex"]["rms"], 0.1)


if __name__ == "__main__":
    unittest.main()
