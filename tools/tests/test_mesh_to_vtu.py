"""Tests de tools/mesh_to_vtu.py sur un cube maillé en tétraèdres (aucune donnée de sujet) :

    python3 -m unittest discover -s tools/tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh_to_vtu as m2v  # noqa: E402


def cube_mesh(n=4, half=1.0):
    """Cube [-half, half]^3 en n^3 cubes de 6 tétraèdres chacun (découpe de Kuhn), orientation quelconque."""
    g = np.linspace(-half, half, n + 1)
    pts = np.array([(x, y, z) for x in g for y in g for z in g])
    idx = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k  # noqa: E731
    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                for path in ([0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]):
                    tets.append([c[q] for q in path])
    return pts, np.array(tets, dtype=np.int64)


def write_dot_mesh(path, pts, tets):
    with open(path, "w") as f:
        f.write(f"{len(pts)}\n")
        for p in pts:
            f.write(f"{p[0]!r} {p[1]!r} {p[2]!r}\n")
        f.write(f"{len(tets)}\n")
        for t in tets:
            f.write("1 " + " ".join(str(int(v) + 1) for v in t) + "\n")


def read_vtu(path):
    text = Path(path).read_text()
    import re

    def block(name):
        mt = re.search(rf'<DataArray[^>]*Name="{name}"[^>]*>(.*?)</DataArray>', text, re.S)
        return np.array(mt.group(1).split(), dtype=float)

    pts = re.search(r'<Points>\s*<DataArray[^>]*>(.*?)</DataArray>', text, re.S).group(1)
    return np.array(pts.split(), dtype=float).reshape(-1, 3), block("connectivity").astype(int).reshape(-1, 4), block


class MeshToVtu(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        pts, tets = cube_mesh()
        rng = np.random.default_rng(3)
        self.perm = rng.permutation(len(pts))  # nœuds mélangés : surface et intérieur mêlés
        inv = np.argsort(self.perm)
        self.pts, self.tets = pts[self.perm], inv[tets]
        self.flip = rng.random(len(tets)) < 0.5  # orientations mélangées
        self.tets[self.flip] = self.tets[self.flip][:, [0, 1, 3, 2]]
        self.mesh = self.dir / "c.mesh"
        write_dot_mesh(self.mesh, self.pts, self.tets)

    def tearDown(self):
        self.tmp.cleanup()

    def test_orientation_surface_en_tete_et_volume(self):
        out = self.dir / "c.vtu"
        r = m2v.convert(self.mesh, out)
        pts, tets, _ = read_vtu(out)
        v = m2v.signed_volumes(pts, tets)
        self.assertGreater(v.min(), 0.0)
        self.assertAlmostEqual(v.sum(), 8.0, places=9)
        self.assertGreater(r["echanges"], 0)  # des tétraèdres retournés ont bien été remis à l'endroit
        self.assertEqual(r["volumes_nuls"], 0)
        # n = 4 : 5^3 nœuds, 3^3 intérieurs, donc 98 nœuds de surface, tous en tête
        self.assertEqual(r["noeuds_surface"], 98)
        on = np.abs(pts).max(axis=1) == 1.0
        self.assertTrue(on[:98].all() and not on[98:].any())
        # l'ordre relatif d'origine est gardé parmi les nœuds de surface
        orig = np.abs(self.pts).max(axis=1) == 1.0
        np.testing.assert_allclose(pts[:98], self.pts[orig])

    def test_bord_tourne_vers_l_exterieur(self):
        pts, tets = cube_mesh()
        tets, _, _ = m2v.orient(pts, tets)
        f = m2v.boundary_faces(tets)
        n = np.cross(pts[f[:, 1]] - pts[f[:, 0]], pts[f[:, 2]] - pts[f[:, 0]])
        centre = pts[f].mean(axis=1)
        self.assertTrue((np.einsum("ij,ij->i", n, centre) > 0).all())

    def test_champs_cortex_et_substance_grise(self):
        # piale = bord du cube, cortex = 1 si x > 0 ; blanche = bord du cube réduit (demi-côté 0,6)
        ppts, ptets = cube_mesh()
        ptets, _, _ = m2v.orient(ppts, ptets)
        fp = m2v.boundary_faces(ptets)
        keep = np.unique(fp)
        remap = np.full(len(ppts), -1)
        remap[keep] = np.arange(len(keep))
        ppts, fp = ppts[keep], remap[fp]
        cortex = np.zeros(len(ppts))
        cortex[ppts[:, 0] > 0] = 1.0
        wpts, wtets = cube_mesh(n=4, half=0.6)
        wtets, _, _ = m2v.orient(wpts, wtets)
        wf = m2v.boundary_faces(wtets)
        keep = np.unique(wf)  # une vraie surface n'a que ses sommets de bord
        remap = np.full(len(wpts), -1)
        remap[keep] = np.arange(len(keep))
        wpts, wf = wpts[keep], remap[wf]
        np.save(self.dir / "cortex.npy", cortex)
        write_surf = lambda p, q: np.save(self.dir / p, q)  # noqa: E731
        write_surf("pial_pts.npy", ppts)
        # les surfaces GIfTI sont lues par nibabel : on remplace read_surface par les tableaux de la sphère de test
        orig = m2v.read_surface
        m2v.read_surface = lambda path: (ppts, fp) if "pial" in str(path) else (wpts, wf)
        try:
            out = self.dir / "c.vtu"
            r = m2v.convert(self.mesh, out, pial="pial.gii", fields=[f"cortex={self.dir / 'cortex.npy'}"],
                            white="white.gii")
        finally:
            m2v.read_surface = orig
        pts, _, block = read_vtu(out)
        self.assertEqual(r["champs"], ["cortex", "gray_matter"])
        cx, gm = block("cortex"), block("gray_matter")
        on = np.abs(pts).max(axis=1) == 1.0
        # nœuds de surface : le masque est celui du sommet de piale (x > 0 ; x = 0 non testé)
        np.testing.assert_array_equal(cx[on & (pts[:, 0] > 0)], 1.0)
        np.testing.assert_array_equal(cx[on & (pts[:, 0] < 0)], 0.0)
        cheb = np.abs(pts).max(axis=1)
        np.testing.assert_array_equal(gm[cheb <= 0.5], 0.0)  # dedans : substance blanche
        np.testing.assert_array_equal(gm[cheb >= 1.0], 1.0)  # dehors : substance grise

    def test_ligne_de_tetraedre_a_quatre_champs_refusee(self):
        bad = self.dir / "bad.mesh"
        lines = self.mesh.read_text().splitlines()
        i = int(lines[len(self.pts) + 1].split()[0]) and len(self.pts) + 2
        lines[i] = " ".join(lines[i].split()[1:])
        bad.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ValueError):
            m2v.read_mesh(bad)


if __name__ == "__main__":
    unittest.main()
