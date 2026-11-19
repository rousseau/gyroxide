"""Tests de câblage des outils à deux hémisphères (tools/hemi.py, hemi_config.py ; E34) sur de petits cas synthétiques :
aucune donnée de sujet, rien hors du dossier temporaire.

    python3 -m unittest discover -s tools/tests -v
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import hemi  # noqa: E402


def write_vtk(path, pts, tri):
    """Surface VTK binaire (big-endian) au format de l'atlas dHCP."""
    n = len(pts)
    body = b"# vtk DataFile Version 3.0\nsurface\nBINARY\nDATASET POLYDATA\n"
    body += f"POINTS {n} float\n".encode() + pts.astype(">f4").tobytes()
    cells = np.column_stack([np.full(len(tri), 3), tri]).astype(">i4")
    body += f"\nPOLYGONS {len(tri)} {4 * len(tri)}\n".encode() + cells.tobytes()
    Path(path).write_bytes(body)


class HemiTest(unittest.TestCase):
    def test_side_of(self):
        self.assertEqual(hemi.side_of("w21_left_610k_s"), "left")
        self.assertEqual(hemi.side_of("data/meshes/w21_right_610k"), "right")
        self.assertEqual(hemi.side_of("data/meshes/w21_right_610k_s.mesh"), "right")
        self.assertEqual(hemi.side_of("quelque_chose"), "left")  # défaut : la gauche, comme avant E34
        self.assertEqual(hemi.mesh_base("right", split=True), "w21_right_610k_s")

    def test_frame_and_vtk(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0], [0.0, 0.0, 1.0], [2.0, 2.0, 2.0]])
        tri = np.array([[0, 1, 2], [1, 2, 3]])
        with tempfile.TemporaryDirectory() as td:
            write_vtk(Path(td) / "s.vtk", pts, tri)
            p, t = hemi.read_vtk(Path(td) / "s.vtk")
        np.testing.assert_allclose(p, pts)
        np.testing.assert_array_equal(t, tri)
        q = hemi.to_frame(pts, np.array([1.0, 1.0, 1.0]), 2.0)
        # x' = -(y - cy)/d, y' = (x - cx)/d, z' = -(z - cz)/d
        np.testing.assert_allclose(q[0], [-0.5, 0.0, -1.0])

    def test_config(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "left.toml"
            src.write_text('mesh_spacing = 0.0136   # x\na = "data/meshes/w21_left_610k_s.nodal_a.npy"\nmidplane = 0.3902   # y\n')
            meta = td / "meta.json"
            meta.write_text(json.dumps({"medial_plane_y": -0.5759, "surface_edge_mean": 0.0135}))
            subprocess.run([sys.executable, "-I", str(TOOLS / "hemi_config.py"), str(src), str(td / "right.toml"), str(meta)], check=True,
                           capture_output=True)
            out = (td / "right.toml").read_text()
        self.assertIn("w21_right_610k_s.nodal_a.npy", out)
        self.assertIn("midplane = -0.5759", out)
        self.assertIn("mesh_spacing = 0.0135", out)
        self.assertNotIn("left", out)

    @unittest.skipUnless(shutil.which("tetgen"), "TetGen absent")
    def test_tetgen_keeps_surface_order(self):
        # tétraèdre régulier subdivisé : octaèdre de 6 sommets, surface fermée
        q = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
        tri = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
        nodes, tets = hemi.tetrahedralize(q, tri, "pq1.5/10YQ")
        np.testing.assert_allclose(nodes[:6], q, atol=1e-9)
        self.assertGreaterEqual(len(tets), 4)


if __name__ == "__main__":
    unittest.main()
