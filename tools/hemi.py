"""Aides communes aux outils qui doivent servir aux deux hémisphères (E34, P02 tâche 4).

Le côté se lit dans le nom du maillage (`w21_left_610k`, `w21_right_610k_s`) ; sans indication, c'est l'hémisphère
gauche, comme avant E34 : les sorties de la gauche ne changent pas. Seuls numpy, scipy et le programme externe TetGen
(`tetgen`, jamais lié) sont nécessaires : ni pyvista, ni le module Python tetgen.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ATLAS = Path(os.environ.get("OXIDE_DHCP_ATLAS", ROOT / "data/atlas_dhcp"))


def side_of(name, default="left"):
    """`left` ou `right` d'après un nom ou un chemin de maillage (`w21_right_610k_s` → `right`)."""
    m = re.search(r"(?:^|[_/.-])(left|right)(?:[_/.-]|$)", str(name))
    return m.group(1) if m else default


def mesh_base(side, split=False):
    """Nom du maillage de référence de l'hémisphère : `w21_<côté>_610k` (`_s` : tétraèdres frontière scindés)."""
    return f"w21_{side}_610k" + ("_s" if split else "")


def read_vtk(path):
    """Points (n, 3) et triangles (m, 3) d'une surface VTK binaire de l'atlas dHCP (lecture sans pyvista)."""
    raw = Path(path).read_bytes()
    m = re.search(rb"POINTS (\d+) float\n", raw)
    n, o = int(m.group(1)), m.end()
    pts = np.frombuffer(raw, ">f4", 3 * n, o).reshape(n, 3).astype(float)
    o += 12 * n
    m = re.search(rb"POLYGONS (\d+) (\d+)\n", raw[o:])
    nf = int(m.group(1))
    tri = np.frombuffer(raw, ">i4", 4 * nf, o + m.end()).reshape(nf, 4)
    assert (tri[:, 0] == 3).all(), "triangles seulement"
    return pts, tri[:, 1:].astype(np.int64)


def atlas_surface(week, side, kind):
    """Surface (`pial` ou `wm`) de l'atlas pour une semaine et un hémisphère : points, triangles."""
    return read_vtk(ATLAS / f"fetal.week{week}.{side}.{kind}.surf.vtk")


def to_frame(p, cog, d):
    """Repère normalisé de Tallinen : x' = -(y - cy)/d, y' = (x - cx)/d, z' = -(z - cz)/d."""
    return np.column_stack([-(p[:, 1] - cog[1]), p[:, 0] - cog[0], -(p[:, 2] - cog[2])]) / d


def tetgen_path():
    """Exécutable TetGen : $GYROXIDE_TETGEN, sinon `tetgen` du PATH."""
    return os.environ.get("GYROXIDE_TETGEN", "tetgen")


def tetrahedralize(q, tri, switches):
    """Maillage TetGen de la surface (q, tri) : (nœuds, tétraèdres), sommets de surface d'abord et dans l'ordre
    d'entrée (option -Y). TetGen est appelé en ligne de commande (licence AGPL : jamais lié)."""
    ns = len(q)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with open(td / "surf.poly", "w") as f:
            f.write(f"{ns} 3 0 0\n")
            np.savetxt(f, np.column_stack([np.arange(1, ns + 1), q]), fmt="%d %.12f %.12f %.12f")
            f.write(f"{len(tri)} 0\n")
            np.savetxt(f, tri + 1, fmt="1 0 0\n3 %d %d %d")
            f.write("0\n0\n")
        r = subprocess.run([tetgen_path(), "-" + switches.lstrip("-"), str(td / "surf.poly")], capture_output=True, text=True)
        if r.returncode != 0 or not (td / "surf.1.ele").exists():
            sys.exit(f"TetGen a échoué : {r.stdout[-500:]} {r.stderr[-500:]}")
        nl = (td / "surf.1.node").read_text().split("\n")
        nn = int(nl[0].split()[0])
        node = np.array([l.split()[:4] for l in nl[1:1 + nn]], dtype=float)
        base = int(node[0, 0])
        el = (td / "surf.1.ele").read_text().split("\n")
        ne = int(el[0].split()[0])
        ele = np.array([l.split()[1:5] for l in el[1:1 + ne]], dtype=np.int64) - base
    return node[:, 1:], ele
