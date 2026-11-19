#!/usr/bin/env python3
"""Configuration de l'hémisphère droit à partir de celle de l'hémisphère gauche (E34, P02 tâche 4).

    python3 tools/hemi_config.py configs/mu_test/mu_r0857_s.toml configs/mu_test/mu_r0857_s_right.toml [meta.json]

Seul ce qui dépend du côté change : le nom du maillage (`w21_left_610k…` → `w21_right_610k…`, donc les tables
nodales et le masque de croissance), le plan médian (le point le plus médial de l'atlas, `medial_plane_y` du `.json`
du maillage, par défaut data/meshes/w21_right_610k.json : il est de l'autre côté du centre) et l'arête moyenne de surface (`mesh_spacing`). Matériau, dynamique,
contact, calendrier : tels quels (mêmes paramètres des deux côtés).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
meta_path = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "data/meshes/w21_right_610k.json"
text = src.read_text()
meta = json.loads(meta_path.read_text())
n = 0
for old, new in (("w21_left_610k", "w21_right_610k"), ("left hemisphere", "right hemisphere")):
    n += text.count(old)
    text = text.replace(old, new)
text, k = re.subn(r"^(midplane = )[-0-9.]+", rf"\g<1>{meta['medial_plane_y']:.4f}", text, flags=re.M)
n += k
text, k = re.subn(r"^(mesh_spacing = )[0-9.]+", rf"\g<1>{meta['surface_edge_mean']:.4f}", text, flags=re.M)
n += k
dst.write_text(text)
print(f"{dst} : {n} remplacements")
