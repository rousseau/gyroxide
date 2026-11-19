#!/usr/bin/env python3
"""Maillage tétraédrique `.mesh` (CORTET / meshtool / BrainGrowth) -> `.vtu` lisible par `gyroxide run --mesh`.

Le `.mesh` de CORTET a trois sections : nœuds (x y z), tétraèdres (étiquette n1 n2 n3 n4, indices à partir de 1),
triangles de surface facultatifs. Le `.vtu` de gyroxide (`gyroxide-mesh::vtu`) est un VTK XML `UnstructuredGrid`
ASCII de tétraèdres, en mm. Conventions vérifiées ici, les mêmes que `gyroxide mesh` :

  * orientation : chaque tétraèdre a un volume signé positif, det(b-a, c-a, d-a) / 6 > 0 (échange des deux derniers
    nœuds sinon) ; un volume nul reste signalé, jamais corrigé ;
  * nœuds de surface en tête : les nœuds des triangles du bord (faces de tétraèdres qui n'en touchent qu'un) sont
    renumérotés les premiers, dans l'ordre d'origine, puis les nœuds intérieurs ;
  * champs par sommet : ceux d'une surface piale (cortex, ...) sont réinterpolés au sommet de piale le plus proche
    (comme le masque `cortex` de `gyroxide mesh`) ; `gray_matter` (1 substance grise, 0 blanche) vient de la surface
    blanche : un nœud est « blanc » s'il est du côté intérieur de la normale du sommet blanc le plus proche.

    python3 tools/mesh_to_vtu.py maillage.mesh sortie.vtu [--pial pial.surf.gii [--field cortex=cortex.shape.gii]]
                                                         [--white white.surf.gii]

Seul numpy est nécessaire ; scipy et nibabel seulement pour les champs et les surfaces GIfTI.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def read_mesh(path):
    """Nœuds (n, 3) et tétraèdres (m, 4), indices à partir de 0, d'un `.mesh` à trois sections."""
    with open(path) as fh:
        n = int(fh.readline().split()[0])
        pts = np.array([fh.readline().split()[:3] for _ in range(n)], dtype=np.float64)
        m = int(fh.readline().split()[0])
        rows = [fh.readline().split() for _ in range(m)]
    if any(len(r) < 5 for r in rows):
        raise ValueError(f"{path}: une ligne de tétraèdre n'a pas 5 champs (étiquette n1 n2 n3 n4)")
    tets = np.array([r[1:5] for r in rows], dtype=np.int64) - 1
    if tets.min() < 0 or tets.max() >= n:
        raise ValueError(f"{path}: indice de nœud hors de 1..{n}")
    return pts, tets


def signed_volumes(pts, tets):
    a, b, c, d = (pts[tets[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def orient(pts, tets):
    """Tétraèdres à volume positif ; renvoie (tets, nombre échangés, nombre de volumes nuls)."""
    tets = tets.copy()
    v = signed_volumes(pts, tets)
    flip = v < 0
    tets[flip] = tets[flip][:, [0, 1, 3, 2]]
    return tets, int(flip.sum()), int((v == 0).sum())


def boundary_faces(tets):
    """Triangles du bord (faces de tétraèdres vues une seule fois), tournés vers l'extérieur pour des tétraèdres de
    volume positif."""
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]])
    key = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[cnt[inv.ravel()] == 1]


def surface_nodes(tets):
    """Indices (triés) des nœuds du bord."""
    return np.unique(boundary_faces(tets))


def surface_first(pts, tets):
    """Renumérote : nœuds de surface en tête (ordre d'origine), puis intérieurs. Renvoie (pts, tets, perm, n_surface)
    où perm[nouveau] = ancien."""
    surf = surface_nodes(tets)
    on = np.zeros(len(pts), bool)
    on[surf] = True
    perm = np.concatenate([np.flatnonzero(on), np.flatnonzero(~on)])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return pts[perm], inv[tets], perm, len(surf)


def read_surface(path):
    import nibabel as nib

    g = nib.load(str(path))
    return g.darrays[0].data.astype(np.float64), g.darrays[1].data.astype(np.int64)


def read_vertex_field(path):
    """Un champ par sommet : `.shape.gii`, `.npy` ou texte."""
    p = str(path)
    if p.endswith(".gii"):
        import nibabel as nib

        return np.asarray(nib.load(p).darrays[0].data, dtype=np.float64)
    if p.endswith(".npy"):
        return np.load(p).astype(np.float64)
    return np.loadtxt(p, dtype=np.float64)


def vertex_normals(pts, faces):
    n = np.zeros_like(pts)
    fn = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def nearest_field(nodes, surf_pts, values):
    """Valeur du sommet de surface le plus proche de chaque nœud."""
    from scipy.spatial import cKDTree

    return values[cKDTree(surf_pts).query(nodes)[1]]


def gray_matter(nodes, white_pts, white_faces):
    """1 (substance grise) / 0 (blanche) : un nœud est blanc s'il est du côté intérieur (opposé à la normale
    sortante) du sommet blanc le plus proche."""
    from scipy.spatial import cKDTree

    _, i = cKDTree(white_pts).query(nodes)
    n = vertex_normals(white_pts, white_faces)
    inside = np.einsum("ij,ij->i", nodes - white_pts[i], n[i]) < 0
    return np.where(inside, 0.0, 1.0)


def write_vtu(path, pts, tets, fields):
    """Même mise en page que `gyroxide_mesh::vtu::write` (ASCII, Float64, Int64)."""
    n, m = len(pts), len(tets)
    for name, v in fields.items():
        if len(v) != n:
            raise ValueError(f"champ {name} : {len(v)} valeurs pour {n} nœuds")

    def arr(f, typ, name, comp, data, fmt):
        attrs = f' Name="{name}"' if name else ""
        attrs += f' NumberOfComponents="{comp}"' if comp else ""
        f.write(f'        <DataArray type="{typ}"{attrs} format="ascii">\n')
        np.savetxt(f, data, fmt=fmt)
        f.write("        </DataArray>\n")

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n')
        f.write(f'  <UnstructuredGrid>\n    <Piece NumberOfPoints="{n}" NumberOfCells="{m}">\n      <PointData>\n')
        for name, v in fields.items():
            arr(f, "Float64", name, 0, np.asarray(v, float), "%.17g")
        f.write("      </PointData>\n      <Points>\n")
        arr(f, "Float64", None, 3, pts, "%.17g")
        f.write("      </Points>\n      <Cells>\n")
        arr(f, "Int64", "connectivity", 0, tets, "%d")
        arr(f, "Int64", "offsets", 0, 4 * np.arange(1, m + 1), "%d")
        arr(f, "UInt8", "types", 0, np.full(m, 10), "%d")
        f.write("      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n")


def convert(mesh_path, out_path, pial=None, fields=(), white=None):
    """Convertit ; renvoie un résumé (dict) des contrôles."""
    pts, tets = read_mesh(mesh_path)
    tets, flipped, zero = orient(pts, tets)
    pts, tets, perm, n_surf = surface_first(pts, tets)
    out = {}
    if pial is not None:
        pp, _ = read_surface(pial)
        for spec in fields:
            name, _, src = spec.partition("=")
            v = read_vertex_field(src)
            if len(v) != len(pp):
                raise ValueError(f"champ {name} : {len(v)} valeurs pour {len(pp)} sommets de piale")
            out[name] = nearest_field(pts, pp, v)
    if white is not None:
        wp, wf = read_surface(white)
        out["gray_matter"] = gray_matter(pts, wp, wf)
    write_vtu(out_path, pts, tets, out)
    vol = signed_volumes(pts, tets)
    return dict(noeuds=len(pts), tetraedres=len(tets), noeuds_surface=n_surf, echanges=flipped, volumes_nuls=zero,
                volume_min=float(vol.min()), champs=sorted(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("vtu", type=Path)
    ap.add_argument("--pial", type=Path, help="surface piale d'origine (champs par sommet)")
    ap.add_argument("--field", action="append", default=[], metavar="NOM=FICHIER",
                    help="champ par sommet de la piale (.shape.gii, .npy, texte), réinterpolé ; répétable")
    ap.add_argument("--white", type=Path, help="surface blanche : écrit le champ gray_matter")
    a = ap.parse_args()
    if a.field and a.pial is None:
        ap.error("--field demande --pial")
    r = convert(a.mesh, a.vtu, a.pial, a.field, a.white)
    print(a.vtu, r)
    if r["volumes_nuls"]:
        print(f"ATTENTION : {r['volumes_nuls']} tétraèdre(s) de volume nul", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
