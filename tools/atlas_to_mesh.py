#!/usr/bin/env python3
"""dHCP fetal atlas surface -> tetrahedral Netgen .mesh for gyroxide.

Reads one hemisphere of the dHCP fetal surface atlas (binary VTK or GIFTI),
tetrahedralises it with TetGen keeping the 32,492 atlas vertices as the first
surface nodes, in atlas order (TetGen -Y: no Steiner point on the boundary),
so every simulated surface node keeps its atlas vertex index and can be
compared with any other week of the atlas without registration.

Writes, next to `out`:
  out              Netgen .mesh in Tallinen's normalised frame (as stl_to_mesh.py)
  out.gr.txt       growth mask per node (0 on the medial wall), for
                   [growth_mask] kind = "file"
  out.json         normalisation (cog, maxd), atlas files, midplane position

    python3 tools/atlas_to_mesh.py ATLAS/fetal.week21.left.pial.surf.vtk \
        --wm ATLAS/fetal.week21.left.wm.surf.vtk data/meshes/w21_left.mesh

The medial wall is where the pial and white surfaces nearly coincide
(distance < --medial-thick mm) on medially facing vertices; the mask is then
smoothed over the surface so growth does not stop on a sharp edge.
"""
import argparse
import base64
import json
import re
import zlib
from pathlib import Path

import numpy as np
import tetgen


def read_surface(path):
    """Points (n, 3) and triangles (m, 3) of a binary VTK or GIFTI surface."""
    raw = Path(path).read_bytes()
    if path.endswith(".vtk"):
        m = re.search(rb"POINTS (\d+) float\n", raw)
        n, o = int(m.group(1)), m.end()
        pts = np.frombuffer(raw, ">f4", 3 * n, o).reshape(n, 3).astype(float)
        o += 12 * n
        m = re.search(rb"POLYGONS (\d+) (\d+)\n", raw[o:])
        nf = int(m.group(1))
        tri = np.frombuffer(raw, ">i4", 4 * nf, o + m.end()).reshape(nf, 4)
        assert (tri[:, 0] == 3).all(), "triangles only"
        return pts, tri[:, 1:].astype(np.int64)
    arrays = []
    text = raw.decode()
    for da in re.finditer(r"<DataArray(.*?)>.*?<Data>(.*?)</Data>", text, re.S):
        attrs, data = da.group(1), da.group(2).strip()
        b = base64.b64decode(data)
        if "GZip" in attrs:
            b = zlib.decompress(b)
        dtype = "<f4" if "FLOAT32" in attrs else "<i4"
        dims = [int(d) for d in re.findall(r'Dim\d="(\d+)"', attrs)]
        arrays.append(np.frombuffer(b, dtype).reshape(dims))
    pts, tri = arrays[0].astype(float), arrays[1].astype(np.int64)
    return pts, tri


def vertex_normals(p, tri):
    fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    nrm = np.zeros_like(p)
    for k in range(3):
        np.add.at(nrm, tri[:, k], fn)
    return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def smooth(values, tri, iterations):
    """Repeated one-ring averaging on the surface."""
    n = len(values)
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    deg = np.bincount(e[:, 0], minlength=n) + np.bincount(e[:, 1], minlength=n)
    v = values.astype(float)
    for _ in range(iterations):
        s = np.bincount(e[:, 0], v[e[:, 1]], n) + np.bincount(e[:, 1], v[e[:, 0]], n)
        v = 0.5 * v + 0.5 * s / deg
    return v


ap = argparse.ArgumentParser()
ap.add_argument("pial", help="atlas pial surface (.surf.vtk or .surf.gii)")
ap.add_argument("out")
ap.add_argument("--wm", help="matching white surface, for the medial-wall mask (default: gr = 1)")
ap.add_argument("--medial-thick", type=float, default=0.6, help="pial-white distance (mm) below which a medial vertex is wall")
ap.add_argument("--medial-dist", type=float, default=8.0, help="max distance (mm) from the most medial point for wall vertices")
ap.add_argument("--mask-smooth", type=int, default=10, help="smoothing iterations of the growth mask")
ap.add_argument("--maxvol", type=float, default=0.0, help="max tet volume, normalised units (0 = graded from the surface)")
ap.add_argument("--quality", type=float, default=1.5, help="TetGen radius-edge bound")
ap.add_argument("--cog", type=float, nargs=3, help="normalisation centre (default: mean of the surface points)")
ap.add_argument("--maxd", type=float, help="normalisation scale (default: max |p - cog| component)")
args = ap.parse_args()

p, tri = read_surface(args.pial)
ns = len(p)

gr_s = np.ones(ns)
if args.wm:
    w, tri_w = read_surface(args.wm)
    assert len(w) == ns and np.array_equal(tri_w, tri), "pial and white must share the atlas triangulation"
    thick = np.linalg.norm(p - w, axis=1)
    nrm = vertex_normals(p, tri)
    # Medial direction: from the hemisphere centre towards the midline (largest |x| extent side).
    side = np.sign(np.abs(p[:, 0].min()) - np.abs(p[:, 0].max()))  # +1 for a left hemisphere (x < 0)
    wall = (thick < args.medial_thick) & (side * nrm[:, 0] > 0.3)
    # Thin, medially facing cortex also exists in the insula / Sylvian floor:
    # keep only vertices close to the medial plane, then the largest connected
    # component, so the non-growing wall is one patch on the midline.
    x_med = p[:, 0].max() if side > 0 else p[:, 0].min()
    wall &= np.abs(p[:, 0] - x_med) < args.medial_dist
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = e[wall[e[:, 0]] & wall[e[:, 1]]]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns))
    _, lab = connected_components(g, directed=False)
    if wall.any():
        big = np.bincount(lab[wall]).argmax()
        wall &= lab == big
    gr_s = smooth(1.0 - wall, tri, args.mask_smooth)
    gr_s = np.clip((gr_s - 0.1) / 0.8, 0.0, 1.0)  # keep a fully non-growing core on the wall
    print(f"medial wall: {wall.sum()} vertices ({100 * wall.mean():.1f} %), gr < 0.5 on {(gr_s < 0.5).sum()}")

# Tallinen normalisation (a reflection: orientation is fixed below), as in stl_to_mesh.py.
c = np.array(args.cog) if args.cog else p.mean(axis=0)
d = args.maxd if args.maxd else np.abs(p - c).max()
q = np.column_stack([-(p[:, 1] - c[1]), p[:, 0] - c[0], -(p[:, 2] - c[2])]) / d

# TetGen switches: p = surface input, q = quality (radius-edge / min dihedral),
# Y = keep the surface triangles, a = max volume, Q = quiet. (The keyword form of
# pytetgen ignores maxvolume, so the switches are spelled out.)
sw = f"pq{args.quality}/10Y" + (f"a{args.maxvol:g}" if args.maxvol > 0 else "") + "Q"
nodes, tets = tetgen.TetGen(q, tri.astype(np.int32)).tetrahedralize(switches=sw)[:2]
tets = np.asarray(tets, dtype=np.int64)
assert np.allclose(nodes[:ns], q), "TetGen reordered the surface vertices"


def vol(t):
    a, b, cc, dd = (nodes[t[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(cc - a, dd - a)) / 6.0


neg = vol(tets) < 0
tets[neg] = tets[neg][:, [0, 1, 3, 2]]  # right-handed
assert (vol(tets) > 0).all()

# Boundary faces, oriented outwards; they must be the atlas triangles.
local = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
allf = tets[:, local].reshape(-1, 3)
opp = tets.reshape(-1)
key = np.sort(allf, axis=1)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
bnd = cnt[inv.ravel()] == 1
faces, oppn = allf[bnd], opp[bnd]
a, b, cc = (nodes[faces[:, k]] for k in range(3))
inward = np.einsum("ij,ij->i", np.cross(b - a, cc - a), nodes[oppn] - a) > 0
faces[inward] = faces[inward][:, [0, 2, 1]]
assert faces.max() < ns and len(faces) == len(tri), "boundary differs from the atlas surface"
assert np.array_equal(np.unique(np.sort(faces, 1), axis=0), np.unique(np.sort(tri, 1), axis=0))

# Growth mask on interior nodes: value of the nearest surface vertex.
gr = np.empty(len(nodes))
gr[:ns] = gr_s
if len(nodes) > ns:
    try:
        from scipy.spatial import cKDTree

        idx = cKDTree(q).query(nodes[ns:])[1]
    except ImportError:
        idx = np.array([np.argmin(((q - x) ** 2).sum(1)) for x in nodes[ns:]])
    gr[ns:] = gr_s[idx]

e = np.concatenate([np.linalg.norm(nodes[faces[:, i]] - nodes[faces[:, j]], axis=1) for i, j in [(0, 1), (1, 2), (2, 0)]])
v = vol(tets)
print(f"{len(nodes)} nodes ({ns} surface), {len(tets)} tets, {len(faces)} faces")
print(f"surface edge min {e.min():.4f} mean {e.mean():.4f} max {e.max():.4f} (normalised); 1 unit = {d:.3f} mm")
print(f"tet volume min {v.min():.2e} median {np.median(v):.2e} max {v.max():.2e}")
try:
    from scipy.spatial import cKDTree

    dc = cKDTree(q).query(nodes[tets].mean(axis=1))[0]
    h = np.cbrt(6 * np.sqrt(2) * v)  # edge of the regular tet of the same volume
    for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.0)]:
        m = (dc >= lo) & (dc < hi)
        if m.any():
            print(f"  depth {lo:.2f}-{hi:.2f}: {m.sum()} tets, equivalent edge median {np.median(h[m]):.4f}")
except ImportError:
    pass

# The medial plane in the normalised frame: y' = (x - cx) / d at the most medial atlas point.
x_med = p[:, 0].max() if side > 0 else p[:, 0].min()
y_mid = (x_med - c[0]) / d
print(f"medial plane y' = {y_mid:.4f}")

with open(args.out, "w") as f:
    f.write(f"{len(nodes)}\n")
    np.savetxt(f, nodes, fmt="%.9f")
    f.write(f"{len(tets)}\n")
    # Netgen order is left-handed: readers swap the last two nodes back.
    np.savetxt(f, np.column_stack([np.ones(len(tets), int), tets[:, [0, 1, 3, 2]] + 1]), fmt="%d")
    f.write(f"{len(faces)}\n")
    np.savetxt(f, np.column_stack([np.ones(len(faces), int), faces + 1]), fmt="%d")
np.savetxt(args.out + ".gr.txt", gr, fmt="%.6f")
Path(args.out + ".json").write_text(json.dumps({
    "pial": str(args.pial), "wm": args.wm, "n_surface": ns, "n_nodes": len(nodes), "n_tets": len(tets),
    "cog_mm": c.tolist(), "maxd_mm": float(d), "frame": "x'=-(y-cy)/d, y'=(x-cx)/d, z'=-(z-cz)/d",
    "medial_plane_y": float(y_mid), "surface_edge_mean": float(e.mean()),
}, indent=2))
