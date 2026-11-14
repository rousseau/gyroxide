#!/usr/bin/env python3
"""Famille de maillages de SG 21 pour la convergence en maillage (P03, tâche 3) : maillage, tables des 16 semaines,
préparation des cibles, configuration.

    python3 tools/mesh_family.py <dossier de sortie> --maxvol 3e-6 [--mu-white 1.167]
    python3 tools/mesh_family.py <dossier> --reuse-mesh data/meshes/w21_left_610k.mesh   # tables seulement (vérification)

Le même pipeline que `tools/atlas_to_mesh.py`, `atlas_nodal_tables.py`, `atlas_core_tensors.py` et `adjoint_prep.py`, sans
pyvista, nibabel ni module TetGen : numpy, scipy et le programme `tetgen` en ligne de commande (externe, jamais lié).
La surface est celle de l'atlas (32 492 sommets, ordre de l'atlas, `-Y` : aucun point de Steiner sur la frontière) ; seul
l'intérieur change de finesse, par le volume maximal `a` de TetGen (`-pq1.5/10Ya<maxvol>Q`). Repère normalisé : celui de
`data/meshes/w21_left_610k.json`. Les tétraèdres à quatre nœuds de surface sont ensuite scindés
(`tools/split_boundary_tets.py`).

Écrit dans <dossier> :
  mesh_s.mesh (+ .json, .gr.txt, .nodal_a.npy, .nodal_h.npy, .nodal_core.npy : 16 nœuds de temps, SG 21 à 36),
  mesh.* (avant scission), prep/w22 .. w36 (target, edges, faces, owner, cortex, theta0), config.toml.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
ATLAS = Path(__import__("os").environ.get("OXIDE_DHCP_ATLAS", ROOT / "data/atlas_dhcp"))
WEEKS = range(21, 37)


def read_vtk(path):
    raw = Path(path).read_bytes()
    m = re.search(rb"POINTS (\d+) float\n", raw)
    n, o = int(m.group(1)), m.end()
    pts = np.frombuffer(raw, ">f4", 3 * n, o).reshape(n, 3).astype(float)
    o += 12 * n
    m = re.search(rb"POLYGONS (\d+) (\d+)\n", raw[o:])
    nf = int(m.group(1))
    tri = np.frombuffer(raw, ">i4", 4 * nf, o + m.end()).reshape(nf, 4)
    assert (tri[:, 0] == 3).all()
    return pts, tri[:, 1:].astype(np.int64)


def surface(week, kind):
    return read_vtk(ATLAS / f"fetal.week{week}.left.{kind}.surf.vtk")


def run_tetgen(q, tri, workdir, maxvol):
    ns = len(q)
    poly = workdir / "surf.poly"
    with open(poly, "w") as f:
        f.write(f"{ns} 3 0 0\n")
        np.savetxt(f, np.column_stack([np.arange(1, ns + 1), q]), fmt="%d %.12f %.12f %.12f")
        f.write(f"{len(tri)} 0\n")
        np.savetxt(f, tri + 1, fmt="1 0 0\n3 %d %d %d")
        f.write("0\n0\n")
    r = subprocess.run(["tetgen", f"-pq1.5/10Ya{maxvol:g}Q", str(poly)], capture_output=True, text=True)
    if r.returncode != 0 or not (workdir / "surf.1.ele").exists():
        sys.exit(f"TetGen a échoué : {r.stdout[-500:]} {r.stderr[-500:]}")
    nl = (workdir / "surf.1.node").read_text().split("\n")
    nn = int(nl[0].split()[0])
    node = np.array([l.split()[:4] for l in nl[1:1 + nn]], dtype=float)
    base = int(node[0, 0])
    el = (workdir / "surf.1.ele").read_text().split("\n")
    ne = int(el[0].split()[0])
    ele = np.array([l.split()[1:5] for l in el[1:1 + ne]], dtype=np.int64) - base
    return node[:, 1:], ele


def tet_volume(nodes, t):
    a, b, c, d = (nodes[t[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def read_mesh(path):
    tok = Path(path).read_text().split()
    n = int(tok[0])
    X = np.array(tok[1:1 + 3 * n], dtype=float).reshape(n, 3)
    nt = int(tok[1 + 3 * n])
    T = np.array(tok[2 + 3 * n:2 + 3 * n + 5 * nt], dtype=np.int64).reshape(nt, 5)[:, 1:] - 1
    return X, T[:, [0, 1, 3, 2]]  # Netgen : dernier couple échangé, on rend l'ordre direct


def boundary_faces(tets, nodes):
    local = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    allf = tets[:, local].reshape(-1, 3)
    opp = tets.reshape(-1)
    key = np.sort(allf, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = cnt[inv.ravel()] == 1
    faces, oppn = allf[bnd], opp[bnd]
    a_, b_, c_ = (nodes[faces[:, k]] for k in range(3))
    inward = np.einsum("ij,ij->i", np.cross(b_ - a_, c_ - a_), nodes[oppn] - a_) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces


def write_mesh(base, nodes, tets, faces):
    with open(str(base) + ".mesh", "w") as f:
        f.write(f"{len(nodes)}\n")
        np.savetxt(f, nodes, fmt="%.9f")
        f.write(f"{len(tets)}\n")
        np.savetxt(f, np.column_stack([np.ones(len(tets), int), tets[:, [0, 1, 3, 2]] + 1]), fmt="%d")
        f.write(f"{len(faces)}\n")
        np.savetxt(f, np.column_stack([np.ones(len(faces), int), faces + 1]), fmt="%d")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--maxvol", type=float, help="volume maximal TetGen, repère normalisé (3e-6 : environ 630 000 tétraèdres)")
    ap.add_argument("--reuse-mesh", help="maillage Netgen existant (sans scission) : tables seulement, pas de nouveau maillage")
    ap.add_argument("--smooth", type=int, default=200, help="passes de lissage de a et H (200 : tables de la référence)")
    ap.add_argument("--core-smooth", type=int, default=200)
    ap.add_argument("--h-min", type=float, default=0.3)
    ap.add_argument("--mu-white", type=float, default=1.167, help="mu_white de la configuration (mu_gray = 1) : 0,857 -> 1,167")
    ap.add_argument("--no-split", action="store_true")
    args = ap.parse_args()
    if (args.maxvol is None) == (args.reuse_mesh is None):
        sys.exit("--maxvol ou --reuse-mesh, un seul")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
    c0, d0, ns = np.array(frame["cog_mm"]), frame["maxd_mm"], frame["n_surface"]
    tf = lambda p: np.column_stack([-(p[:, 1] - c0[1]), p[:, 0] - c0[0], -(p[:, 2] - c0[2])]) / d0

    pial = {w: surface(w, "pial")[0] for w in WEEKS}
    wm = {w: surface(w, "wm")[0] for w in WEEKS}
    _, tri = surface(21, "pial")
    assert len(pial[21]) == ns
    q = tf(pial[21])

    # --- maillage de SG 21
    base = out / "mesh"
    if args.reuse_mesh:
        nodes, tets = read_mesh(args.reuse_mesh)
        faces = boundary_faces(tets, nodes)
    else:
        with tempfile.TemporaryDirectory() as td:
            nodes, tets = run_tetgen(q, tri, Path(td), args.maxvol)
        assert np.allclose(nodes[:ns], q, atol=1e-9), "TetGen a réordonné les sommets de surface"
        v = tet_volume(nodes, tets)
        tets[v < 0] = tets[v < 0][:, [0, 1, 3, 2]]
        faces = boundary_faces(tets, nodes)
    vol = tet_volume(nodes, tets)
    assert (vol > 0).all()
    assert faces.max() < ns and len(faces) == len(tri), "la frontière n'est pas la surface de l'atlas"
    assert np.array_equal(np.unique(np.sort(faces, 1), axis=0), np.unique(np.sort(tri, 1), axis=0))
    nn = len(nodes)
    write_mesh(base, nodes, tets, faces)
    e = np.concatenate([np.linalg.norm(nodes[faces[:, i]] - nodes[faces[:, j]], axis=1) for i, j in [(0, 1), (1, 2), (2, 0)]])
    meta = dict(frame)
    meta.update({"n_nodes": nn, "n_tets": len(tets), "n_surface": ns, "surface_edge_mean": float(e.mean()),
                 "maxvol": args.maxvol, "reuse_mesh": args.reuse_mesh})
    Path(str(base) + ".json").write_text(json.dumps(meta, indent=2))

    # --- masque de croissance : celui de SG 21 (mur médial), valeur du sommet le plus proche à l'intérieur
    gr21 = np.loadtxt(ROOT / "data/meshes/w21_left_610k.gr.txt")[:ns]
    nearest = cKDTree(q).query(nodes[ns:])[1]
    gr = np.concatenate([gr21, gr21[nearest]])
    np.savetxt(str(base) + ".gr.txt", gr, fmt="%.6f")
    to_nodes = lambda x: np.concatenate([x, x[nearest]], axis=0)

    # --- a et H des 16 semaines (atlas_nodal_tables.py)
    ed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    adj = sp.coo_matrix((np.ones(len(ed)), (ed[:, 0], ed[:, 1])), shape=(ns, ns))
    adj = ((adj + adj.T) > 0).astype(float).tocsr()
    deg = np.asarray(adj.sum(1)).ravel()

    def smooth(x, k):
        for _ in range(k):
            x = 0.5 * x + 0.5 * (adj @ x) / deg
        return x

    def vertex_area(p):
        t = 0.5 * np.linalg.norm(np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]]), axis=1)
        return np.bincount(tri.ravel(), weights=np.repeat(t / 3, 3), minlength=ns)

    area21 = smooth(vertex_area(pial[21]), args.smooth)
    A, H = [], []
    for w in WEEKS:
        A.append(to_nodes(np.sqrt(smooth(vertex_area(pial[w]), args.smooth) / area21) - 1))
        H.append(to_nodes(np.maximum(smooth(np.linalg.norm(pial[w] - wm[w], axis=1), args.smooth), args.h_min) / d0))
    A, H = np.array(A), np.array(H)
    np.save(str(base) + ".nodal_a.npy", A)
    np.save(str(base) + ".nodal_h.npy", H)

    # --- cœur : déplacement de la surface blanche depuis SG 21, lissé, prolongé harmoniquement (atlas_core_tensors.py)
    S = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / deg) @ adj
    pairs = np.concatenate([tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    W = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(nn, nn))
    W = ((W + W.T) > 0).astype(float).tocsr()
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    interior = np.arange(ns, nn)
    Lii = L[interior][:, interior].tocsc()
    Lib = L[interior][:, :ns]
    solve = spl.factorized(Lii)
    D0 = np.stack([nodes[tets[:, k]] - nodes[tets[:, 0]] for k in (1, 2, 3)], axis=2)
    D0inv = np.linalg.inv(D0)
    wsum = np.bincount(tets.ravel(), weights=np.repeat(vol, 4), minlength=nn)
    wm21 = tf(wm[21])
    core = np.zeros((len(WEEKS), nn, 6))
    for k, w in enumerate(WEEKS):
        us = tf(wm[w]) - wm21
        for _ in range(args.core_smooth):
            us = S @ us
        u = np.zeros((nn, 3))
        u[:ns] = us
        u[ns:] = np.column_stack([solve(-Lib @ us[:, d]) for d in range(3)])
        Y = nodes + u
        D = np.stack([Y[tets[:, j]] - Y[tets[:, 0]] for j in (1, 2, 3)], axis=2)
        F = D @ D0inv
        lam, vec = np.linalg.eigh(np.einsum("nji,njk->nik", F, F))
        U = np.einsum("nij,nj,nkj->nik", vec, np.sqrt(np.clip(lam, 1e-12, None)), vec)
        comp = np.stack([U[:, 0, 0], U[:, 1, 1], U[:, 2, 2], U[:, 0, 1], U[:, 0, 2], U[:, 1, 2]], axis=1)
        for c in range(6):
            core[k, :, c] = np.bincount(tets.ravel(), weights=np.repeat(comp[:, c] * vol, 4), minlength=nn) / wsum
    np.save(str(base) + ".nodal_core.npy", core)

    # --- scission des tétraèdres à quatre nœuds de surface, avec les tables et le masque
    if args.no_split:
        stem, extra = "mesh", []
    else:
        stem = "mesh_s"
        prep_src = out / "prep_unsplit"
        make_prep(prep_src, nodes, tri, ns, pial, tf, np.loadtxt(str(base) + ".gr.txt"), A)
        subprocess.run([sys.executable, str(ROOT / "tools/split_boundary_tets.py"), str(base) + ".mesh", str(out / "mesh_s.mesh"),
                        "--node-npy", str(base) + ".nodal_a.npy:1", str(base) + ".nodal_h.npy:1", str(base) + ".nodal_core.npy:1",
                        "--node-txt", str(base) + ".gr.txt", "--prep", f"{prep_src}:{out / 'prep'}"],
                       check=True, cwd=ROOT, capture_output=True)
        import shutil
        shutil.rmtree(prep_src)
    if args.no_split:
        make_prep(out / "prep", nodes, tri, ns, pial, tf, gr, A)

    # --- configuration : celle de la référence à 0,857 (mu2_s.toml, mu_white changé), tables de ce maillage
    cfg = (ROOT / "configs/mu_test/mu2_s.toml").read_text()
    cfg = re.sub(r"mu_white = [0-9.]+", f"mu_white = {args.mu_white}", cfg)
    for key in ("nodal_a", "nodal_h", "nodal_core"):
        cfg, k = re.subn(rf'"data/meshes/w21_left_610k_s\.{key}\.npy"', f'"{out / stem}.{key}.npy"', cfg)
        assert k, key
    cfg = cfg.replace("data/meshes/w21_left_610k_s.gr.txt", f"{out / stem}.gr.txt")
    (out / "config.toml").write_text(cfg)
    print(f"{out} : {nn} nœuds, {len(tets)} tétraèdres avant scission, surface {e.mean() * d0:.3f} mm, volume min {vol.min():.2e}")


def make_prep(root, nodes, tri, ns, pial, tf, gr, A):
    ed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    ed = np.unique(np.sort(ed, axis=1), axis=0)
    owner = np.arange(len(nodes))
    owner[ns:] = cKDTree(nodes[:ns]).query(nodes[ns:])[1]
    cortex = (gr[:ns] > 0.5).astype(float)
    for w in range(22, 37):
        d = root / f"w{w}"
        d.mkdir(parents=True, exist_ok=True)
        for name, arr in [("target", tf(pial[w])), ("edges", ed.astype(float)), ("faces", tri.astype(float)), ("owner", owner.astype(float)),
                          ("cortex", cortex), ("theta0", A[w - 21, :ns])]:
            np.save(d / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float64))


if __name__ == "__main__":
    main()
