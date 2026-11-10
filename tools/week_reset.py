#!/usr/bin/env python3
"""Bras « modèle réinitialisé sur l'atlas w-1 » (persistance équitable, P02 tâche 5) : entrées d'une semaine.

    python3 tools/week_reset.py <SG précédente w-1> <dossier de sortie> [--mu-white 1.167]

L'état de référence sans contrainte est la géométrie de l'atlas dHCP à la semaine w-1 (pial et substance blanche,
mêmes 32 492 sommets que SG 21), remaillée par TetGen (programme externe, `/usr/bin/tetgen`, jamais lié ; mêmes
réglages que `tools/atlas_to_mesh.py` : `-pq1.5/10Y` avec un volume maximal de 3e-6 (630 000 tétraèdres sur SG 21, 610 828 pour le
maillage de référence, qui a une arête interne de 0,0237 jusqu'au centre), les sommets de surface en tête, dans l'ordre de l'atlas),
dans le repère normalisé du maillage de SG 21 (`data/meshes/w21_left_610k.json`), puis les tétraèdres à quatre
nœuds de surface sont scindés (`tools/split_boundary_tets.py`). Une semaine de croissance mesurée de w-1 à w :
  a  = sqrt(A_i(w) / A_i(w-1)) - 1 (aires de sommet lissées, 20 passes), nulle au nœud 0 ;
  h  = distance pial-blanc lissée, plancher 0,3 mm, à w-1 puis à w ;
  U  = étirement du cœur de w-1 à w : déplacement de la surface blanche lissé à l'échelle des lobes (200 passes),
       prolongé harmoniquement à l'intérieur, étirement droit par tétraèdre moyenné aux nœuds (comme
       `tools/atlas_core_tensors.py`, mais relatif à w-1).
Deux nœuds de temps (t = 0 : état de référence, t = 1 : une semaine plus tard). Écrit dans <dossier> :
  mesh_s.mesh (+ .json, .gr.txt, .nodal_a.npy, .nodal_h.npy, .nodal_core.npy), prep/ (cible = atlas w brut dans le
  repère du maillage, cortex, arêtes, faces, owner, theta0), ckpt/ (état au repos, t = 1), config.toml.
Pas de dépendance à pyvista, tetgen (module Python) ni nibabel : numpy et scipy.
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
TETGEN = "tetgen"


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


def smoother(tri, ns):
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ns, ns))
    a = ((a + a.T) > 0).astype(float).tocsr()
    deg = np.asarray(a.sum(1)).ravel()
    return a, deg


def run_tetgen(q, tri, workdir, maxvol):
    """Maillage TetGen en ligne de commande : sommets de surface d'abord, dans l'ordre d'entrée."""
    ns = len(q)
    poly = workdir / "surf.poly"
    with open(poly, "w") as f:
        f.write(f"{ns} 3 0 0\n")
        np.savetxt(f, np.column_stack([np.arange(1, ns + 1), q]), fmt="%d %.12f %.12f %.12f")
        f.write(f"{len(tri)} 0\n")
        np.savetxt(f, tri + 1, fmt="1 0 0\n3 %d %d %d")
        f.write("0\n0\n")
    r = subprocess.run([TETGEN, f"-pq1.5/10Ya{maxvol:g}Q", str(poly)], capture_output=True, text=True)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prev", type=int, help="semaine de l'état de référence (w-1)")
    ap.add_argument("out")
    ap.add_argument("--maxvol", type=float, default=3e-6, help="volume maximal TetGen (repère normalisé) : 3e-6 donne environ 116 000 nœuds et 630 000 tétraèdres sur SG 21, comme le maillage de référence")
    ap.add_argument("--smooth", type=int, default=20)
    ap.add_argument("--core-smooth", type=int, default=200)
    ap.add_argument("--h-min", type=float, default=0.3)
    ap.add_argument("--mu-white", type=float, default=1.167, help="mu_white de la configuration (mu_gray = 1) : 0,857 -> 1,167")
    args = ap.parse_args()
    w0, w1 = args.prev, args.prev + 1
    out = Path(args.out)
    (out / "prep").mkdir(parents=True, exist_ok=True)
    frame = json.loads((ROOT / "data/meshes/w21_left_610k.json").read_text())
    c0, d0, ns = np.array(frame["cog_mm"]), frame["maxd_mm"], frame["n_surface"]
    tf = lambda p: np.column_stack([-(p[:, 1] - c0[1]), p[:, 0] - c0[0], -(p[:, 2] - c0[2])]) / d0

    pial0, tri = surface(w0, "pial")
    wm0, _ = surface(w0, "wm")
    pial1, _ = surface(w1, "pial")
    wm1, _ = surface(w1, "wm")
    assert len(pial0) == ns
    q = tf(pial0)

    # --- maillage de l'atlas w-1
    with tempfile.TemporaryDirectory() as td:
        nodes, tets = run_tetgen(q, tri, Path(td), args.maxvol)
    assert np.allclose(nodes[:ns], q, atol=1e-9), "TetGen a réordonné les sommets de surface"
    v = tet_volume(nodes, tets)
    tets[v < 0] = tets[v < 0][:, [0, 1, 3, 2]]
    assert (tet_volume(nodes, tets) > 0).all()
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
    assert faces.max() < ns and len(faces) == len(tri)
    nn = len(nodes)
    vol = tet_volume(nodes, tets)

    # --- masque de croissance : celui de SG 21 (mur médial), valeur du sommet le plus proche à l'intérieur
    gr21 = np.loadtxt(ROOT / "data/meshes/w21_left_610k.gr.txt")[:ns]
    nearest = cKDTree(q).query(nodes[ns:])[1]
    gr = np.concatenate([gr21, gr21[nearest]])
    to_nodes = lambda x: np.concatenate([x, x[nearest]], axis=0)

    # --- écriture du maillage Netgen, mêmes conventions que atlas_to_mesh.py
    base = out / "mesh"
    with open(str(base) + ".mesh", "w") as f:
        f.write(f"{nn}\n")
        np.savetxt(f, nodes, fmt="%.9f")
        f.write(f"{len(tets)}\n")
        np.savetxt(f, np.column_stack([np.ones(len(tets), int), tets[:, [0, 1, 3, 2]] + 1]), fmt="%d")
        f.write(f"{len(faces)}\n")
        np.savetxt(f, np.column_stack([np.ones(len(faces), int), faces + 1]), fmt="%d")
    np.savetxt(str(base) + ".gr.txt", gr, fmt="%.6f")
    e = np.concatenate([np.linalg.norm(nodes[faces[:, i]] - nodes[faces[:, j]], axis=1) for i, j in [(0, 1), (1, 2), (2, 0)]])
    meta = dict(frame)
    meta.update({"week_reference": w0, "n_nodes": nn, "n_tets": len(tets), "n_surface": ns, "surface_edge_mean": float(e.mean())})
    Path(str(base) + ".json").write_text(json.dumps(meta, indent=2))

    # --- tables d'une semaine de croissance (deux nœuds de temps)
    a_sm, deg = smoother(tri, ns)
    smooth = lambda x, k=args.smooth: _smooth(x, a_sm, deg, k)

    def vertex_area(p):
        t = 0.5 * np.linalg.norm(np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]]), axis=1)
        return np.bincount(tri.ravel(), weights=np.repeat(t / 3, 3), minlength=ns)

    a_row = np.sqrt(smooth(vertex_area(pial1)) / smooth(vertex_area(pial0))) - 1
    h0 = np.maximum(smooth(np.linalg.norm(pial0 - wm0, axis=1)), args.h_min) / d0
    h1 = np.maximum(smooth(np.linalg.norm(pial1 - wm1, axis=1)), args.h_min) / d0
    A = np.stack([to_nodes(np.zeros(ns)), to_nodes(a_row)])
    H = np.stack([to_nodes(h0), to_nodes(h1)])
    np.save(str(base) + ".nodal_a.npy", A)
    np.save(str(base) + ".nodal_h.npy", H)

    # cœur : étirement de la surface blanche de w-1 à w, prolongé harmoniquement
    S = 0.5 * sp.identity(ns) + 0.5 * sp.diags(1 / deg) @ a_sm
    us = tf(wm1) - tf(wm0)
    for _ in range(args.core_smooth):
        us = S @ us
    pairs = np.concatenate([tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    W = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(nn, nn))
    W = ((W + W.T) > 0).astype(float).tocsr()
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    interior = np.arange(ns, nn)
    solve = spl.factorized(L[interior][:, interior].tocsc())
    u = np.zeros((nn, 3))
    u[:ns] = us
    u[ns:] = np.column_stack([solve(-L[interior][:, :ns] @ us[:, d]) for d in range(3)])
    D0 = np.stack([nodes[tets[:, k]] - nodes[tets[:, 0]] for k in (1, 2, 3)], axis=2)
    Y = nodes + u
    D = np.stack([Y[tets[:, j]] - Y[tets[:, 0]] for j in (1, 2, 3)], axis=2)
    F = D @ np.linalg.inv(D0)
    lam, vec = np.linalg.eigh(np.einsum("nji,njk->nik", F, F))
    U = np.einsum("nij,nj,nkj->nik", vec, np.sqrt(np.clip(lam, 1e-12, None)), vec)
    comp = np.stack([U[:, 0, 0], U[:, 1, 1], U[:, 2, 2], U[:, 0, 1], U[:, 0, 2], U[:, 1, 2]], axis=1)
    wsum = np.bincount(tets.ravel(), weights=np.repeat(vol, 4), minlength=nn)
    core1 = np.stack([np.bincount(tets.ravel(), weights=np.repeat(comp[:, c] * vol, 4), minlength=nn) / wsum for c in range(6)], axis=1)
    core0 = np.tile(np.array([1.0, 1, 1, 0, 0, 0]), (nn, 1))
    np.save(str(base) + ".nodal_core.npy", np.stack([core0, core1]))

    # --- scission des tétraèdres à quatre nœuds de surface
    base_s = out / "mesh_s"
    subprocess.run([sys.executable, str(ROOT / "tools/split_boundary_tets.py"), str(base) + ".mesh", str(base_s) + ".mesh",
                    "--node-npy", str(base) + ".nodal_a.npy:1", str(base) + ".nodal_h.npy:1", str(base) + ".nodal_core.npy:1",
                    "--node-txt", str(base) + ".gr.txt"], check=True, cwd=ROOT, capture_output=True)

    # --- prep : cible = atlas w brut dans le repère du maillage ; arêtes, faces, cortex, propriétaire
    target = tf(pial1)
    ed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    ed = np.unique(np.sort(ed, axis=1), axis=0)
    tok = str(base_s) + ".mesh"
    nl = Path(tok).read_text().split("\n")
    n_s = int(nl[0])
    Xs = np.array([l.split() for l in nl[1:1 + n_s]], dtype=float)
    owner = np.arange(n_s)
    owner[ns:] = cKDTree(Xs[:ns]).query(Xs[ns:])[1]
    cortex = (np.loadtxt(str(base) + ".gr.txt")[:ns] > 0.5).astype(float)
    for name, arr in [("target", target), ("edges", ed.astype(float)), ("faces", tri.astype(float)), ("owner", owner.astype(float)),
                      ("cortex", cortex), ("theta0", a_row)]:
        np.save(out / "prep" / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float64))

    # --- point de contrôle au repos (t = 1) et configuration
    ck = out / "ckpt"
    ck.mkdir(exist_ok=True)
    nt_s = int(nl[1 + n_s])
    np.save(ck / "x.npy", Xs)
    np.save(ck / "v.npy", np.zeros_like(Xs))
    np.save(ck / "growth_prev.npy", np.zeros(nt_s))
    (ck / "state.json").write_text(json.dumps({"step": 0, "t": 1.0}))
    cfg = (ROOT / "configs/mu_test/mu2_s.toml").read_text()
    cfg = re.sub(r"mu_white = [0-9.]+", f"mu_white = {args.mu_white}", cfg)
    cfg = re.sub(r"^t = \[.*\]$", "t = [0.0, 1.0]", cfg, flags=re.M)
    for key in ("nodal_a", "nodal_h", "nodal_core"):
        cfg = re.sub(rf'"data/meshes/w21_left_610k_s\.{key}\.npy"', f'"{base_s}.{key}.npy"', cfg)
    cfg = cfg.replace("data/meshes/w21_left_610k_s.gr.txt", f"{base_s}.gr.txt")
    (out / "config.toml").write_text(cfg)
    print(f"SG {w0} -> {w1} : {nn} nœuds ({n_s} après scission), {len(tets)} tétraèdres, a médian {np.median(a_row):.3f}, "
          f"arête de surface moyenne {e.mean() * d0:.3f} mm, volume min {vol.min():.2e}, {out}")


def _smooth(x, a, deg, k):
    for _ in range(k):
        x = 0.5 * x + 0.5 * (a @ x) / deg
    return x


if __name__ == "__main__":
    main()
