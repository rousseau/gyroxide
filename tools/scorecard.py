#!/usr/bin/env python3
"""Tableau de bord d'un run (atlas ou sujet) : `gyroxide eval` appelle ce script.

    python3 tools/scorecard.py <run> --week SG [--out scorecard.json] [--label NOM]
    python3 tools/scorecard.py --persistence [--weeks 22-36] --out <dossier>
    python3 tools/scorecard.py --compare a.json b.json [--tol 1e-6]

Écrit scorecard.json : les métriques du plan de travail, toutes calculées par les outils existants
(rien n'est recodé ici, voir docs/validation.md pour les définitions et les sources) :

  dev_rigid_mm      écart rigide au cortex de l'atlas (mm), RMS et p95, aux échelles S^200, S^50 et sommets
                    (tools/deviation_mm.py : la meilleure rotation + translation est recalculée à chaque échelle)
  curvature_corr    corrélation de la courbure moyenne lissée (S^2) avec l'atlas (model_gallery.metrics)
  fold_depth_mm     profondeur des plis, p95 - p5 de la profondeur le long des normales de la surface S^30
  area_mm2          aire du cortex (hors mur médian)
  gi                indice de gyrification A / aire de l'enveloppe convexe (fold_standard.py)
  folding_index     A / V^(2/3), colonne « folding index » de la Table 1 ISBI (paper_table.py)
  thickness_mm      épaisseur, convention de mota_law.py (volume de la couche corticale / aire, étalonné sur SG 21)
  curvature_bands   puissance de la courbure par bandes 1-2, 2-4, 4-8, 8-16 mm (fold_standard.py)
  diagnostics       éléments retournés (recomptés sur le maillage) et résidu final (journaux du run), si disponibles

Chaque grandeur d'aire, de gi, de profondeur, d'épaisseur ou de bande est donnée avec sa valeur pour le
modèle, pour l'atlas de la même semaine, et leur rapport modèle / atlas.

Entrées reconnues :
  - un .npy de positions ou un dossier de run qui en contient (x_stress.npy, checkpoint/x.npy, x.npy,
    x_<k>.npy) : repère de calcul (longueurs normalisées, × maxd_mm du maillage), tous les nœuds du maillage
    (ou les seuls sommets de surface) ;
  - une surface .surf.gii, ou un dossier de run BIDS (dernier *_pial.surf.gii, sequence.tsv) : en mm par
    défaut, en unités normalisées avec --units normalized (maillage de l'atlas avec normalization = "none").
Si la surface a les sommets de l'atlas (même nombre, mêmes triangles), les métriques sommet à sommet
(écart rigide, corrélation de courbure) sont calculées ; sinon elles valent null et seules les grandeurs
intrinsèques sont comparées à l'atlas.

L'atlas : surfaces piales dHCP, $OXIDE_DHCP_ATLAS (défaut data/atlas_dhcp/) ; maillages de l'atlas
(masque du cortex, épaisseurs, tétraèdres) : data/meshes/ (--meshes-dir).
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from fold_standard import Surface  # noqa: E402
from mota_law import Mesh, model_T, tet_volumes, vertex_areas  # noqa: E402
from surface_metrics import Surf, deviation_levels, edges_of, folding_index, metrics, smoother  # noqa: E402

SCHEMA = 1
DEV_LEVELS = {"s200": 200, "s50": 50, "vertex": 0}
BANDS_MM = [(1, 2), (2, 4), (4, 8), (8, 16)]
MESH_NAMES = ("w21_left_610k_s", "w21_left_610k", "w21_left_215k")
WEEKS = range(21, 37)


# ------------------------------------------------------------------ atlas et maillages

def atlas_dir():
    return Path(os.environ.get("OXIDE_DHCP_ATLAS", ROOT / "data/atlas_dhcp"))


def read_gii(path):
    import nibabel as nib
    img = nib.load(str(path))
    return np.asarray(img.agg_data("pointset"), dtype=float), np.asarray(img.agg_data("triangle"), dtype=np.int64)


def atlas_pial(week, adir):
    return read_gii(Path(adir) / f"fetal.week{week}.left.pial.surf.gii")[0]


def atlas_surface(week, adir, pattern=None, ms=None, normalized=False):
    """Cible atlas de la semaine, en mm. Par défaut la surface piale dHCP (repère de calcul si `normalized`).
    Avec `pattern` (ex. data/full/prep/w{w}/target.npy) : positions en unités normalisées d'un autre fichier, tel
    quel (la Table 1 ISBI évalue contre l'atlas mis à l'échelle par similitude sur le modèle, tools/make_full.py)."""
    if pattern:
        if ms is None:
            sys.exit("--atlas-npy demande un maillage de l'atlas (--meshes-dir / --mesh-name)")
        return np.load(pattern.format(w=week)).astype(float)[: ms.ns] * ms.mm
    q = atlas_pial(week, adir)
    return ms.sim_mm(q) if normalized else q


class MeshSet:
    """Un maillage de l'atlas et ses fichiers compagnons (données hors Git de data/meshes/)."""

    def __init__(self, name, meshes_dir):
        self.name, self.dir = name, Path(meshes_dir)
        meta = json.loads((self.dir / f"{name}.json").read_text())
        self.ns, self.mm, self.n_nodes = meta["n_surface"], meta["maxd_mm"], meta["n_nodes"]
        self.cog = np.array(meta["cog_mm"])
        self.mesh_path = self.dir / f"{name}.mesh"
        self.gr_path = self.dir / f"{name}.gr.txt"
        self.h_path = self.dir / f"{name}.nodal_h.npy"
        self._mesh = None

    def sim_mm(self, q):
        """Repère de calcul, en mm (sans la mise à l'échelle) : x' = -(y - cy), y' = x - cx, z' = -(z - cz)."""
        c = self.cog
        return np.column_stack([-(q[:, 1] - c[1]), q[:, 0] - c[0], -(q[:, 2] - c[2])])

    def cortex(self):
        return np.loadtxt(self.gr_path)[: self.ns] > 0.5 if self.gr_path.exists() else None

    def thickness_table(self):
        """H en mm, [semaine - 21, sommet de surface] (épaisseur pial-blanc de l'atlas), ou None."""
        return np.load(self.h_path)[:, : self.ns] * self.mm if self.h_path.exists() else None

    def mesh(self):
        if self._mesh is None and self.mesh_path.exists() and self.h_path.exists():
            self._mesh = Mesh(self.mesh_path, self.ns, np.load(self.h_path)[0, : self.ns])
        return self._mesh


def find_meshset(rows, meshes_dir, name=None):
    """Le maillage dont le nombre de nœuds (ou de sommets de surface) est `rows`."""
    names = [name] if name else MESH_NAMES
    for n in names:
        if (Path(meshes_dir) / f"{n}.json").exists():
            ms = MeshSet(n, meshes_dir)
            if name or rows in (ms.n_nodes, ms.ns):
                return ms
    return None


# ------------------------------------------------------------------ une surface

def atlas_thickness(ms, week, y_mm, faces, cortex):
    """T de l'atlas : épaisseur pial-blanc mesurée, pondérée par l'aire des sommets du cortex (mota_law.py)."""
    h = ms.thickness_table()
    if h is None:
        return None
    va = vertex_areas(y_mm, faces, len(y_mm))
    return float((h[week - 21] * va)[cortex].sum() / va[cortex].sum())


def surface_scores(x, faces, cortex):
    """Grandeurs intrinsèques d'une surface (x en mm) : aire du cortex, IG, indice de plissement,
    profondeur des plis, puissance de courbure par bandes."""
    S = Surf(faces, x, 1.0)
    d = S.depth(x)[cortex]
    fs = Surface(faces, cortex).metrics(x)
    scales = [1, 2, 4, 8, 16, 32]
    bands = [fs["bands"][scales.index(a)] for a, _ in BANDS_MM]       # bande a-2a = bands[index de a]
    return {
        "area_mm2": float(vertex_areas(x, faces, len(x))[cortex].sum()),
        "gi": fs["gi"],
        "folding_index": float(folding_index(x, S.f)),
        "fold_depth_mm": float(np.percentile(d, 95) - np.percentile(d, 5)),
        "bands": [float(b) for b in bands],
    }


def pair(model, atlas):
    return {"model": model, "atlas": atlas, "ratio": None if model is None or not atlas else model / atlas}


def scorecard(x, atlas, faces, cortex, *, atlas_faces=None, atlas_cortex=None, thickness=None, diagnostics=None):
    """Métriques de la surface x (mm) contre `atlas` (mm). `faces`, `cortex` : triangulation et masque de x ;
    `atlas_faces`, `atlas_cortex` : ceux de l'atlas s'ils diffèrent (sinon sommet à sommet). `thickness` :
    {"model": mm, "atlas": mm} ou None. Renvoie le dictionnaire "metrics" de scorecard.json."""
    atlas_faces = faces if atlas_faces is None else atlas_faces
    atlas_cortex = cortex if atlas_cortex is None else atlas_cortex
    corr = len(x) == len(atlas) and np.array_equal(faces, atlas_faces)
    m, a = surface_scores(x, faces, cortex), surface_scores(atlas, atlas_faces, atlas_cortex)
    out = {"dev_rigid_mm": None, "curvature_corr": None}
    if corr:
        levels = deviation_levels(x, atlas, cortex, smoother(edges_of(faces), len(x)), 1.0, sorted(DEV_LEVELS.values()))
        out["dev_rigid_mm"] = {k: levels[f"k{v}"]["rigid_mm"] for k, v in DEV_LEVELS.items()}
        out["curvature_corr"] = float(metrics(Surf(faces, atlas, 1.0), x, atlas, cortex)["corr"])
    for k in ("fold_depth_mm", "area_mm2", "gi", "folding_index"):
        out[k] = pair(m[k], a[k])
    out["thickness_mm"] = pair(**thickness) if thickness else None
    out["curvature_bands"] = {"scales_mm": [list(b) for b in BANDS_MM], "model": m["bands"], "atlas": a["bands"],
                              "ratio": [p / q for p, q in zip(m["bands"], a["bands"])]}
    out["diagnostics"] = diagnostics or {"n_inverted": None, "residual": None}
    return out


def count_inverted(x_norm, mesh):
    """Tétraèdres dont le volume signé a changé de signe depuis le maillage de référence."""
    def signed(x):
        e = np.stack([x[mesh.tets[:, k]] - x[mesh.tets[:, 0]] for k in (1, 2, 3)], 1)
        return np.linalg.det(e)
    return int((np.sign(signed(x_norm)) != np.sign(signed(mesh.x0))).sum())


# ------------------------------------------------------------------ lecture des runs

def find_positions(path):
    path = Path(path)
    if path.suffix == ".npy":
        return path
    for rel in ("x_stress.npy", "checkpoint/x.npy", "x.npy"):
        if (path / rel).exists():
            return path / rel
    ks = {int(q.stem.split("_")[1]): q for q in path.glob("x_*.npy") if q.stem.split("_")[1].isdigit()}
    return ks[max(ks)] if ks else None


def run_dir(path):
    p = Path(path)
    p = p if p.is_dir() else p.parent
    return p.parent if p.name == "checkpoint" else p


def read_diagnostics(d, n_inverted):
    """Résidu final lu dans les journaux du run (jamais recalculé) : trace/diagnostics.jsonl, sinon le
    dernier tableau de run.log (step t Uk Ue area volume dx dy ninv), sinon sequence.tsv ; et, pour un
    équilibre (adjoint_week), « released: |F| a -> b ». Valeurs absentes : null."""
    res = {"source": None, "kinetic_energy": None, "elastic_energy": None, "force_residual": None,
           "n_inverted_logged": None}
    jl = d / "trace/diagnostics.jsonl"
    log = d / "run.log"
    if jl.exists() and jl.read_text().strip():
        last = json.loads(jl.read_text().strip().splitlines()[-1])
        res.update(source="trace/diagnostics.jsonl", kinetic_energy=last.get("Uk"), elastic_energy=last.get("Ue"),
                   n_inverted_logged=last.get("ninv"))
    elif log.exists():
        rows = [ln.split() for ln in log.read_text().splitlines()
                if re.match(r"^\s*\d+\s+[-\d.e+]+(\s+[-\d.e+]+){6}\s+\d+", ln)]
        if rows:
            r = rows[-1]
            res.update(source="run.log", kinetic_energy=float(r[2]), elastic_energy=float(r[3]),
                       n_inverted_logged=int(r[8]))
        m = re.findall(r"released: \|F\| ([-\d.e+]+) -> ([-\d.e+]+)", log.read_text())
        if m:
            res["force_residual"] = float(m[-1][1])
    elif (d / "sequence.tsv").exists():
        lines = (d / "sequence.tsv").read_text().strip().splitlines()
        head, last = lines[0].split(), lines[-1].split()
        res.update(source="sequence.tsv", elastic_energy=float(last[head.index("energy")]))
    return {"n_inverted": n_inverted, "residual": None if res["source"] is None and res["force_residual"] is None
            else res}


def infer_week(d, explicit):
    if explicit is not None:
        return explicit
    sg = None
    st = d / "checkpoint/state.json"
    st = st if st.exists() else d / "state.json"
    if st.exists():
        sg = 21 + 15 * json.loads(st.read_text())["t"]          # t = (SG - 21) / 15, docs/model.md
    seq = d / "sequence.tsv"
    if seq.exists():
        lines = seq.read_text().strip().splitlines()
        v = lines[-1].split()[lines[0].split().index("ga_weeks")]
        sg = float(v) if v != "n/a" else sg
    if sg is None or abs(sg - round(sg)) > 0.01:
        sys.exit(f"{d} : semaine de gestation introuvable ou non entière ({sg}) ; la donner avec --week")
    return int(round(sg))


def load_cortex_file(path):
    p = Path(path)
    if p.suffix == ".gii":
        import nibabel as nib
        return np.asarray(nib.load(str(p)).darrays[0].data) > 0.5
    return np.load(p) > 0.5


# ------------------------------------------------------------------ commandes

def find_surface(run):
    run = Path(run)
    if run.suffix == ".gii":
        return run
    found = sorted(run.glob("*_pial.surf.gii")) if run.is_dir() else []
    return found[-1] if found else None             # pas écrits sur 7 chiffres : le dernier est l'état final


def evaluate(args):
    run = Path(args.run)
    d = run_dir(run)
    week = infer_week(d, args.week)
    adir = Path(args.atlas_dir) if args.atlas_dir else atlas_dir()
    pos = find_positions(run) if run.is_dir() or run.suffix == ".npy" else None
    surf = None if pos is not None else find_surface(run)
    if pos is None and surf is None:
        sys.exit(f"{run} : ni positions (.npy) ni surface .surf.gii")
    units = args.units or ("normalized" if pos is not None else "mm")
    faces = None
    if pos is not None:
        x = np.load(pos).astype(float)
    else:
        x, faces = read_gii(surf)
    ms = find_meshset(len(x), args.meshes_dir, args.mesh_name)
    x_norm = None
    if units == "normalized":
        if ms is None:
            sys.exit(f"{run} : aucun maillage de {args.meshes_dir} n'a {len(x)} nœuds ; voir --meshes-dir et --mesh-name")
        x_norm, x = x, x[: ms.ns] * ms.mm
    a_faces = read_gii(adir / "fetal.week21.left.pial.surf.gii")[1]
    a_mm = atlas_surface(week, adir, args.atlas_npy, ms, units == "normalized")
    faces = a_faces if faces is None else faces
    # masque du cortex : celui de l'atlas (growth mask du maillage) ; à défaut, tous les sommets des deux côtés
    a_real = ms.cortex() if ms is not None and len(a_mm) == ms.ns else None
    if len(x) == len(a_mm):
        cortex = load_cortex_file(args.cortex_mask) if args.cortex_mask else a_real
    else:
        cortex = load_cortex_file(args.cortex_mask) if args.cortex_mask and a_real is not None else None
    if cortex is None or a_real is None:
        cortex, a_cortex, mask = np.ones(len(x), bool), np.ones(len(a_mm), bool), "all"
    else:
        a_cortex, mask = a_real, "cortex"
    thickness = n_inv = None
    if x_norm is not None and mask == "cortex" and len(x_norm) == ms.n_nodes and ms.mesh() is not None:
        mesh = ms.mesh()
        x0 = mesh.x0[: ms.ns] * ms.mm                            # surface de SG 21 : étalonnage de l'épaisseur
        t21 = model_T(mesh.x0, mesh, vertex_areas(x0, faces, ms.ns), cortex, ms.mm)
        calib = atlas_thickness(ms, 21, x0, faces, cortex) / t21
        thickness = {"model": float(calib * model_T(x_norm, mesh, vertex_areas(x, faces, ms.ns), cortex, ms.mm)),
                     "atlas": atlas_thickness(ms, week, a_mm, a_faces, a_cortex)}
        n_inv = count_inverted(x_norm, mesh)
    out = scorecard(x, a_mm, faces, cortex, atlas_faces=a_faces, atlas_cortex=a_cortex, thickness=thickness,
                    diagnostics=read_diagnostics(d, n_inv))
    card = {"schema": SCHEMA, "label": args.label or d.name, "week": week, "metrics": out,
            "inputs": {"run": str(run), "positions": pos.name if pos else None, "surface": surf.name if surf else None,
                       "units": units, "mesh": ms.name if ms else None, "mask": mask,
                       "vertex_correspondence": out["dev_rigid_mm"] is not None,
                       "atlas": args.atlas_npy or f"dHCP fetal surface atlas, week {week}, left pial"}}
    dest = Path(args.out) if args.out else (d / "scorecard.json" if run.is_dir() else run.with_suffix(".scorecard.json"))
    write(dest, card)
    print(summary(card), f"→ {dest}")
    return 0


def persistence(args):
    """L'atlas de la semaine w - 1, aligné rigidement, contre celui de la semaine w (ligne de référence)."""
    adir = Path(args.atlas_dir) if args.atlas_dir else atlas_dir()
    name = args.mesh_name or "w21_left_610k"
    ms = MeshSet(name, args.meshes_dir) if (Path(args.meshes_dir) / f"{name}.json").exists() else None
    w0, w1 = map(int, args.weeks.split("-"))
    _, faces = read_gii(adir / "fetal.week21.left.pial.surf.gii")
    n = len(atlas_pial(21, adir))
    cortex = ms.cortex() if ms is not None and ms.ns == n else None
    mask = "all" if cortex is None else "cortex"
    cortex = np.ones(n, bool) if cortex is None else cortex
    for w in range(w0, w1 + 1):
        x, y = (atlas_surface(v, adir, args.atlas_npy, ms) for v in (w - 1, w))
        thickness = None
        if mask == "cortex" and ms.thickness_table() is not None:
            thickness = {"model": atlas_thickness(ms, w - 1, x, faces, cortex),
                         "atlas": atlas_thickness(ms, w, y, faces, cortex)}
        card = {"schema": SCHEMA, "label": "persistence", "week": w,
                "metrics": scorecard(x, y, faces, cortex, thickness=thickness),
                "inputs": {"model": f"atlas week {w - 1}", "units": "mm", "mesh": ms.name if ms else None,
                           "mask": mask, "vertex_correspondence": True,
                           "atlas": args.atlas_npy or f"dHCP fetal surface atlas, week {w}, left pial"}}
        write(Path(args.out) / f"persistence_sg{w}.json", card)
        print(summary(card))
    return 0


def write(dest, card):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(clean(card), indent=1) + "\n")


def clean(o):
    """Types JSON : numpy → Python, valeurs non finies → null."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    return None if isinstance(o, float) and not math.isfinite(o) else o


def flatten(metrics_):
    """Les métriques en colonnes scalaires {nom: valeur}, pour scorecard_table.py et compare."""
    m, f = metrics_, {}
    for k, v in (m.get("dev_rigid_mm") or {}).items():
        f[f"dev_{k}_rms_mm"], f[f"dev_{k}_p95_mm"] = v["rms"], v["p95"]
    f["curvature_corr"] = m.get("curvature_corr")
    for k in ("fold_depth_mm", "area_mm2", "gi", "folding_index", "thickness_mm"):
        for q in ("model", "atlas", "ratio"):
            f[f"{k}.{q}"] = (m.get(k) or {}).get(q)
    b = m["curvature_bands"]
    for (lo, hi), mo, at, ra in zip(b["scales_mm"], b["model"], b["atlas"], b["ratio"]):
        f[f"band_{lo}_{hi}.model"], f[f"band_{lo}_{hi}.atlas"], f[f"band_{lo}_{hi}.ratio"] = mo, at, ra
    dg = m.get("diagnostics") or {}
    f["n_inverted"] = dg.get("n_inverted")
    f["residual_kinetic_energy"] = (dg.get("residual") or {}).get("kinetic_energy")
    f["residual_force"] = (dg.get("residual") or {}).get("force_residual")
    return f


def summary(card):
    f = flatten(card["metrics"])
    g = lambda k, fmt: "—" if f.get(k) is None else format(f[k], fmt)
    return (f"{card['label']} SG {card['week']}: écart rigide S200/S50/sommets {g('dev_s200_rms_mm', '.2f')}/"
            f"{g('dev_s50_rms_mm', '.2f')}/{g('dev_vertex_rms_mm', '.2f')} mm, corr. courbure {g('curvature_corr', '.3f')}, "
            f"profondeur ×{g('fold_depth_mm.ratio', '.3f')}, aire ×{g('area_mm2.ratio', '.3f')}, "
            f"IG ×{g('gi.ratio', '.3f')}, épaisseur ×{g('thickness_mm.ratio', '.3f')}")


def compare(args):
    """Deux scorecard.json égaux à --tol près (relatif ; les entrées `inputs`, dépendantes de la machine, sont ignorées)."""
    a = flatten(json.loads(Path(args.compare[0]).read_text())["metrics"])
    b = flatten(json.loads(Path(args.compare[1]).read_text())["metrics"])
    worst, bad = 0.0, []
    for k in sorted(set(a) | set(b)):
        u, v = a.get(k), b.get(k)
        if u is None or v is None:
            if u is not v:
                bad.append((k, u, v))
            continue
        e = abs(u - v) / max(abs(u), abs(v), 1e-300) if u != v else 0.0
        worst = max(worst, e)
        if e > args.tol:
            bad.append((k, u, v))
    for k, u, v in bad:
        print(f"DIFFÉRENT {k}: {u} contre {v}")
    print(f"{len(a)} grandeurs, écart relatif maximal {worst:.2e} (tolérance {args.tol:g}) : "
          + ("identiques" if not bad else f"{len(bad)} différences"))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 prog="gyroxide eval")
    ap.add_argument("run", nargs="?", help="dossier de run, .npy de positions ou surface .surf.gii")
    ap.add_argument("--week", type=int, help="semaine de gestation de la cible atlas (défaut : lue dans le run)")
    ap.add_argument("--out", help="scorecard.json (défaut : dans le dossier du run) ; dossier avec --persistence")
    ap.add_argument("--label", help="nom de la ligne dans les tableaux (défaut : nom du dossier du run)")
    ap.add_argument("--units", choices=["mm", "normalized"],
                    help="unités des coordonnées (défaut : normalized pour un .npy, mm pour un .gii)")
    ap.add_argument("--cortex-mask", help=".npy ou .shape.gii du masque du cortex de la surface (sinon : tous les sommets)")
    ap.add_argument("--atlas-dir", help="surfaces de l'atlas dHCP (défaut : $OXIDE_DHCP_ATLAS ou data/atlas_dhcp)")
    ap.add_argument("--atlas-npy", metavar="MOTIF",
                    help="cible = positions normalisées d'un .npy par semaine, ex. data/full/prep/w{w}/target.npy "
                         "(atlas mis à l'échelle par similitude de la Table 1 ISBI), à la place des surfaces dHCP")
    ap.add_argument("--meshes-dir", default=str(ROOT / "data/meshes"), help="maillages de l'atlas (défaut : data/meshes)")
    ap.add_argument("--mesh-name", help="maillage de l'atlas (défaut : celui qui a le nombre de nœuds du run)")
    ap.add_argument("--persistence", action="store_true",
                    help="ligne de référence : atlas w-1 aligné rigidement contre l'atlas w, un fichier par semaine")
    ap.add_argument("--weeks", default="22-36", help="semaines de --persistence (défaut 22-36)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare deux scorecard.json")
    ap.add_argument("--tol", type=float, default=1e-6, help="tolérance relative de --compare (défaut 1e-6)")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(args)
    if args.persistence:
        if not args.out:
            ap.error("--persistence demande --out <dossier>")
        return persistence(args)
    if not args.run:
        ap.error("donner un run (ou --persistence, ou --compare)")
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
