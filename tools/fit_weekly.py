#!/usr/bin/env python3
"""Greedy week-by-week fit of the tangential expansion a_t to the atlas folding index.

Segment k goes from week 21+k to 22+k (8769 steps at step_factor 0.025). For each
segment, a_t is extended continuously with a slope s from CANDIDATES,
a(t_{k+1}) = a(t_k) + s / 15, the run resumes from the kept checkpoint of the
previous week, and the candidate whose folding index A / V^(2/3) (outer surface,
enclosed volume) is closest to the atlas value of that week is kept. Candidates
with an inverted element or a NaN are discarded (and killed as soon as it shows).
Near-ties (within --tie) go to the smaller s. Candidates are tried in increasing
s and the scan stops once the FI overshoots and the error grows again (FI is
assumed monotonic in s).

    python3 tools/fit_weekly.py --out out/atlas_fit [--last-week 36]

Restartable: the kept weeks are read back from <out>/fit.json.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [0.0, 0.25, 0.5, 1.0, 1.85]
ATLAS_FI = {21: 6.18, 22: 6.18, 23: 6.20, 24: 6.23, 25: 6.29, 26: 6.40, 27: 6.60, 28: 6.83, 29: 7.09,
            30: 7.43, 31: 7.82, 32: 8.10, 33: 8.34, 34: 8.59, 35: 8.78, 36: 8.92}
STEPS_PER_WEEK = 8769
N_KNOTS = 16  # weeks 21..36, t = k/15

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out/atlas_fit")
ap.add_argument("--base", default="configs/atlas_w21_tabulated.toml")
ap.add_argument("--mesh", default="data/meshes/w21_left_610k.mesh")
ap.add_argument("--bin", default="target-dev/release/gyroxide")
ap.add_argument("--last-week", type=int, default=36)
ap.add_argument("--tie", type=float, default=0.005, help="FI difference treated as a tie (smaller s wins)")
ap.add_argument("--candidates", type=float, nargs="+", default=CANDIDATES, help="slopes s tried each week")
ap.add_argument("--k-inverted", type=float, help="override [material] k_inverted for the new weeks")
ap.add_argument("--max-ninv", type=int, default=0, help="accept up to this many inverted tets at the segment end")
ap.add_argument("--max-uk", type=float, default=float("inf"), help="accept only if the last printed Uk is below this")
args = ap.parse_args()

out = (ROOT / args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
base = (ROOT / args.base).read_text()
state_file = out / "fit.json"
state = json.loads(state_file.read_text()) if state_file.exists() else {"a": [0.0], "weeks": []}


def mesh_arrays():
    """Tets (0-based) and rest positions, cached as .npz next to the fit."""
    cache = out / "mesh_cache.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X0"], z["T"]
    lines = (ROOT / args.mesh).read_text().split("\n")
    n = int(lines[0])
    X0 = np.loadtxt(lines[1:1 + n])
    m = int(lines[1 + n])
    T = np.loadtxt(lines[2 + n:2 + n + m], dtype=np.int64)[:, 1:] - 1
    np.savez(cache, X0=X0, T=T)
    return X0, T


X0, TETS = mesh_arrays()


def signed_vol(P):
    a, b, c, d = (P[TETS[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))


REST_SIGN = np.sign(signed_vol(X0))


def folding_index(vtk):
    L = Path(vtk).read_text().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    n = int(L[i].split()[1])
    X = np.loadtxt(L[i + 1:i + 1 + n])
    j = next(k for k, l in enumerate(L) if l.startswith("POLYGONS"))
    m = int(L[j].split()[1])
    F = np.loadtxt(L[j + 1:j + 1 + m], dtype=np.int64)[:, 1:]
    a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    V = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
    return A / V ** (2 / 3), A, V


def config(a_knots, name):
    """Base TOML with the a column replaced (held constant after the last fitted knot) and check_every = 1."""
    a = list(a_knots) + [a_knots[-1]] * (N_KNOTS - len(a_knots))
    txt = re.sub(r"(?m)^a = \[.*$", "a = [" + ", ".join(f"{x:.6f}" for x in a) + "]", base)
    txt = re.sub(r"(?m)^check_every = .*$", "check_every = 1", txt)
    txt = re.sub(r'(?m)^name = .*$', f'name = "{name}"', txt)
    if args.k_inverted is not None:
        txt = re.sub(r"(?m)^k_inverted = .*$", f"k_inverted = {args.k_inverted}", txt)
    return txt


ROW = re.compile(r"^\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s")


def run_candidate(week, s, a_knots, resume, cdir):
    """Runs one segment; returns a result dict (ok = no NaN, no inverted element)."""
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    toml = cdir / "config.toml"
    toml.write_text(config(a_knots, f"atlas fit w{week} s={s}"))
    cmd = [str(ROOT / args.bin), "run", "--config", str(toml), "--mesh", str(ROOT / args.mesh),
           "--steps", str(STEPS_PER_WEEK), "--print-every", "500",
           "--checkpoint-every", str(STEPS_PER_WEEK), "--out", str(cdir)]
    if resume:
        cmd += ["--resume", str(resume)]
    t0 = time.time()
    res = {"s": s, "ok": False, "max_ninv": 0, "Uk": None, "reason": ""}
    with open(cdir / "run.log", "w") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            m = ROW.match(line)
            if not m:
                continue
            uk, ninv = m.group(3), int(m.group(9))
            res["Uk"] = uk
            res["max_ninv"] = max(res["max_ninv"], ninv)
            # Strict rule: stop at the first inverted tet. Relaxed rule: let it run
            # unless it is clearly diverging (10x the accepted count).
            if "NaN" in line or ninv > 10 * args.max_ninv:
                res["reason"] = "NaN" if "NaN" in line else f"ninv {ninv} at step {m.group(1)}"
                p.kill()
                break
        p.wait()
    res["wall_s"] = round(time.time() - t0, 1)
    if res["reason"]:
        return res
    if p.returncode != 0 or not (cdir / "surface_final.vtk").exists():
        res["reason"] = f"exit code {p.returncode}"
        return res
    x = np.load(cdir / "checkpoint" / "x.npy")
    if not np.isfinite(x).all():
        res["reason"] = "non-finite positions"
        return res
    ninv_end = int((signed_vol(x) * REST_SIGN <= 0).sum())
    res["ninv_end"] = ninv_end
    if ninv_end > args.max_ninv:
        res["reason"] = f"ninv {ninv_end} at segment end"
        return res
    if res["Uk"] is None or not float(res["Uk"]) <= args.max_uk:
        res["reason"] = f"Uk {res['Uk']} above {args.max_uk:g}"
        return res
    fi, A, V = folding_index(cdir / "surface_final.vtk")
    res.update(ok=True, fi=round(fi, 4), area=round(A, 4), volume=round(V, 4))
    return res


def log(msg):
    print(msg, flush=True)
    with open(out / "fit.log", "a") as f:
        f.write(msg + "\n")


start_week = 21 + len(state["weeks"])
for week in range(start_week, args.last_week):
    target_week = week + 1
    target = ATLAS_FI[target_week]
    resume = None if week == 21 else out / f"w{week}" / "checkpoint"
    wdir = out / f"w{target_week}"
    t_week = time.time()
    results, best = [], None
    for s in args.candidates:
        a_knots = state["a"] + [state["a"][-1] + s / 15]
        r = run_candidate(target_week, s, a_knots, resume, wdir / f"cand_s{s}")
        r["err"] = abs(r["fi"] - target) if r["ok"] else None
        results.append(r)
        log(f"  w{week}->w{target_week} s={s}: " + (f"FI {r['fi']:.3f} (atlas {target:.2f}), Uk {r['Uk']}, "
            f"ninv end {r['ninv_end']} (max printed {r['max_ninv']})"
            if r["ok"] else f"discarded ({r['reason']})") + f", {r['wall_s']} s")
        if r["ok"] and (best is None or r["err"] < best["err"] - args.tie):
            best = r
        ok = [q for q in results if q["ok"]]
        # FI increases with s: stop once it overshoots and the error grows again.
        if r["ok"] and r["fi"] > target and len(ok) >= 2 and r["err"] > ok[-2]["err"]:
            break
        if not r["ok"]:
            break  # a larger s compresses more; it would blow up too
    if best is None:
        log(f"week {week}->{target_week}: every candidate discarded, stopping")
        break
    # Keep the chosen candidate as wNN/{surface_final.vtk, checkpoint, config.toml, run.log}.
    chosen = wdir / f"cand_s{best['s']}"
    for f in ["surface_final.vtk", "config.toml", "run.log", "params.toml"]:
        shutil.copy2(chosen / f, wdir / f)
    if (wdir / "checkpoint").exists():
        shutil.rmtree(wdir / "checkpoint")
    shutil.copytree(chosen / "checkpoint", wdir / "checkpoint")
    for c in wdir.glob("cand_s*"):
        if (c / "checkpoint").exists():
            shutil.rmtree(c / "checkpoint")
    state["a"].append(state["a"][-1] + best["s"] / 15)
    state["weeks"].append({"week": target_week, "s": best["s"], "a": round(state["a"][-1], 6),
                           "fi": best["fi"], "atlas_fi": target, "Uk": best["Uk"],
                           "max_ninv": best["max_ninv"], "ninv_end": best.get("ninv_end", 0),
                           "rule": {"k_inverted": args.k_inverted, "max_ninv": args.max_ninv,
                                    "max_uk": args.max_uk if args.max_uk != float("inf") else None},
                           "wall_s": round(time.time() - t_week, 1),
                           "candidates": results})
    state_file.write_text(json.dumps(state, indent=1))
    # Refresh the atlas vs simulation figure and the parameter plot (best effort).
    for tool in ["tools/atlas_grid.py", "tools/fit_params_plot.py"]:
        subprocess.run([sys.executable, str(ROOT / tool), str(out)], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"week {target_week}: s = {best['s']}, a_t = {state['a'][-1]:.4f}, FI {best['fi']:.3f} vs atlas "
        f"{target:.2f}, Uk {best['Uk']}, ninv end {best.get('ninv_end', 0)}, {time.time() - t_week:.0f} s")

log("table: week, s, a_t, FI model, FI atlas, Uk, ninv end, k_inverted")
for w in state["weeks"]:
    k = (w.get("rule") or {}).get("k_inverted")
    log(f"  {w['week']}  {w['s']:<5} {w['a']:.4f}  {w['fi']:.3f}  {w['atlas_fi']:.2f}  {w['Uk']}  "
        f"{w.get('ninv_end', 0)}  {0.0 if k is None else k}")
