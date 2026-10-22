"""Quasi-static chain of weekly equilibria with a remodelling core.

Week by week from SG 21, the equilibrium of week w is computed from that of week w - 1 with the
measured cortical growth, thickening and core growth of week w (adjoint_week, no fit). The core
growth is the scheduled tensor U(w) corrected by the accumulated remodelling D (per tetrahedron):

    U_eff(w) = exp(log U(w) + D),   after the equilibrium x_w:   D += alpha (log S(x_w) - log U_eff(w))

with S the right stretch of the total deformation: alpha = 0 is Tallinen's elastic core, alpha = 1
leaves the core stress free at the end of every week (relaxation time << 1 week); a Maxwell-like
relaxation time tau gives alpha = 1 - exp(-1 week / tau).

    python3 tools/remodel_chain.py out/remodel_a1 --alpha 1 [--w1 28]
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sym_fun(m, f):
    w, v = np.linalg.eigh(m)
    return np.einsum("tij,tj,tkj->tik", v, f(w), v)


def to6(m):
    return np.ascontiguousarray(np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], m[:, 0, 1], m[:, 0, 2], m[:, 1, 2]], 1))


def from6(c):
    m = np.empty((len(c), 3, 3))
    for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
        m[:, i, j] = m[:, j, i] = c[:, k]
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--w1", type=int, default=28)
    ap.add_argument("--config", default="configs/mu_test/mu2_s.toml")
    ap.add_argument("--mesh", default="data/meshes/w21_left_610k_s.mesh")
    ap.add_argument("--prep", default="data/adjoint_prep_s")
    ap.add_argument("--bin", default="target/mu/adjoint_week_core")
    ap.add_argument("--relax", type=int, default=0, help="damped steps before Newton on release (0: Newton only)")
    ap.add_argument("--newton", type=int, default=40)
    ap.add_argument("--conv", type=float, default=1e-2, help="converged if final |F| <= conv x initial |F|")
    ap.add_argument("--fallback-relax", type=int, default=3000)
    ap.add_argument("--ftol", type=float, default=1e-4, help="or converged if final |F| <= ftol")
    ap.add_argument("--resume", type=int, default=0, help="first week to compute; earlier weeks are read back from <out>")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tok = (ROOT / args.mesh).read_text().split()
    n = int(tok[0])
    X = np.array(tok[1:1 + 3 * n], dtype=float).reshape(n, 3)
    nt = int(tok[1 + 3 * n])
    T = np.array(tok[2 + 3 * n:2 + 3 * n + 5 * nt], dtype=np.int64).reshape(nt, 5)[:, 1:] - 1
    edges = lambda p: np.stack([p[T[:, k]] - p[T[:, 0]] for k in (1, 2, 3)], axis=2)
    inv_ref = np.linalg.inv(edges(X))
    core = np.load(ROOT / args.mesh.replace(".mesh", ".nodal_core.npy"), mmap_mode="r")

    D = np.zeros((nt, 3, 3))
    x = X.copy()
    log = json.loads((out / "chain.json").read_text())["weeks"] if args.resume else []
    log = [e for e in log if e["sg"] < args.resume]
    for w in range(22, args.resume):
        # Rebuild the accumulated remodelling from the saved equilibria and effective tensors.
        x = np.load(out / f"w{w}" / "eq" / "x_released.npy")
        Ueff = from6(np.load(out / f"w{w}" / "core_eff.npy"))
        F = edges(x) @ inv_ref
        S = sym_fun(np.einsum("tki,tkj->tij", F, F), np.sqrt)
        D = D + args.alpha * (sym_fun(S, np.log) - sym_fun(Ueff, np.log))
    for w in range(max(22, args.resume), args.w1 + 1):
        knot = w - 21
        U = from6(np.asarray(core[knot])[T].mean(1))
        Ueff = sym_fun(sym_fun(U, np.log) + D, np.exp)
        W = out / f"w{w}"
        ck = W / "start"
        ck.mkdir(parents=True, exist_ok=True)
        np.save(ck / "x.npy", x)
        np.save(ck / "v.npy", np.zeros_like(x))
        np.save(ck / "growth_prev.npy", np.zeros(nt))
        (ck / "state.json").write_text(json.dumps({"step": 0, "t": knot / 15}))
        np.save(W / "core_eff.npy", to6(Ueff))
        def run(start, relax, tag):
            np.save(ck / "x.npy", start)
            cmd = [str(ROOT / args.bin), args.config, args.mesh, str(ck), f"{args.prep}/w{w}", str(W / "eq"),
                   "--rigid", "--multiscale", "200:0,50:0,10:0,0:0", "--cumulative", "--theta0-table",
                   "--relax0", "0", "--newton", str(args.newton), "--min-growth", "0", "--inverse-eq", str(ck / "x.npy"),
                   "--iters", "0", "--release-relax", str(relax), "--core-override", str(W / "core_eff.npy")]
            with open(W / f"run{tag}.log", "w") as f:
                r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            text = (W / f"run{tag}.log").read_text().splitlines()
            line = [l for l in text if l.startswith("released mm k=  0")]
            inv = [l for l in text if l.startswith("inverted elements")]
            res = [l for l in text if l.startswith("released with")]
            ok = r.returncode == 0 and bool(line)
            conv = False
            if ok and res:
                a, b = (float(v) for v in res[0].split("|F|")[1].split("->"))
                conv = b <= args.conv * a or b <= args.ftol
            return ok, conv, line, inv
        # Newton from the previous equilibrium; if it stalls (inverted elements, buckling),
        # damped steps from the partially converged state, then Newton again.
        ok, conv, line, inv = run(x, args.relax, "")
        if ok and not conv:
            xp = np.load(W / "eq" / "x_released.npy")
            if np.isfinite(xp).all():
                print(f"SG {w}: Newton stalled, damped relaxation ({args.fallback_relax} steps) then Newton", flush=True)
                ok, conv, line, inv = run(xp, args.fallback_relax, "_fallback")
        if not ok:
            print(f"SG {w}: failed, see {W}", flush=True)
            break
        x = np.load(W / "eq" / "x_released.npy")
        if not np.isfinite(x).all():
            print(f"SG {w}: non-finite positions", flush=True)
            break
        if not conv:
            print(f"SG {w}: not converged, stopping (see {W})", flush=True)
            break
        rigid = float(line[0].split("rigid rms")[1].split()[0])
        F = edges(x) @ inv_ref
        S = sym_fun(np.einsum("tki,tkj->tij", F, F), np.sqrt)
        elastic = np.median(np.linalg.norm((S - Ueff).reshape(nt, 9), axis=1))
        D = D + args.alpha * (sym_fun(S, np.log) - sym_fun(Ueff, np.log))
        log.append({"sg": w, "rigid_mm": rigid, "core_elastic_median": float(elastic), "inv": inv[0] if inv else ""})
        print(f"SG {w}: rigid {rigid:.3f} mm, core elastic stretch (median |S - U_eff|) {elastic:.3f}, {inv[0] if inv else ''}", flush=True)
        (out / "chain.json").write_text(json.dumps({"alpha": args.alpha, "weeks": log}, indent=1))
        shutil.rmtree(ck)


if __name__ == "__main__":
    main()
