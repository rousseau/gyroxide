"""Synthetic test of fold losses (step 3 of the plan, protocol of reports/Pertes pour surfaces
corticales plissées.md): target patch z = h sin(2 pi x / lambda) on a square of side 4 lambda;
candidates in phase, shifted by lambda/4 and lambda/2, flat, half amplitude, doubled wavelength.
A loss passes if L(in phase) < L(shift lambda/2) < L(flat) and L(in phase) < L(half) < L(flat)
for every (lambda, h). Losses: vertex L2 and point-to-plane (fixed correspondence), unoriented
varifold (Gaussian x cos^2 kernel on triangles, area weighted) at sigma = lambda/4, lambda/2,
lambda, and the band power of z' curvature (phase-insensitive, H approximated by the Laplacian of
the height on the grid) at octaves around lambda.

    python3 tools/loss_synthetic.py
"""
import numpy as np

N = 16          # grid points per wavelength (patch of 4 lambda -> 4N cells per side)


def patch(lam, h, kind):
    n = 4 * N
    u = np.linspace(0, 4 * lam, n + 1)
    x, y = np.meshgrid(u, u, indexing="ij")
    z = {
        "in phase": h * np.sin(2 * np.pi * x / lam),
        "shift l/4": h * np.sin(2 * np.pi * (x + lam / 4) / lam),
        "shift l/2": h * np.sin(2 * np.pi * (x + lam / 2) / lam),
        "flat": 0 * x,
        "half": 0.5 * h * np.sin(2 * np.pi * x / lam),
        "2 lambda": h * np.sin(np.pi * x / lam),
    }[kind]
    p = np.stack([x, y, z], -1).reshape(-1, 3)
    i = np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]
    i = i.ravel()
    f = np.concatenate([np.stack([i, i + n + 1, i + 1], 1), np.stack([i + 1, i + n + 1, i + n + 2], 1)])
    return p, f, z


def tri(p, f):
    c = p[f].mean(1)
    nrm = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    a = 0.5 * np.linalg.norm(nrm, axis=1)
    return c, nrm / (2 * a[:, None]), a


def vkernel(s, t, sig, stride=4):
    """<S, T> of unoriented varifolds, Gaussian(sigma) x cos^2, one triangle in two."""
    (cs, ns, as_), (ct, nt, at) = s, t
    acc = 0.0
    for k in range(0, len(cs), 2000):
        d2 = ((cs[k:k + 2000, None, :] - ct[None, :, :]) ** 2).sum(-1)
        cos2 = (ns[k:k + 2000] @ nt.T) ** 2
        acc += (as_[k:k + 2000, None] * at[None, :] * np.exp(-d2 / sig ** 2) * cos2).sum()
    return acc


def varifold(p, f, q, g, sig):
    s, t = tri(p, f), tri(q, g)
    sel = lambda v: tuple(a[::2] * 2 if a.ndim == 1 else a[::2] for a in v)
    s, t = sel(s), sel(t)
    return vkernel(s, s, sig) - 2 * vkernel(s, t, sig) + vkernel(t, t, sig)


def band_power(z, lam, octave):
    """Power of the Laplacian of z in the octave [lam * 2^(octave-1/2), lam * 2^(octave+1/2)] of
    wavelengths (FFT along x and y of the periodic patch)."""
    n = z.shape[0] - 1
    zz = z[:n, :n]
    k = np.fft.fftfreq(n, d=4 * lam / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kr = np.hypot(kx, ky)
    lap = np.abs(np.fft.fft2(zz) * (2 * np.pi * kr) ** 2) ** 2
    wl = np.where(kr > 0, 1 / np.maximum(kr, 1e-12), np.inf)
    lo, hi = lam * 2 ** (octave - 0.5), lam * 2 ** (octave + 0.5)
    return lap[(wl >= lo) & (wl < hi)].sum() / n ** 4


def main():
    kinds = ["in phase", "shift l/4", "shift l/2", "flat", "half", "2 lambda"]
    verdict = {}
    for lam in (5, 10, 15):
        for h in (1, 2, 3):
            q, g, zt = patch(lam, h, "in phase")
            rows = {}
            for kind in kinds:
                p, f, z = patch(lam, h, kind)
                r = {"vertex L2": np.mean(np.sum((p - q) ** 2, 1)),
                     "point-plane": np.mean(((p - q) @ np.array([0, 0, 1.0])) ** 2)}
                for s in (0.25, 0.5, 1.0):
                    r[f"varifold s={s}l"] = varifold(p, f, q, g, s * lam)
                pb = sum((np.log(band_power(z, lam, o) + 1e-9) - np.log(band_power(zt, lam, o) + 1e-9)) ** 2 for o in (-1, 0, 1))
                r["band power (log)"] = pb
                rows[kind] = r
            print(f"lambda {lam} mm, h {h} mm: " + " | ".join(
                f"{name}: " + " ".join(f"{rows[k][name] / max(rows['flat'][name], 1e-30):.2f}" for k in kinds)
                for name in rows["flat"]))
            for name in rows["flat"]:
                L = {k: rows[k][name] for k in kinds}
                ok = L["in phase"] < L["shift l/2"] < L["flat"] and L["in phase"] < L["half"] < L["flat"]
                verdict.setdefault(name, []).append(ok)
    print("\nratios to flat in the order:", ", ".join(kinds))
    for name, oks in verdict.items():
        print(f"{name:20s} passes {sum(oks)}/{len(oks)}")


if __name__ == "__main__":
    main()
