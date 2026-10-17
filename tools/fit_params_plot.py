#!/usr/bin/env python3
"""Fitted simulation parameters vs gestational week, from <fit>/fit.json.

    python3 tools/fit_params_plot.py out/atlas_fit [out/atlas_fit/params.png]

Panels: a_t (with Tallinen's 1.85 t), weekly slope s, cortical thickness H (atlas
table), folding index model vs atlas, its relative error, inverted tets and kinetic
energy at the end of each week. Weeks accepted under the relaxed rule
(k_inverted > 0, a few inverted tets allowed) are drawn with hollow markers on a
shaded band.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
fit = Path(sys.argv[1])
png = sys.argv[2] if len(sys.argv) > 2 else str(fit / "params.png")
state = json.loads((fit / "fit.json").read_text())
W = state["weeks"]

# Atlas thickness table (mm) from the comment line of the tabulated config.
cfg = (ROOT / "configs/atlas_w21_tabulated.toml").read_text()
h_mm = [float(x) for x in re.search(r"(?m)^# (1\.\d\d(?:, \d\.\d\d)+)$", cfg).group(1).split(", ")]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL, REF = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

sg = np.array([21] + [w["week"] for w in W])
a = np.array([0.0] + [w["a"] for w in W])
relaxed = np.array([False] + [bool((w.get("rule") or {}).get("k_inverted")) for w in W])
fi = np.array([6.176] + [w["fi"] for w in W])
fi_atlas = np.array([6.18] + [w["atlas_fi"] for w in W])
uk = np.array([np.nan] + [float(w["Uk"]) for w in W])
ninv = np.array([0] + [w.get("ninv_end", 0) for w in W])
first_relaxed = sg[relaxed].min() if relaxed.any() else None


def frame(ax, title, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_xlim(20.5, 36.5)
    ax.set_xticks(range(21, 37, 2))
    if first_relaxed is not None:
        ax.axvspan(first_relaxed - 0.5, sg.max() + 0.5, color="#f1efe8", zorder=0)


def points(ax, x, y, color, rel=None, **kw):
    """Line through all weeks; filled markers = strict rule, hollow = relaxed rule."""
    rel = relaxed[-len(x):] if rel is None else rel
    ax.plot(x, y, color=color, lw=2, **kw)
    ax.plot(x[~rel], y[~rel], "o", ms=5, color=color)
    ax.plot(x[rel], y[rel], "o", ms=6, mfc="white", mec=color, mew=1.6)


fig, axs = plt.subplots(2, 4, figsize=(17, 7.6), dpi=110)
axs = axs.ravel()

ax = axs[0]
frame(ax, "(a) Expansion tangentielle a_t", "a_t")
tt = np.linspace(21, 36, 100)
ax.plot(tt, 1.85 * (tt - 21) / 15, color=REF, lw=1.5, ls="--", label="Tallinen 1,85·t")
points(ax, sg, a, MODEL, label="ajusté")
ax.legend(frameon=False)

ax = axs[1]
frame(ax, "(b) Pente choisie s par semaine", "s = 15·Δa_t")
ax.step(sg[1:], [w["s"] for w in W], where="pre", color=MODEL, lw=2)
s = np.array([np.nan] + [w["s"] for w in W])
ax.plot(sg[1:][~relaxed[1:]], s[1:][~relaxed[1:]], "o", ms=5, color=MODEL)
ax.plot(sg[1:][relaxed[1:]], s[1:][relaxed[1:]], "o", ms=6, mfc="white", mec=MODEL, mew=1.6)
ax.axhline(1.85, color=REF, lw=1.2, ls="--")
ax.text(21.2, 1.9, "Tallinen", color=INK2, fontsize=8)

ax = axs[2]
frame(ax, "(c) Épaisseur corticale H (atlas, imposée)", "H (mm)")
ax.plot(range(21, 37), h_mm, color=INK2, lw=1.2, ls=":")
ax.plot(sg, np.array(h_mm)[sg - 21], "o-", color=MODEL, lw=2, ms=5)

ax = axs[3]
frame(ax, "(d) Indice de plissement A/V^(2/3)", "FI")
ax.plot(range(21, 37), [6.18, 6.18, 6.20, 6.23, 6.29, 6.40, 6.60, 6.83, 7.09, 7.43, 7.82, 8.10, 8.34, 8.59, 8.78, 8.92],
        color=REF, lw=1.5, ls="--", label="atlas")
points(ax, sg, fi, MODEL, label="simulation")
ax.legend(frameon=False)

ax = axs[4]
frame(ax, "(e) Erreur relative sur FI", "(FI sim − FI atlas) / FI atlas (%)")
ax.axhline(0, color=INK2, lw=0.8)
points(ax, sg, 100 * (fi - fi_atlas) / fi_atlas, MODEL)

ax = axs[5]
frame(ax, "(f) Tétraèdres inversés en fin de semaine", "n inversés")
points(ax, sg, ninv.astype(float), MODEL)
ax.set_ylim(bottom=0, top=max(2, ninv.max() * 1.2))
ax.yaxis.get_major_locator().set_params(integer=True)

ax = axs[6]
frame(ax, "(g) Énergie cinétique Uk en fin de semaine", "Uk (log)")
points(ax, sg[1:], uk[1:], MODEL)
ax.set_yscale("log")

ax = axs[7]
ax.axis("off")
last = W[-1]["week"] if W else 21
ax.text(0, 1, "\n".join([
    f"Ajustement glouton semaine par semaine, SG 21–{last}.",
    "Chaque semaine : a_t prolongé continûment avec une pente s,",
    "le candidat dont le FI est le plus proche de l'atlas est gardé.",
    "",
    "● plein : règle stricte (aucun tétraèdre inversé, k_inverted = 0)",
    "○ creux, fond grisé : règle assouplie (k_inverted = 0,5,",
    "   ≤ 20 inversés, Uk ≤ 1e-3 en fin de semaine)",
    "",
    "Pas de temps 7,6e-6 (step_factor 0,025), contact vérifié à chaque pas.",
    "Maillage : atlas dHCP SG 21, hémisphère gauche, 610k tétraèdres.",
]), va="top", fontsize=9, color=INK2)
for ax in axs[:7]:
    ax.set_xlabel("semaine de gestation (SG)")
fig.tight_layout()
fig.savefig(png, facecolor="white")
Path(png).with_name(f"params_w{last}.png").write_bytes(Path(png).read_bytes())
print(png)
