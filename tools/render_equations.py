#!/usr/bin/env python3
"""LaTeX-like renders (matplotlib mathtext, Computer Modern) of the deck's equations and tables.

    python3 tools/render_equations.py out/deck
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/deck")
out.mkdir(parents=True, exist_ok=True)
INK, INK2 = "#14213D", "#3D4A63"
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.serif": ["STIXGeneral"]})


def lines(name, rows, width, height, size=30):
    """rows: list of (text, size factor, colour); one per line, left aligned."""
    fig = plt.figure(figsize=(width, height), dpi=200)
    y = 1.0
    step = 1.0 / len(rows)
    for text, k, col in rows:
        fig.text(0.0, y - step / 2, text, fontsize=size * k, color=col, va="center", ha="left")
        y -= step
    fig.savefig(out / f"{name}.png", transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(out / f"{name}.png")


lines("eq_model", [
    (r"$\mathbf{G} = \mathbf{I} + (\mathbf{I} - \mathbf{n}\,\mathbf{n}^{\top})\; g_m\, a_t,"
     r"\qquad g_m = g_r\,/\,\left(1 + e^{\,10\,(d/H - 1)}\right)$", 1.0, INK),
    (r"$\boldsymbol{\sigma} = \mu\, J^{-5/3}\, \mathrm{dev}(\mathbf{B}) + K\,(J - 1)\,\mathbf{I},"
     r"\qquad \mu = \mu_w\,(1 - g_m) + \mu_g\, g_m,\qquad \mathbf{F}_e = \mathbf{F}\,\mathbf{G}^{-1}$", 1.0, INK),
], 15, 1.9)

lines("eq_fi", [
    (r"$\mathrm{FI} = A\,/\,V^{2/3},\qquad t = (\mathrm{SG} - 21)\,/\,15$", 1.0, INK),
], 8, 0.8)

lines("eq_global", [
    (r"$s_k^{\star} = \underset{s \in \mathcal{S}}{\arg\min}\;"
     r"\left|\,\mathrm{FI}_{\mathrm{sim}}(t_{k+1};\, s) - \mathrm{FI}_{\mathrm{atlas}}(\mathrm{SG}_{k+1})\,\right|$", 1.0, INK),
    (r"$a(t_{k+1}) = a(t_k) + s_k^{\star}/15,\qquad \mathcal{S} = \{0,\ 0.25,\ 0.5,\ 1,\ 1.85\}\ \ (\to 5\ \mathrm{d\grave{e}s\ SG\ 30})$", 0.8, INK2),
], 9.5, 1.8)

lines("eq_local", [
    (r"$a_i(t) = \sqrt{\tilde{A}_i(t)\,/\,\tilde{A}_i(t_{21})} - 1,\qquad H_i(t) = \|\,\mathbf{p}_i(t) - \mathbf{w}_i(t)\,\|$", 1.0, INK),
    (r"$\tilde{A}_i$ : aire du sommet $i$ lissée ($\approx$ 4–5 mm) ;  $\mathbf{p}_i, \mathbf{w}_i$ : sommets pial et blanc", 0.72, INK2),
], 9.5, 1.6)

# Parameter table of the model slide.
rows = [
    (r"$a_t$", "expansion tangentielle relative du cortex : aire × $(1 + a_t)^2$", r"$1.85\,t$", "mesuré sur l'atlas"),
    (r"$H$", "épaisseur de la couche corticale qui croît", r"$0.084 - 0.042\,t$", "mesuré (piale–blanche)"),
    (r"$\mu_g,\ \mu_w$", "modules de cisaillement : cortex (gray), subst. blanche (white)", r"$1,\ 1.167$", r"libre : $\mu_g/\mu_w$"),
    (r"$K$", "module de compressibilité (quasi-incompressible)", r"$5$", "fixé"),
    (r"$g_r$", "masque de croissance (0 : paroi médiale, 1 : cortex)", "ellipsoïde", "atlas"),
    (r"$\rho,\ \gamma,\ \Delta t$", "densité, amortissement, pas de temps", r"$0.0025;\ 0.5;\ 5.6\cdot10^{-6}$", "numérique"),
    (r"$h_c,\ k_c$", "épaisseur et raideur du contact pénalisé entre surfaces", r"$0.2\,a;\ 10\,K$", "numérique"),
]
fig = plt.figure(figsize=(16, 5.6), dpi=200)
cols = [0.0, 0.09, 0.58, 0.81]
heads = ["Symbole", "Signification", "Valeur (Tallinen)", "Statut ici"]
for x, h in zip(cols, heads):
    fig.text(x, 0.96, h, fontsize=22, color=INK, weight="bold", va="center")
fig.add_artist(plt.Line2D([0, 1], [0.91, 0.91], color=INK2, lw=1.2, transform=fig.transFigure))
for r, row in enumerate(rows):
    y = 0.84 - r * 0.125
    for x, cell, k in zip(cols, row, [1.15, 0.92, 1.0, 0.92]):
        bold = r == 2 and x == cols[3]
        fig.text(x, y, cell, fontsize=22 * k, color=INK, va="center", weight="bold" if bold else "normal")
    fig.add_artist(plt.Line2D([0, 1], [y - 0.0625, y - 0.0625], color="#E3E1D8", lw=0.8, transform=fig.transFigure))
fig.savefig(out / "table_params.png", transparent=True, bbox_inches="tight", pad_inches=0.05)
print(out / "table_params.png")
