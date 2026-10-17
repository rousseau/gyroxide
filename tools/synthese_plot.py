#!/usr/bin/env python3
"""Unified synthesis figure from tools/scan_metrics.py output: FI gap to the atlas and gain in fold
placement (curvature correlation minus that of the SG 21 geometry), per model and week.

    python3 tools/synthese_plot.py out/synthese/scan_metrics.csv out/synthese/synthese_metrics.png [--absolute]

--absolute: the values themselves, with the atlas FI and the SG 21 geometry as reference curves.
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(sys.argv[1])))
absolute = "--absolute" in sys.argv
runs = [r for r in dict.fromkeys(r["run"] for r in rows) if not r.startswith("atlas")]
atlas = {int(r["week"]): float(r["fi"]) for r in rows if r["run"] == "atlas"}
base = {int(r["week"]): float(r["r_curv"]) for r in rows if r["run"] == "atlas SG 21"}
COLORS = ["#898781", "#e87ba4", "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
INK, INK2, GRID = "#14213d", "#5a6573", "#e4e3df"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), dpi=120)
for i, run in enumerate(runs):
    pts = sorted((int(r["week"]), float(r["fi"]), float(r["r_curv"])) for r in rows if r["run"] == run)
    w = [p[0] for p in pts]
    col = COLORS[i % len(COLORS)]
    ra = (lambda k: 0.0) if absolute else (lambda k: atlas[k])
    rb = (lambda k: 0.0) if absolute else (lambda k: base[k])
    axs[0].plot(w, [p[1] - ra(p[0]) for p in pts], "-o", color=col, lw=2.4, ms=5, label=run)
    axs[1].plot(w, [p[2] - rb(p[0]) for p in pts], "-o", color=col, lw=2.4, ms=5, label=run)
    for ax, y in ((axs[0], pts[-1][1] - ra(w[-1])), (axs[1], pts[-1][2] - rb(w[-1]))):
        if w[-1] < 36:
            ax.plot(w[-1], y, "x", color=col, ms=12, mew=2.5)
if absolute:
    ws = sorted(atlas)
    axs[0].plot(ws, [atlas[k] for k in ws], "--o", color=INK, lw=3, ms=5, label="atlas dHCP (référence)", zorder=5)
    ws = sorted(base)
    axs[1].plot(ws, [base[k] for k in ws], "--o", color="#9aa3b2", lw=3, ms=5,
                label="géométrie de SG 21 seule (référence)", zorder=5)
for ax in axs:
    if not absolute:
        ax.axhline(0, color=INK, lw=1.2)
    ax.set_xlabel("semaine de gestation (SG)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_xlim(21.5, 36.5)
if absolute:
    axs[0].set_ylim(6.0, 9.8)
    axs[0].set_title("(a) Indice de plissement FI = A / V^(2/3)", loc="left", color=INK, fontsize=14)
    axs[1].set_ylim(0.0, 1.02)
    axs[1].set_title("(b) Placement des plis : corrélation de la courbure\nsimulée avec celle de l'atlas, sommet par sommet", loc="left", color=INK, fontsize=14)
    axs[0].legend(frameon=False, fontsize=11, loc="upper left")
    axs[1].legend(frameon=False, fontsize=11, loc="lower left", handles=axs[1].get_lines()[-1:])
else:
    axs[0].set_ylim(-0.6, 0.8)
    axs[0].set_title("(a) Écart de l'indice de plissement à l'atlas (FI simulé − FI atlas)", loc="left", color=INK, fontsize=14)
    axs[1].set_ylim(-0.15, 0.08)
    axs[1].set_title("(b) Placement des plis : gain sur la géométrie de SG 21\n(corrélation de courbure avec l'atlas, modèle − géométrie seule)", loc="left", color=INK, fontsize=14)
    axs[0].legend(frameon=False, fontsize=11, loc="upper left")
fig.text(0.01, 0.01, "× : arrêt numérique de la simulation. " + ("En (b), 1 = mêmes crêtes et sillons aux mêmes endroits que l'atlas ; référence : garder la forme de SG 21 sans simuler." if absolute else "Au-dessus de 0 en (b) : la mécanique place les plis mieux que la forme de SG 21 seule."), fontsize=11, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(sys.argv[2], facecolor="white")
print(sys.argv[2])
