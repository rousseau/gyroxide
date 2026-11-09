#!/usr/bin/env python3
"""Agrège des scorecard.json (tools/scorecard.py, `gyroxide eval`) en tableau Markdown, CSV et figure.

    python3 tools/scorecard_table.py <scorecard.json | dossier>... --out out/tableau [--no-figure]

Les dossiers sont parcourus récursivement (tout *.json qui est un scorecard). Une ligne par `label`
(--label à l'évaluation, sinon le nom du dossier du run) et par semaine. La ligne « persistence » (atlas w-1
aligné rigidement contre l'atlas w, `gyroxide eval --persistence`) est la référence de chaque bloc : elle est
tracée en pointillé.

Écrit <out>.md (un bloc par métrique, semaines en colonnes), <out>.csv (une ligne par label et semaine, toutes
les grandeurs de scorecard.flatten) et <out>.png (une courbe par métrique, semaines en abscisse).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard import flatten  # noqa: E402

PERSISTENCE = "persistence"
# (colonne de scorecard.flatten, titre, décimales, sens : « bas » = plus petit vaut mieux, « un » = viser 1, « haut »)
METRICS = [
    ("dev_s200_rms_mm", "écart rigide S²⁰⁰, RMS (mm)", 2, "bas"),
    ("dev_s50_rms_mm", "écart rigide S⁵⁰, RMS (mm)", 2, "bas"),
    ("dev_vertex_rms_mm", "écart rigide aux sommets, RMS (mm)", 2, "bas"),
    ("curvature_corr", "corrélation de courbure moyenne", 3, "haut"),
    ("fold_depth_mm.ratio", "profondeur des plis / atlas", 3, "un"),
    ("area_mm2.ratio", "aire du cortex / atlas", 3, "un"),
    ("gi.ratio", "IG (A / enveloppe convexe) / atlas", 3, "un"),
    ("thickness_mm.ratio", "épaisseur / atlas", 3, "un"),
    ("band_1_2.ratio", "puissance de courbure 1-2 mm / atlas", 2, "un"),
    ("band_2_4.ratio", "puissance de courbure 2-4 mm / atlas", 2, "un"),
    ("band_4_8.ratio", "puissance de courbure 4-8 mm / atlas", 2, "un"),
    ("band_8_16.ratio", "puissance de courbure 8-16 mm / atlas", 2, "un"),
    ("n_inverted", "éléments retournés", 0, "bas"),
]
# palette catégorielle de référence (skill dataviz), dans l'ordre fixe des séries ; texte en encre, pas en couleur de série
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e1"


def collect(paths):
    """{label: {semaine: colonnes}} depuis des fichiers ou des dossiers de scorecard.json."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.json")) if p.is_dir() else [p]
    rows = {}
    for f in files:
        card = json.loads(f.read_text())
        if not (isinstance(card, dict) and "schema" in card and "metrics" in card):
            continue                                    # autre JSON du dossier (deviation_mm.json, fit.json...)
        rows.setdefault(card["label"], {})[card["week"]] = flatten(card["metrics"])
    return rows


def order(rows):
    return sorted((k for k in rows if k != PERSISTENCE), key=str) + ([PERSISTENCE] if PERSISTENCE in rows else [])


def markdown(rows, weeks):
    out = []
    for col, title, nd, _ in METRICS:
        if not any(r.get(w, {}).get(col) is not None for r in rows.values() for w in weeks):
            continue
        out += [f"**{title}**", "", "| | " + " | ".join(f"SG {w}" for w in weeks) + " |",
                "|---|" + "---:|" * len(weeks)]
        for label in order(rows):
            cells = []
            for w in weeks:
                v = rows[label].get(w, {}).get(col)
                cells.append("—" if v is None else f"{v:.{nd}f}")
            out.append(f"| {'atlas w−1 (persistance)' if label == PERSISTENCE else label} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def write_csv(rows, dest):
    cols = sorted({c for r in rows.values() for f in r.values() for c in f})
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "week"] + cols)
        for label in order(rows):
            for wk in sorted(rows[label]):
                w.writerow([label, wk] + ["" if rows[label][wk].get(c) is None else repr(rows[label][wk][c])
                                          for c in cols])


def figure(rows, weeks, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [m for m in METRICS if any(r.get(w, {}).get(m[0]) is not None for r in rows.values() for w in weeks)]
    ncol = 4 if len(cols) > 8 else 3
    nrow = -(-len(cols) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow + 0.9), dpi=130, facecolor=SURFACE,
                             squeeze=False)
    runs = [k for k in order(rows) if k != PERSISTENCE]
    for ax, (col, title, _, sens) in zip(axes.ravel(), cols):
        ax.set_facecolor(SURFACE)
        if sens == "un":
            ax.axhline(1.0, color=MUTED, lw=0.8, zorder=0)
        for i, label in enumerate(order(rows)):
            ws = [w for w in sorted(rows[label]) if rows[label][w].get(col) is not None]
            if not ws:
                continue
            ys = [rows[label][w][col] for w in ws]
            if label == PERSISTENCE:
                ax.plot(ws, ys, color=MUTED, lw=1.6, ls=(0, (4, 3)), label="atlas w−1 (persistance)")
            else:
                ax.plot(ws, ys, color=SERIES[runs.index(label) % len(SERIES)], lw=2, marker="o", ms=4,
                        mec=SURFACE, mew=1, label=label)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xticks(weeks if len(weeks) <= 8 else weeks[::2])
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("semaine de gestation", fontsize=8, color=INK2)
    h, lab = axes[0][0].get_legend_handles_labels()
    for ax in axes.ravel():
        if len(ax.get_legend_handles_labels()[0]) > len(h):
            h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 4), frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(dest, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="scorecard.json ou dossiers qui en contiennent")
    ap.add_argument("--out", required=True, help="préfixe des sorties (.md, .csv, .png)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    rows = collect(args.paths)
    if not rows:
        sys.exit("aucun scorecard.json trouvé")
    weeks = sorted({w for r in rows.values() for w in r})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.md").write_text(markdown(rows, weeks) + "\n")
    write_csv(rows, Path(f"{out}.csv"))
    if not args.no_figure:
        figure(rows, weeks, Path(f"{out}.png"))
    print(f"{sum(len(r) for r in rows.values())} scorecards, {len(rows)} lignes → {out}.md / .csv" +
          ("" if args.no_figure else " / .png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
