"""ISBI 2027 table: per week and method (rows of tools/paper_figure.py), rigid deviation from the atlas
on the cortex (RMS and 95th percentile, mm; also for the atlas of the previous week, a persistence
baseline), correlation of the mean curvature, fold depth (5-95
percentile range of the sulcal depth, mm) and folding index A / V^(2/3) relative to the atlas.

    python3 tools/paper_table.py ~/Exp/isbi_2027_gyroxide/table_metrics.tex [weeks ...]

Also prints the values and writes them as JSON next to the .tex file.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_gallery import Surf, load_full, metrics, rigid_to  # noqa: E402
from paper_figure import ROWS, atlas_xyz, fitted  # noqa: E402
from surface_metrics import folding_index  # noqa: E402


def main():
    out = Path(sys.argv[1])
    weeks = [int(w) for w in sys.argv[2:]] or [26, 28, 30, 33, 36]
    ns, mm = load_full()
    faces = np.load("data/full/prep/w30/faces.npy").astype(int)
    m = np.load("data/full/prep/w30/cortex.npy")[:ns] > 0.5
    res, persist = {}, {}
    for w in weeks:
        y = atlas_xyz(w, ns)
        S = Surf(faces, y, mm)
        fi_atlas = folding_index(y, S.f)
        # persistence baseline: the atlas of the previous week, rigidly aligned (no model)
        e = np.linalg.norm(rigid_to(atlas_xyz(w - 1, ns), y, m) - y, axis=1)[m] * mm
        persist[w] = {"rms": float(np.sqrt((e ** 2).mean())), "p95": float(np.percentile(e, 95))}
        for name, _, fn in ROWS[1:]:
            x = fn(w, ns)
            if x is None or not np.isfinite(x).all():
                continue
            x = rigid_to(x, y, m)
            mt = metrics(S, x, y, m)
            res.setdefault(name, {})[w] = {
                "rms": float(mt["mm"]), "p95": float(np.percentile(mt["err"][m], 95)), "corr": float(mt["corr"]),
                "depth": float(mt["depth"]), "depth_atlas": float(mt["depth_atlas"]),
                "fi": float(folding_index(x, S.f) / fi_atlas)}
            if name.startswith("Estimated"):
                res[name][w]["run"] = str(fitted(w))
            print(w, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name][w].items()})
    out.with_suffix(".json").write_text(json.dumps({**res, "persistence": persist}, indent=1))

    def cell(name, w, k, nd=2, best=False):
        v = res.get(name, {}).get(w, {}).get(k)
        if v is None:
            return "--"
        s = f"{v:.{nd}f}"
        return f"\\textbf{{{s}}}" if best else s

    lines = [r"\begin{tabular}{@{}ll" + "c" * len(weeks) + "@{}}", r"\toprule",
             " & GA (weeks) & " + " & ".join(str(w) for w in weeks) + r" \\", r"\midrule"]
    crit = [("rms", "RMS dev. (mm) $\\downarrow$", min),
            ("corr", "curvature corr. $\\uparrow$", max), ("depth", "fold depth (mm)", None),
            ("fi", "folding index ratio", None)]
    names = [n for n, _, _ in ROWS[1:]]
    for k, lab, better in crit:
        for i, name in enumerate(names):
            cells = []
            for w in weeks:
                vals = {n: res.get(n, {}).get(w, {}).get(k) for n in names}
                vals = {n: v for n, v in vals.items() if v is not None}
                b = better is not None and vals and better(vals, key=vals.get) == name and len(vals) > 1
                cells.append(cell(name, w, k, 2, b))
            lines.append((lab if i == 0 else "") + f" & {name.split()[0].lower()} & " + " & ".join(cells) + r" \\")
        if k in persist[weeks[0]]:
            lines.append(" & atlas $w{-}1$ & " + " & ".join(f"{persist[w][k]:.2f}" for w in weeks) + r" \\")
        if k == "depth":
            lines.append(" & atlas & " + " & ".join(
                f"{next(iter(r[w]['depth_atlas'] for r in res.values() if w in r)):.2f}" for w in weeks) + r" \\")
        lines.append(r"\midrule" if k != crit[-1][0] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    out.write_text("\n".join(lines) + "\n")
    print("saved", out)


if __name__ == "__main__":
    main()
