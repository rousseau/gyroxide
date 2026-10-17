#!/usr/bin/env python3
"""Renders surfaces with the camera of Brains.cpp's writePov (Tallinen et al. 2016, Fig. 1d).

    python3 tools/render_paper_view.py out.png t0.vtk t1.vtk ... --times 0 0.2 ... [--zoom folding]

Camera at (-1.5, 1.5, -1.5) * zoom looking at the origin, sky -z; zoom = 2.2 - 1.2 t
("folding" block of Brains.cpp), so the brain grows on screen as in the paper. Pale
yellow, specular, white background.
"""
import argparse

import pyvista as pv

ap = argparse.ArgumentParser()
ap.add_argument("png")
ap.add_argument("vtk", nargs="+")
ap.add_argument("--times", type=float, nargs="+", required=True)
ap.add_argument("--labels", nargs="*")
ap.add_argument("--view-angle", type=float, default=60.0)
args = ap.parse_args()

n = len(args.vtk)
pl = pv.Plotter(shape=(1, n), off_screen=True, window_size=(520 * n, 520), border=False)
for i, (f, t) in enumerate(zip(args.vtk, args.times)):
    pl.subplot(0, i)
    m = pv.read(f).extract_surface().compute_normals(auto_orient_normals=True)
    pl.add_mesh(m, color="#e3d78a", smooth_shading=True, specular=0.35, specular_power=25, diffuse=0.8, ambient=0.12)
    zoom = 2.2 - 1.2 * t
    pl.camera.position = (-1.5 * zoom, 1.5 * zoom, -1.5 * zoom)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up = (0.0, 0.0, -1.0)
    pl.camera.view_angle = args.view_angle
    label = args.labels[i] if args.labels else f"t = {t:g}"
    pl.add_text(label, font_size=14, color="black")
pl.set_background("white")
pl.screenshot(args.png)
print(args.png)
