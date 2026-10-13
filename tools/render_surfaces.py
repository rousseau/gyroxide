#!/usr/bin/env python3
"""Renders gyroxide VTK surfaces side by side (off-screen) into a PNG.

    python3 tools/render_surfaces.py out.png out/surface_0000000.vtk out/surface_final.vtk --labels "t = 0" "t = 1"
"""
import argparse

import pyvista as pv

ap = argparse.ArgumentParser()
ap.add_argument("png")
ap.add_argument("vtk", nargs="+")
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--view", default="xz", help="camera plane: xy, xz or yz")
ap.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, -1.0],
                help="screen-up vector; Tallinen's normalisation flips z, so up is -z (POV-Ray sky <0,0,-1>)")
args = ap.parse_args()

n = len(args.vtk)
pl = pv.Plotter(shape=(1, n), off_screen=True, window_size=(700 * n, 700), border=False)
for i, f in enumerate(args.vtk):
    pl.subplot(0, i)
    m = pv.read(f).compute_normals(auto_orient_normals=True)
    pl.add_mesh(m, color="#e8c9a8", smooth_shading=True, specular=0.3, ambient=0.15)
    if args.labels and i < len(args.labels):
        pl.add_text(args.labels[i], font_size=14, color="black")
    getattr(pl, f"view_{args.view}")()
    pl.camera.up = tuple(args.up)
    pl.camera.zoom(1.3)
pl.set_background("white")
pl.screenshot(args.png)
print(args.png)
