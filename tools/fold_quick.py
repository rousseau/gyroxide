"""Quick folding metrics of equilibrium states against the atlas of their week (week read from the
/w<SG>, /fit<SG> or /sg<SG> part of the path): cortex area, then hull area, mean and p90 depth to the convex hull, as
ratios to the atlas.

    python3 tools/fold_quick.py out/lever/mu/w30/v0.25/eq/x_released.npy ...
"""
import json, re, sys, numpy as np
sys.path.insert(0, 'tools')
from mota_law import vertex_areas
meta = json.loads(open('data/meshes/w21_left_610k_s.json').read()); ns = meta['n_surface']
f = np.load('data/adjoint_prep_s/w22/faces.npy').astype(int); c = np.load('data/adjoint_prep_s/w22/cortex.npy')[:ns] > 0.5
for p in sys.argv[1:]:
    w = int(re.search(r"/(?:w|fit|sg)(\d\d)", p).group(1))
    y = np.load(f'data/adjoint_prep_s/w{w}/target.npy')[:ns]
    print(f"{p}: realized/atlas {vertex_areas(np.load(p)[:ns], f, ns)[c].sum() / vertex_areas(y, f, ns)[c].sum():.3f}")
from scipy.spatial import ConvexHull
mm = meta['maxd_mm']
def depth(q):
    eq = ConvexHull(q).equations
    return (-(q @ eq[:, :3].T + eq[:, 3])).min(1) * mm
print("A_e ratio, mean depth / atlas, p90 depth / atlas")
for p in sys.argv[1:]:
    w = int(re.search(r"/(?:w|fit|sg)(\d\d)", p).group(1)); y = np.load(f'data/adjoint_prep_s/w{w}/target.npy')[:ns]; x = np.load(p)[:ns]
    dy, dx = depth(y)[c], depth(x)[c]
    print(f"{p}: {ConvexHull(x).area / ConvexHull(y).area:.3f} {dx.mean() / dy.mean():.3f} {np.percentile(dx, 90) / np.percentile(dy, 90):.3f}")
