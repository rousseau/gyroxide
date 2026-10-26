"""Off-screen rendering of triangle surfaces with numpy only (no OpenGL, no pyvista): orthographic
camera, z-buffer, normals interpolated per pixel (smooth shading), 2x supersampling.

    img = render(vertices, faces, colors, view="lateral", width=900)   # (h, w, 3) floats in [0, 1]
"""
import numpy as np

# Camera directions in the mesh frame (Tallinen's normalisation: x antero-posterior, y left-right,
# z flipped, so screen up is -z):
# (direction the camera looks along, screen up)
VIEWS = {
    "lateral": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "medial": ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "superior": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}


def camera(view):
    """Rows: screen x, screen y (up), towards the viewer."""
    look, up = (np.asarray(v, float) for v in (VIEWS[view] if isinstance(view, str) else view))
    look /= np.linalg.norm(look)
    up = up - (up @ look) * look
    up /= np.linalg.norm(up)
    return np.stack([np.cross(look, up), up, -look])


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def render(v, f, colors, view="lateral", width=900, box=None, ss=2, background=(0.988, 0.988, 0.984),
           light=(-0.4, 0.5, 0.75), ambient=0.38, specular=0.12):
    """v (n, 3), f (m, 3) int, colors (n, 3) per vertex. box = (x0, x1, y0, y1) in camera coordinates
    (same units as v) sets the framing, so that several surfaces share one scale; default: fit v."""
    r = camera(view)
    p = v @ r.T
    vn = vertex_normals(v, f) @ r.T
    if box is None:
        lo, hi = p[:, :2].min(0), p[:, :2].max(0)
        pad = 0.03 * (hi - lo).max()
        box = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
    x0, x1, y0, y1 = box
    W = int(width * ss)
    H = int(round(W * (y1 - y0) / (x1 - x0)))
    sx = W / (x1 - x0)
    px = (p[:, 0] - x0) * sx
    py = (y1 - p[:, 1]) * sx
    pz = p[:, 2]
    # triangles facing the viewer only (closed surfaces), visible in the frame
    area = ((px[f[:, 1]] - px[f[:, 0]]) * (py[f[:, 2]] - py[f[:, 0]])
            - (px[f[:, 2]] - px[f[:, 0]]) * (py[f[:, 1]] - py[f[:, 0]]))
    t = f[area != 0]
    tx, ty = px[t], py[t]
    bx0 = np.clip(np.floor(tx.min(1)).astype(int), 0, W - 1)
    bx1 = np.clip(np.ceil(tx.max(1)).astype(int), 0, W - 1)
    by0 = np.clip(np.floor(ty.min(1)).astype(int), 0, H - 1)
    by1 = np.clip(np.ceil(ty.max(1)).astype(int), 0, H - 1)
    nw, nh = bx1 - bx0 + 1, by1 - by0 + 1
    keep = (tx.max(1) >= 0) & (tx.min(1) < W) & (ty.max(1) >= 0) & (ty.min(1) < H)
    t, tx, ty, bx0, by0, nw, nh = t[keep], tx[keep], ty[keep], bx0[keep], by0[keep], nw[keep], nh[keep]
    zbuf = np.full(H * W, np.inf)
    tri = np.full(H * W, -1)
    bary = np.zeros((H * W, 3))
    # candidate pixels, in chunks of triangles to bound memory
    cnt = nw * nh
    starts = np.concatenate([[0], np.cumsum(cnt)])
    chunk = 4_000_000
    i = 0
    while i < len(t):
        j = np.searchsorted(starts, starts[i] + chunk, side="right") - 1
        j = max(j, i + 1)
        c = cnt[i:j]
        idx = np.repeat(np.arange(i, j), c)
        loc = np.arange(c.sum()) - np.repeat(starts[i:j] - starts[i], c)
        qx = bx0[idx] + loc % nw[idx] + 0.5
        qy = by0[idx] + loc // nw[idx] + 0.5
        ax, ay = tx[idx, 0], ty[idx, 0]
        bxv, byv = tx[idx, 1], ty[idx, 1]
        cx, cy = tx[idx, 2], ty[idx, 2]
        den = (byv - cy) * (ax - cx) + (cx - bxv) * (ay - cy)
        den = np.where(den == 0, 1e-12, den)
        l0 = ((byv - cy) * (qx - cx) + (cx - bxv) * (qy - cy)) / den
        l1 = ((cy - ay) * (qx - cx) + (ax - cx) * (qy - cy)) / den
        l2 = 1 - l0 - l1
        ins = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        idx, l0, l1, l2 = idx[ins], l0[ins], l1[ins], l2[ins]
        pix = (qy[ins].astype(int)) * W + qx[ins].astype(int)
        z = l0 * pz[t[idx, 0]] + l1 * pz[t[idx, 1]] + l2 * pz[t[idx, 2]]
        # nearest (largest z: towards the viewer) per pixel, merged with the buffer
        o = np.lexsort((-z, pix))
        pix, z, idx, l0, l1, l2 = pix[o], z[o], idx[o], l0[o], l1[o], l2[o]
        first = np.r_[True, pix[1:] != pix[:-1]]
        pix, z, idx, l0, l1, l2 = pix[first], z[first], idx[first], l0[first], l1[first], l2[first]
        better = -z < zbuf[pix]
        pix = pix[better]
        zbuf[pix] = -z[better]
        tri[pix] = idx[better]
        bary[pix] = np.c_[l0, l1, l2][better]
        i = j
    img = np.tile(np.asarray(background, float), (H * W, 1))
    hit = tri >= 0
    tv = t[tri[hit]]
    b = bary[hit][:, :, None]
    n = (vn[tv] * b).sum(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n *= np.where(n[:, 2:3] < 0, -1.0, 1.0)      # two-sided: normals towards the viewer
    col = (colors[tv] * b).sum(1)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    diff = np.clip(n @ L, 0, 1)
    hvec = L + np.array([0.0, 0.0, 1.0])
    hvec /= np.linalg.norm(hvec)
    spec = specular * np.clip(n @ hvec, 0, 1) ** 24
    img[hit] = np.clip(col * (ambient + (1 - ambient) * diff)[:, None] + spec[:, None], 0, 1)
    img = img.reshape(H, W, 3)
    if ss > 1:
        H2, W2 = H // ss, W // ss
        img = img[:H2 * ss, :W2 * ss].reshape(H2, ss, W2, ss, 3).mean((1, 3))
    return img
