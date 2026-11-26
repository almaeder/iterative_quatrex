"""Structure of the hierarchy, independent of the energy and of the bases: the contour of the density matrices, the fine
CAP-Schwarz windows (level-1 blocks = their cores), the levels made by merging pairs of blocks, and the windows of every level."""
from dataclasses import dataclass, field

import numpy as np

from .backend import log
from .device import periodic_axes


def contour(mu1, mu2, npts):
    """Upper half circle over [mu1, mu2]: points z, weights dz (Gauss-Legendre in the angle)."""
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    t, w = np.polynomial.legendre.leggauss(int(npts))
    th, w = 0.5 * np.pi * (t + 1), 0.5 * np.pi * w
    return c + r * np.exp(1j * th), -1j * r * np.exp(1j * th) * w


def contours(text, npts, Ec):
    """"M1,M2[,NPTS[,WEIGHT]]:..." relative to Ec -> points and weights of rho = sum_s weight_s rho_s."""
    zs, ws = [], []
    for c in text.split(":"):
        v = c.split(",")
        z, w = contour(Ec + float(v[0]), Ec + float(v[1]), int(v[2]) if len(v) > 2 else npts)
        zs.append(z), ws.append(w * (float(v[3]) if len(v) > 3 else 1.0))
    return np.concatenate(zs), np.concatenate(ws)


# =====================================================================================================================
# fine windows (CAP-Schwarz geometry: x windows centred on the slab interfaces and between them, optional transverse strips)
# =====================================================================================================================
@dataclass
class FineWindow:
    idx: np.ndarray                         # device orbitals of the window (sorted)
    core: np.ndarray                        # bool over idx: the orbitals this window owns (the cores partition the device)
    ramp: np.ndarray                        # absorbing ramp 0..1 over idx towards the artificial cuts


def fine_windows(device, inp, width, d, spacing, split, tov):
    """Windows of `width` Å at the slab interfaces and at points between them (spacing), core = the orbitals nearest to the
    centre; each periodic transverse axis cut into split[i] strips of period / n + tov Å on each side. Ramp p = clip(...)^2
    over d Å towards every artificial cut (x cuts that are not lead faces, transverse cuts)."""
    x, sb = device.x, device.slab_bounds
    slabs = [np.arange(sb[k], sb[k + 1]) for k in range(len(sb) - 1)]
    ifaces = [0.5 * (x[slabs[k]].max() + x[slabs[k + 1]].min()) for k in range(len(slabs) - 1)]
    anchors, fill = [x.min()] + ifaces + [x.max()], []
    for lo, hi in zip(anchors[:-1], anchors[1:]):
        m = max(1, int(np.ceil((hi - lo) / spacing)) - 1) if spacing else 1
        fill += [lo + (hi - lo) * (q + 1) / (m + 1) for q in range(m)]
    centres = np.sort(np.array(ifaces + fill))
    near = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)
    wins = []                                                         # (idx, core, transverse cuts [(|du|, half width)])
    for k, c in enumerate(centres):
        i = np.where((np.abs(x - c) < width / 2) | (near == k))[0]
        wins.append((i, near[i] == k, []))
    for (ax, L), n in zip(periodic_axes(device.meta.get("periodic")), split or []):
        if n <= 1:
            continue
        u = np.asarray(inp["xyz"])[device.atom, ax]
        box = np.minimum((u % L * (n / L)).astype(int), n - 1)        # owner strip of every orbital
        out = []
        for i, c, cuts in wins:
            for j in range(n):
                du = np.abs((u[i] - (j + 0.5) * L / n + L / 2) % L - L / 2)   # minimum-image distance to the strip centre
                keep = du < L / (2 * n) + tov
                out.append((i[keep], c[keep] & (box[i[keep]] == j), [(a[keep], h) for a, h in cuts] + [(du[keep], L / (2 * n) + tov)]))
        wins = out
    xmin, xmax, res = x.min(), x.max(), []
    for i, c, cuts in wins:
        xi, ramp = x[i], np.zeros(len(i))
        if xi.min() > xmin + 1e-6:                                    # cut on the left (not the lead face)
            ramp = np.maximum(ramp, np.clip((xi.min() + d - xi) / d, 0, 1) ** 2)
        if xi.max() < xmax - 1e-6:
            ramp = np.maximum(ramp, np.clip((xi - (xi.max() - d)) / d, 0, 1) ** 2)
        for du, h in cuts:
            ramp = np.maximum(ramp, np.clip((du - (h - d)) / d, 0, 1) ** 2)
        res.append(FineWindow(i, c, ramp))
    cover = np.zeros(device.n, int)
    for w in res:
        cover[w.idx[w.core]] += 1
    assert (cover == 1).all(), "window cores must partition the device"
    log(f"fine windows: {len(res)} ({len(centres)} x positions at {', '.join(f'{c:.1f}' for c in centres)} Å), "
        f"{min(len(w.idx) for w in res)}..{max(len(w.idx) for w in res)} orbitals")
    return res, len(centres)


# =====================================================================================================================
# structure of the hierarchy (energy and basis independent)
# =====================================================================================================================
@dataclass
class Window:
    blocks: np.ndarray                      # level-k blocks of the window
    parents: list                           # level-(k+1) blocks whose kids are kept (rows of the smoother, basis of the parents)
    owner: int = 0                          # rank that builds and applies it (the owner of its parents)


@dataclass
class Level:
    kids: list                              # level-(k-1) blocks of each block (None on level 1)
    grid: tuple                             # (x index, y index) of each block
    G: np.ndarray                           # coupled blocks (bool matrix)
    of_core: np.ndarray                     # level-k block of every fine core
    owner: np.ndarray                       # rank of every block (level 1: by x position; merged: the owner of its first kid)
    wins: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.G)

    def nbr(self, i):
        return np.where(self.G[i])[0]

    def pairs(self):
        return [(i, int(j)) for i in range(self.n) for j in self.nbr(i) if i <= j]

    def inside(self, blocks):
        """Coupled pairs (i <= j) with both blocks in the list."""
        s = set(int(b) for b in blocks)
        return [(i, int(j)) for i in sorted(s) for j in self.nbr(i) if i <= j and int(j) in s]


def merge(L, axis, halo):
    """Next level: pairs of blocks of L along axis. Fills L.wins: per parent, the blocks coupled to its kids (within halo x
    positions; None: all); parents of one owner with the same blocks share a window."""
    ix, iy = L.grid
    key = (ix // 2) * 100000 + iy if axis == "x" else ix * 100000 + iy // 2
    kids = [np.where(key == v)[0] for v in np.unique(key)]
    first = np.array([ks[0] for ks in kids])
    parent = np.empty(L.n, int)
    for c, ks in enumerate(kids):
        parent[ks] = c
    M = np.zeros((len(kids), L.n), np.int32)
    M[parent, np.arange(L.n)] = 1
    grid = (ix[first] // 2, iy[first]) if axis == "x" else (ix[first], iy[first] // 2)
    N = Level(kids, grid, (M @ L.G.astype(np.int32) @ M.T) > 0, parent[L.of_core], L.owner[first])
    groups = {}
    for c, ks in enumerate(kids):
        w = np.unique(np.concatenate([L.nbr(b) for b in ks]))
        if halo is not None:
            w = w[np.abs(ix[w][:, None] - ix[ks][None, :]).min(1) <= halo]
        groups.setdefault((int(N.owner[c]), tuple(int(b) for b in w)), []).append(c)
    L.wins = [Window(np.array(w), ps, o) for (o, w), ps in groups.items()]
    return N

