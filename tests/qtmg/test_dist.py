"""qtmg.dist and qtmg.gpu: ownership chunks, halos, slab domains, sharing of the contour points, LU host views."""
import numpy as np
from qtmg import dist, gpu


def test_lu_host_view():
    """LU factors through the transposed, C-contiguous host form of a Streamer item (CPU mode: numpy)."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
    b = rng.standard_normal((9, 2)) + 0j
    lu = gpu.lu_factor(np.asfortranarray(M.copy()))
    it = gpu.lu_host(lu)
    assert it[0].flags.c_contiguous
    assert np.allclose(M @ gpu.lu_solve(gpu.lu_view(it), b), b)
    xs = {}
    gpu.Streamer([it, it]).run(lambda k, v: xs.update({k: gpu.lu_solve(gpu.lu_view(v), b)}))
    assert all(np.allclose(M @ x, b) for x in xs.values())


def test_chunks_and_halo():
    assert dist.chunks([1, 1, 100, 1, 1], 4).tolist() and set(dist.chunks([1, 1, 100, 1, 1], 4)) == {0, 1, 2, 3}
    assert dist.chunks(np.ones(8), 4).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    n = 23

    def f(c):
        part = dist.Part(c, [np.arange(0, 7), np.arange(7, 9), np.arange(9, 23)], [0, 1, 2], n)
        need = np.random.default_rng(c.rank).choice(n, 9, replace=False)
        ex = dist.Exch(part, need)
        g = np.arange(n, dtype=complex)[:, None] * np.ones(2)
        x = ex(part.local(g))
        assert np.array_equal(x[ex.pos[need]], g[need])
        z = part.local(np.zeros((n, 2), complex))
        zx = np.zeros((part.nloc + ex.ng, 2), complex)
        zx[:part.nloc] = z
        zx[part.nloc:] = g[np.array(sorted(range(n), key=lambda i: ex.pos[i]))[-ex.ng:]] if ex.ng else 0
        return part.gather(ex.rev(zx, z))[:, 0], need, part.gown
    outs = dist.ThreadComm.run(3, f)
    sent = {int(i) for r, (_, need, gown) in enumerate(outs) for i in need if gown[i] != r}
    for z, _, _ in outs:
        assert all(z[i] == (i if i in sent else 0) for i in range(n))


def test_share_points_is_balanced():
    rng = np.random.default_rng(1)
    for size, nc, npts in [(4, 3, 24), (4, 5, 24), (8, 3, 24), (3, 11, 8), (2, 2, 1), (5, 1, 4)]:
        own = dist.chunks(np.ones(nc), min(size, nc)) * (size // min(size, nc))
        cost = rng.uniform(1, 3, nc) ** 3
        pts = dist.share_points(cost, own, npts, size)
        assert pts == dist.share_points(cost, own, npts, size)
        load0, load = np.zeros(size), np.zeros(size)
        for c in range(nc):
            assert sorted(q for d in pts[c].values() for q in d) == list(range(npts)) and own[c] in pts[c]
            load0[own[c]] += cost[c] * npts
            for r, q in pts[c].items():
                load[r] += cost[c] * len(q)
        assert load.max() <= load0.max() + 1e-12
        if npts == 24:
            assert load.max() <= 1.1 * load.mean() + cost.max()      # one point is the unit; helpers pay one more
    # the points really move in the hierarchy tests
    assert any(len(d) > 1 for d in dist.share_points([8.0, 1.0, 1.0], [0, 1, 2], 8, 3))


def test_domains():
    for S in range(1, 30):
        for size in range(1, 10):
            dom, sep, ranks = dist.domains(np.ones(S), size)
            assert len(dom) == min(size, (S + 1) // 2) == len(set(ranks)) and max(ranks) < size
            assert all(b > a for a, b in dom)
            assert sorted([s for a, b in dom for s in range(a, b)] + sep) == list(range(S))
    assert sum(b - a for a, b in dist.domains(np.ones(40), 4)[0]) == 37
