"""qtxmg.comm on thread ranks: ownership chunks, slab domains, distributed vectors and halos, keyed exchanges, sums;
the block-tridiagonal LU of qtxmg.hierarchy against a dense solve."""
import numpy as np
import pytest

from qtxmg.comm import Comm, Halo, Part, ThreadComm, chunks, domains, swap
from qtxmg.hierarchy import tri_factor, tri_solve


def test_chunks():
    assert chunks(np.ones(8), 4).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    own = chunks([1, 1, 100, 1, 1], 4)
    assert set(own) == {0, 1, 2, 3} and np.all(np.diff(own) >= 0)
    for n in range(1, 20):
        for size in range(1, n + 1):
            own = chunks(np.random.default_rng(n).uniform(1, 5, n), size)
            assert set(own) == set(range(size)) and np.all(np.diff(own) >= 0)


def test_domains():
    for S in range(1, 30):
        for size in range(1, 10):
            dom, sep, ranks = domains(np.ones(S), size)
            assert len(dom) == min(size, (S + 1) // 2) == len(set(ranks)) and max(ranks) < size
            assert all(b > a for a, b in dom)
            assert sorted([s for a, b in dom for s in range(a, b)] + sep) == list(range(S))
            assert all(dom[j][1] == s and dom[j + 1][0] == s + 1 for j, s in enumerate(sep))     # one slab between domains


@pytest.mark.parametrize("size", [1, 2, 3])
def test_part_halo(size):
    """Halo reads the rows of other ranks; rev writes the rows computed here for other ranks back to their owners."""
    n, blocks = 23, [np.arange(0, 7), np.arange(7, 9), np.arange(9, 16), np.arange(16, 23)]
    owner = [0, min(1, size - 1), min(2, size - 1), size - 1]
    g = np.arange(n, dtype=complex)[:, None] * np.array([1, 1j])

    def f(c):
        part = Part(c, blocks, owner, n)
        need = np.random.default_rng(c.rank).choice(n, 9, replace=False)
        h = Halo(part, need)
        x = h(g[part.idx])
        assert np.array_equal(x[h.pos[need]], g[need])
        assert np.array_equal(part.gather(g[part.idx]), g)
        z = np.zeros((part.nloc + h.nghost, 2), complex)            # rev: this rank computes its ghost rows (value 10 + rank)
        z[part.nloc:] = 10 + c.rank
        return part.gather(h.rev(z)), need, part.owner_of
    outs = ThreadComm.run(size, f)
    writer = {}
    for r, (_, need, own) in enumerate(outs):
        for i in need:
            if own[i] != r:
                writer[int(i)] = r
    for z, _, _ in outs:
        assert all(z[i, 0] == (10 + writer[i] if i in writer else 0) for i in range(n))


def test_swap_sum_merge():
    def f(c):
        have = {("a", c.rank): np.full((2, 3), c.rank, complex)}
        wants = [[("a", (r + 1) % c.size)] for r in range(c.size)]
        got = swap(c, have, wants, lambda q: q[1], lambda q: (2, 3), complex)
        s = c.sum(np.full(4, c.rank + 0.1))
        return got, s, c.merge({c.rank: c.rank ** 2}), c.bcast(c.rank, root=2)
    for r, (got, s, m, b) in enumerate(ThreadComm.run(4, f)):
        assert np.all(got[("a", (r + 1) % 4)] == (r + 1) % 4)
        assert np.allclose(s, 6.4) and m == {0: 0, 1: 1, 2: 4, 3: 9} and b == 2
    assert Comm().sum(np.ones(3)).tolist() == [1, 1, 1]


@pytest.mark.parametrize("S", [1, 2, 5])
def test_tri_lu(S):
    rng = np.random.default_rng(S)
    n = rng.integers(2, 6, S)
    o = np.concatenate([[0], np.cumsum(n)])
    rnd = lambda a, b: rng.standard_normal((a, b)) + 1j * rng.standard_normal((a, b))
    D = [rnd(n[a], n[a]) + 4 * np.eye(n[a]) for a in range(S)]
    U = [rnd(n[a], n[a + 1]) for a in range(S - 1)]
    L = [rnd(n[a + 1], n[a]) for a in range(S - 1)]
    A = np.zeros((o[-1], o[-1]), complex)
    for a in range(S):
        A[o[a]:o[a + 1], o[a]:o[a + 1]] = D[a]
        if a + 1 < S:
            A[o[a]:o[a + 1], o[a + 1]:o[a + 2]], A[o[a + 1]:o[a + 2], o[a]:o[a + 1]] = U[a], L[a]
    b = rnd(o[-1], 3)
    x = np.concatenate(tri_solve(tri_factor([d.copy() for d in D], [u.copy() for u in U], L), [b[o[a]:o[a + 1]] for a in range(S)]))
    assert np.linalg.norm(A @ x - b) < 1e-10 * np.linalg.norm(b)
