"""qtxmg on the synthetic strip (CPU): the multi-GPU solver (qtxmg.run) on one rank against the dense T and the single-GPU
solver; on 2, 3, 4 thread ranks against one rank; Galerkin blocks, the distributed bottom LU, Sigma points computed once,
cached bases, Beyn against the folded lead."""
import collections
import time

import numpy as np
import pytest

from conftest import dense_T
from qtxmg import obc, run, single
from qtxmg.comm import Comm, ThreadComm

ARGS = ["--cap", "12,1,2", "--cap-spacing", "4", "--cap-split", "2", "--cap-tov", "1.5", "--up", "y:1e-3,x:1e-3,x:1e-3",
        "--mu=-0.25,0.5", "--npts", "8", "--cut", "1e-3", "--dE=-0.02,0.05", "--maxit", "60"]


def go(d, comm=None, *more, obc_model="folded"):
    a = run.parse(["--inputs", str(d)] + ARGS + ["--obc", obc_model] + list(more))
    return run._main(a, comm or Comm(), time.time())


def same(r, ref, tol=1e-9, its=1):
    assert r["N"] == ref["N"]
    for k, v in ref["E"].items():
        w = r["E"][k]
        assert w["converged"] and abs(w["T"] - v["T"]) < tol, (k, w["T"], v["T"])
        assert abs(max(w["its"]) - max(v["its"])) <= its


@pytest.fixture(scope="module")
def one(device_dir, tmp_path_factory):
    return go(device_dir[0], None, "--cache", str(tmp_path_factory.mktemp("one")))


def test_dense_T(device_dir, one, tmp_path):
    """T = the dense Caroli T when every channel is injected (--inj-tol 1e-12); the default drops channels with Gamma_L
    eigenvalues below 1e-4 eV (here up to 6e-7 of T)."""
    res, _ = go(device_dir[0], None, "--cache", str(tmp_path), "--inj-tol", "1e-12")
    for k, v in res["E"].items():
        ref = dense_T(device_dir[1], v["E"])
        assert v["converged"] and abs(v["T"] / ref - 1) < 1e-10
        assert abs(one[0]["E"][k]["T"] / ref - 1) < 1e-6


def test_single_equals_run(device_dir, one, tmp_path):
    res, _ = one
    ref, _ = single.main(["--inputs", str(device_dir[0])] + ARGS + ["--obc", "folded", "--cache", str(tmp_path)])
    same(res, ref, 1e-10)


def test_galerkin(one):
    """S_k, H_k blocks = P^T (S, H) P with P = P_1 ... P_k (dense)."""
    _, h = one
    P = np.eye(h.dev.n)
    for k in range(1, h.K + 1):
        Uk = np.zeros((P.shape[1], h.N[k]))
        for b in range(h.levels[k].n):
            rows = h.cores[b] if k == 1 else h.entries(k - 1, h.levels[k].kids[b])
            Uk[np.ix_(rows, h.entries(k, [b]))] = h.U[k][b]
        P = P @ Uk
        for q, M in enumerate((h.dev.S, h.dev.H)):
            D = P.T @ (M @ P)
            for (i, j) in h.levels[k].pairs():
                assert abs(h.SH[k][i, j][q] - D[np.ix_(h.entries(k, [i]), h.entries(k, [j]))]).max() < 1e-10


@pytest.mark.parametrize("size", [2, 3, 4])
def test_ranks(device_dir, one, tmp_path, size, monkeypatch):
    """Every rank: the same N, T and iterations as one rank; every Sigma point computed once over the ranks."""
    calls = collections.Counter()
    compute = obc.OBC.compute

    def counted(self, z, real):
        calls[complex(z)] += 1
        return compute(self, z, real)
    monkeypatch.setattr(obc.OBC, "compute", counted)
    rs = ThreadComm.run(size, lambda c: go(device_dir[0], c, "--cache", str(tmp_path)))
    assert len(calls) == 8 + 2 and set(calls.values()) == {1}        # 8 contour + 2 real points
    for res, h in rs:
        same(res, one[0])
    pairs = [{p for p in h.SH[1].keys()} for _, h in rs]              # Galerkin blocks: on every rank, the same
    assert all(p == pairs[0] for p in pairs)


@pytest.mark.parametrize("size", [3, 4])
def test_bottom_domains(device_dir, one, tmp_path, size):
    """The distributed LU of A_K (domains + separators on one rank) solves A_K; T as on one rank."""
    def f(c):
        res, h = go(device_dir[0], c, "--cache", str(tmp_path), "--up", "y:1e-3,x:1e-3")
        bt, k = h.bottom, h.K
        part = h.part[k]
        r = np.random.default_rng(1).standard_normal((h.N[k], 2)) + 0j
        x = part.gather(bt(r[part.idx]).astype(complex))
        sg = [Y.T @ (S @ Y) for S, (Y, _, _) in zip(h.Sig, h.Y[k])]   # A_K at the last energy, dense from the slab blocks
        ns = len(bt.slabs)
        A = np.block([[bt.block(a, b, h.E, sg) for b in range(ns)] for a in range(ns)])
        perm = np.concatenate([h.entries(k, s) for s in bt.slabs])
        return res, len(bt.sep), np.linalg.norm(A @ x[perm] - r[perm]) / np.linalg.norm(r)
    ref = go(device_dir[0], None, "--cache", str(tmp_path / "ref"), "--up", "y:1e-3,x:1e-3")[0]
    for res, nsep, err in ThreadComm.run(size, f):
        assert nsep >= 1 and err < 1e-5                               # complex64 in and out
        same(res, ref)


def test_cached_bases(device_dir, one, tmp_path, monkeypatch):
    """A second run reads the bases and every Sigma from the cache: nothing is computed, the same results."""
    go(device_dir[0], None, "--cache", str(tmp_path))
    calls = []
    monkeypatch.setattr(obc.OBC, "compute", lambda self, z, real: calls.append(z))
    res, h = go(device_dir[0], None, "--cache", str(tmp_path))
    assert not calls and h.have_bases()
    same(res, one[0], 1e-12, 0)


def test_beyn_equals_folded(device_dir, tmp_path):
    """The strip lead is exactly periodic: Beyn on its primitive cell (real points at E itself, incoming propagating modes) =
    the folded cell at E + 1e-9 i with every channel injected."""
    res, _ = go(device_dir[0], None, "--cache", str(tmp_path / "b"), obc_model="beyn")
    ref, _ = go(device_dir[0], None, "--cache", str(tmp_path / "f"), "--eta", "1e-9", "--inj-tol", "1e-12")
    for k, v in ref["E"].items():
        assert res["E"][k]["converged"] and abs(res["E"][k]["T"] / v["T"] - 1) < 1e-8
