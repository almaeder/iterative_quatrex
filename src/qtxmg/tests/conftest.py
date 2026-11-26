"""Synthetic transport device of the unit tests: a strip of sites on a cubic grid (spacing 2 Å), `no` orbitals per site,
nearest-neighbour H and S (non-orthogonal), random on-site disorder in the device blocks. Transport blocks are 3 x-planes, so
the lead cell has an interior plane (folded by the folded lead, periodized by Beyn). dense_T: Caroli T on the whole device."""
import numpy as np
import pytest
import scipy.sparse as sp

from qtxmg.prep.inputs import write_inputs


def strip(nd=8, ny=4, nz=2, no=2, a=2.0, disorder=0.4, seed=0):
    rng = np.random.default_rng(seed)
    e0 = np.diag(np.linspace(0.0, 1.5, no))
    t = -np.eye(no) + 0.2 * (lambda M: M + M.T)(rng.normal(size=(no, no)))
    s1 = 0.08 * np.eye(no) + 0.02 * (lambda M: M + M.T)(rng.normal(size=(no, no)))
    npl, nb = 3, nd + 2                                               # x-planes per block, blocks [lead | device | lead]

    def build(nx, dis):
        sites = np.array([(ix, iy, iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)])
        idx = {tuple(s): i for i, s in enumerate(sites)}
        n = len(sites) * no
        H, S = sp.lil_matrix((n, n)), sp.lil_matrix((n, n))
        for i, s in enumerate(sites):
            o = slice(i * no, (i + 1) * no)
            H[o, o], S[o, o] = e0 + dis[i] * np.eye(no), np.eye(no)
            for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                j = idx.get(tuple(s + d))
                if j is not None:
                    p = slice(j * no, (j + 1) * no)
                    H[o, p], H[p, o], S[o, p], S[p, o] = t, t.T, s1, s1.T
        return sites, H.tocsr(), S.tocsr()
    nx = nb * npl
    ns = nx * ny * nz
    sites0 = np.array([(ix, iy, iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)])
    blocks = sites0[:, 0] // npl
    dis = np.where((blocks > 0) & (blocks < nb - 1), disorder * rng.uniform(-1, 1, ns), 0.0)
    sites, H, S = build(nx, dis)
    _, H2, S2 = build(2 * npl, np.zeros(2 * npl * ny * nz))           # two pristine cells: H00, H01
    m = npl * ny * nz * no
    lead = dict(H00=H2[:m, :m].tocsr(), H01=H2[:m, m:].tocsr(), S00=S2[:m, :m].tocsr(), S01=S2[:m, m:].tocsr())
    meta = dict(name="strip", units=dict(energy="eV", length="Angstrom"), axis=0, nblocks=nb, periodic=[dict(axis=1, length=2.0 * ny)],
                transport=dict(device_blocks=list(range(1, nb - 1)), lead_left_block=0, lead_right_block=nb - 1),
                edges=dict(Ec_lead=0.35))
    return dict(labels=np.array(["X"] * ns), xyz=(sites * a).astype(float), norb=np.full(ns, no), blocks=blocks, H=H, S=S,
                lead=lead, meta=meta)


def surface_gf(A00, A01, A10, tol=1e-13, maxit=500):
    """Sancho-Rubio: surface Green's function of the semi-infinite chain ... A10 [A00] A01 ... (the surface cell couples on by A01)."""
    a, b, es, e = A01.copy(), A10.copy(), A00.copy(), A00.copy()
    for _ in range(maxit):
        g = np.linalg.inv(e)
        ag, bg = a @ g, b @ g
        es = es - ag @ b
        e = e - ag @ b - bg @ a
        a, b = ag @ a, bg @ b
        if max(abs(a).max(), abs(b).max()) < tol:
            break
    return np.linalg.inv(es)


def dense_T(inp, E, eta=1e-6):
    """Caroli T(E) of the device, Sigma from the lead cells as given (no folding)."""
    m = inp["meta"]["transport"]
    ob = np.repeat(inp["blocks"], inp["norb"])
    dev = np.concatenate([np.where(ob == b)[0] for b in m["device_blocks"]])
    L, R = np.where(ob == m["lead_left_block"])[0], np.where(ob == m["lead_right_block"])[0]
    z, ld = E + 1j * eta, inp["lead"]
    A = (z * inp["S"] - inp["H"]).toarray()
    A00, A01 = (z * ld["S00"] - ld["H00"]).toarray(), (z * ld["S01"] - ld["H01"]).toarray()
    A10 = (z * ld["S01"].T - ld["H01"].T).toarray()
    gL, gR = surface_gf(A00, A10, A01), surface_gf(A00, A01, A10)
    SL = A[np.ix_(dev, L)] @ gL @ A[np.ix_(L, dev)]
    SR = A[np.ix_(dev, R)] @ gR @ A[np.ix_(R, dev)]
    G = np.linalg.inv((E * inp["S"] - inp["H"]).toarray()[np.ix_(dev, dev)] - SL - SR)
    GL, GR = 1j * (SL - SL.conj().T), 1j * (SR - SR.conj().T)
    return float(np.real(np.trace(GR @ G @ GL @ G.conj().T)))


@pytest.fixture(scope="session")
def device_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("strip")
    inp = strip()
    write_inputs(d / "dev", inp["labels"], inp["xyz"], inp["norb"], inp["blocks"], inp["H"], inp["S"], inp["lead"], inp["meta"])
    return d / "dev", inp
