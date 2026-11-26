"""Right-preconditioned flexible GMRES for a block of right-hand sides: every column has its own Krylov space and Hessenberg
matrix; the operator and the preconditioner act on the block of the active columns' newest vectors at once. Vectors are the
rows a rank owns; dot products are summed over the ranks (one message per Gram-Schmidt pass for all columns)."""
import numpy as np

from .backend import to_host, xp
from .comm import Comm


def fgmres(A, b, M, tol=1e-10, m=100, comm=None, callback=None):
    """x = argmin |b - A x| over span(Z), Z = M^-1 V (x0 = 0); a column stops when its residual estimate is below tol |b_j|.
    Returns x, iterations per column, true relative residuals |b - A x| / |b| (host)."""
    comm = comm or Comm()
    n, nc = b.shape
    norms = lambda X: np.sqrt(np.asarray(comm.sum(to_host((xp.abs(X) ** 2).sum(axis=0)))))
    bn = norms(b)
    beta = bn.copy()
    V = xp.zeros((nc, m + 1, n), complex)                             # Krylov basis, one row per vector
    Z = xp.zeros((nc, m, n), complex)                                 # preconditioned vectors
    Hh = np.zeros((nc, m + 1, m), complex)
    V[:, 0] = (b / xp.asarray(np.where(beta > 0, beta, 1))).T
    its, est, y = np.zeros(nc, int), beta / bn, [np.zeros(0, complex)] * nc
    act = np.where(beta > tol * bn)[0]
    for i in range(m):
        if not len(act):
            break
        Zi = M(xp.ascontiguousarray(V[act, i].T))
        Z[act, i] = Zi.T
        W = A(Zi)
        for _ in range(2):                                            # classical Gram-Schmidt, twice
            h = comm.sum(xp.stack([V[j, :i + 1].conj() @ W[:, q] for q, j in enumerate(act)]))
            for q, j in enumerate(act):
                W[:, q] -= h[q] @ V[j, :i + 1]
            Hh[act, :i + 1, i] += to_host(h)
        hn = norms(W)
        keep = []
        for q, j in enumerate(act):
            Hh[j, i + 1, i] = hn[q]
            if hn[q] > 0:
                V[j, i + 1] = W[:, q] / hn[q]
            e1 = np.zeros(i + 2, complex)
            e1[0] = beta[j]
            y[j] = np.linalg.lstsq(Hh[j, :i + 2, :i + 1], e1, rcond=None)[0]
            its[j] = i + 1
            est[j] = np.linalg.norm(Hh[j, :i + 2, :i + 1] @ y[j] - e1) / bn[j]
            if est[j] > tol and hn[q] > 0:
                keep.append(j)
        act = np.array(keep, int)
        if callback:
            callback(i, est)
    x = xp.zeros_like(b)
    for j in range(nc):
        if its[j]:
            x[:, j] = xp.asarray(y[j]) @ Z[j, :its[j]]
    return x, its, norms(b - A(x)) / bn
