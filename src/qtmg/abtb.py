"""Ab initio tight-binding input: sparse H, S, xyz structure, orbitals per atom.

Reads the device, sorts atoms along the transport direction, partitions them into slabs so that
A = E S - H is block tridiagonal, and builds the lead self-energies (Sancho-Rubio, non-orthogonal).

Assumptions (check against your files):
  * H and S are the full device matrices (no k-points), orbitals of each atom consecutive and atoms
    in the same order as in the xyz file.
  * The first and last slabs are one lead unit cell each, and the device starts and ends with
    at least two identical copies of that cell (standard for OMEN/NEGF device files).
"""
import numpy as np

UNITS = {"eV": 1.0, "Ry": 13.605693, "Ha": 27.211386}


# ------------------------------------------------------------------ readers


# ------------------------------------------------------------------ device


def sancho_rubio(A00, a, b, tol=1e-12, maxit=200):
    """Surface GF of a semi-infinite chain with on-site A00, coupling a to the next cell away from the
    surface and b back toward it.  Returns inv(A_surface)."""
    W, Ws = A00.copy(), A00.copy()
    a, b = a.copy(), b.copy()
    for _ in range(maxit):
        g = np.linalg.inv(W)
        agb, bga = a @ g @ b, b @ g @ a
        Ws = Ws - agb
        W = W - agb - bga
        a, b = -a @ g @ a, -b @ g @ b
        if np.abs(a).max() + np.abs(b).max() < tol:
            break
    return np.linalg.inv(Ws)


def injection(SigL, tol=1e-4):
    """Columns B with Gamma_L = B B^dagger (propagating channels only; eigenvalues of Gamma below
    tol [eV] come from the eta broadening of evanescent modes and are dropped)."""
    Gam = 1j * (SigL - SigL.conj().T)
    lam, V = np.linalg.eigh((Gam + Gam.conj().T) / 2)
    k = lam > tol
    return V[:, k] * np.sqrt(lam[k])


# ------------------------------------------------------------------ coarse bases
