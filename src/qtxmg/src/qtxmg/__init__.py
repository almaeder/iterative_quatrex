"""qtxmg: open-boundary Schroedinger (QTBM) transport with a density-matrix mode-space hierarchy as the coarse levels of a
multigrid preconditioner in flexible GMRES, on one or several GPUs.

    single      the single-GPU solver in one file (python -m qtxmg, or the qtxmg command)
    run         the multi-GPU solver, one MPI rank per GPU (python -m qtxmg.run, or qtxmg-multi); one rank without --dist
      backend     numpy / cupy switch, dense LU helpers, host-held blocks
      comm        communicators (one rank, MPI, NCCL), distributed vectors and halos, ownership chunks, slab domains
      device      device H, S, geometry, lead faces; S and H rows on the GPU
      structure   contour, fine CAP-Schwarz windows, levels and their windows
      obc         lead Sigma and injection (Beyn or folded), computed once over the ranks
      hierarchy   bases, Galerkin blocks, smoothers, distributed block-tridiagonal LU of the deepest level, V-cycle
      krylov      flexible GMRES on distributed vectors
    beyn        lead self-energy and injection from the Bloch modes of the periodized lead (Beyn)
    gpu         numpy / cupy helpers used by beyn
    prep        input sets: prep.inputs (read / write), prep.quatrex (QuaTrEx device folder -> input set)
"""
__version__ = "0.2.0"
