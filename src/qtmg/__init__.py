"""qtmg: open-boundary Schroedinger (QTBM) transport with a density-matrix mode-space hierarchy as the coarse levels of a
multigrid preconditioner in flexible GMRES, on one or several GPUs.

    single      the single-GPU solver in one file (python -m qtmg, or the qtmg command)
    hier        the multi-GPU solver qtx_ms.hier of the development code, as it is (python -m qtmg.hier.run, or qtmg-multi):
                run (driver), plan, levels, fine, obc, slow; with the parts of the development modules it uses:
      dist        communicators (one rank, MPI, NCCL), distributed vectors and halos, ownership chunks, slab domains
      schwarz     CAP-Schwarz fine windows, block FGMRES
      msbasis     device, geometry, lead faces, log
      mlevel      launcher (start, guarded)
      devstore    sliced device store (python -m qtmg.devstore), input reader
      abtb        Sancho decimation
    obc_beyn    Beyn lead self-energy and injection (single and hier)
    gpu         numpy / cupy helpers
    prep        input sets: prep.inputs (read / write, also qtx_modespace.prep folders), prep.quatrex (QuaTrEx folder -> input set)
The code of the development modules that none of these reaches is removed (workflows/prune_release.py)."""
__version__ = "0.2.0"
