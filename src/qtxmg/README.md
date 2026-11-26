# qtxmg 0.2.0

Open-boundary Schrödinger (QTBM) transport on one or several GPUs. A density-matrix mode-space hierarchy acts as the coarse levels of a
multigrid V-cycle, and that cycle preconditions flexible GMRES. The result is the exact T(E) of the tight-binding / ab initio
device, not a mode-space approximation.

## Method

At each energy z = E + iη (Σ at z, every operator at E = Re z):

```
A    = E S − H − Σ_L − Σ_R                                   fine operator (complex128)
M⁻¹r : z₁ = S₀ r;  z₂ = z₁ + P₁ C₁ P₁ᵀ (r − A z₁);  M⁻¹r = z₂ + S₀ (r − A z₂)
C_k  : the same on A_k with S_k, P_{k+1}, C_{k+1};  C_K = A_K⁻¹ (block-tridiagonal LU)
A_k  = E S_k − H_k − Y_kᵀ Σ_L Y_k − Y_kᵀ Σ_R Y_k,  S_k = U_kᵀ S_{k−1} U_k,  Y_k = rows of P₁⋯P_k on a lead face
```

- **Smoothers** (S₀ on the fine level, S_k on level k): restricted additive Schwarz. Each window matrix is closed by an
  absorbing CAP on its artificial cuts and by Σ on the lead faces; only the core rows of its inverse are kept.
- **Bases U_k**: eigenvectors of local density matrices ρ = −(1/π) Im Σ_q w_q [M(z_q)⁻¹], integrated over a complex
  contour that covers the energy window.
  - Level 1: per fine window.
  - Level k+1: per merged pair of level-k blocks (`--up y:CUT,x:CUT,…`).
- **Outer solver**: FGMRES from x₀ = 0, stopping at `--tol`. T = Σ_cols yᴴ Γ_R y on the right face.
- **Lead OBC**: Σ and the injection are computed once, in setup, for all points.
  - Beyn on the periodized lead (default, as in QuaTrEx). The injection is the incoming modes.
  - The folded lead cell, with Sancho decimation (`--obc folded`). The injection is the eigenchannels of Γ_L.

## Files

| file | content |
|---|---|
| `src/qtxmg/single.py` | the single-GPU solver in one file: device, OBC, fine windows, hierarchy, V-cycle, FGMRES, CLI |
| `src/qtxmg/hier/` | the multi-GPU solver `qtx_ms.hier` of the development code, as it is (one MPI rank per GPU; one rank without `--dist`): `run` driver and CLI, `plan` levels, windows and who needs which block, `levels` bases, Galerkin blocks, smoothers, V-cycle, bottom LU, `fine` fine operator and windows, `obc` lead Σ over the ranks, `slow` slow-mode coarse level |
| `src/qtxmg/dist.py`, `schwarz.py`, `msbasis.py`, `mlevel.py`, `devstore.py`, `abtb.py` | the parts of the development modules that `hier` uses (communicators, halos, domains; CAP-Schwarz windows and block FGMRES; device; launcher; sliced device store; Sancho) |
| `src/qtxmg/obc_beyn.py` | Beyn lead self-energy / injection (periodized lead, Woodbury face fixed point), for both solvers |
| `src/qtxmg/gpu.py` | numpy / cupy helpers |
| `src/qtxmg/prep/inputs.py` | input set: read / write (`device.xyz`, `norb.txt`, `blocks.txt`, `H.npz`, `S.npz`, `lead_*.npz`, `meta.json`); also reads the prepared folders of `qtx_modespace.prep` (`prep.json`, upper-triangle H / S: faster for large devices) |
| `src/qtxmg/prep/quatrex.py` | QuaTrEx / deviceBuilder device folder → input set |
| `tests/` | unit tests (pytest), `nccl_probe.py` (NCCL over real ranks), `regression/` (daint jobs, references, compare.py) |
| `scripts/daint/` | daint.alps: environment (`setup.sh`, `activate.sh`, `requirements.txt`; MPI / NCCL: `setup_mpi.sh`, `multinode_env.sh`, `rank.sh`), jobs (`run.sbatch`, `run_multi.sbatch`, `prep.sbatch`) |

## Install and run

```
pip install -e .[prep,gpu13]                        # or [gpu] for a CUDA 12 driver; CPU only: pip install -e .[prep]
qtxmg-prep-quatrex /path/to/device_MoS2HfO2_2808a_29088o --out inputs/mos2_hfo2
qtxmg --inputs inputs/mos2_hfo2 --dE 0.05,0.15,0.30 --cap-split 2,1 --up y:1e-5,x:1e-5 --mu=-0.15,0.6 --gpu \
      --cache /scratch/$USER/qtxmg/mos2_hfo2 --out T.json
```

Main options:
- `--up AXIS:CUT,…`: merges and their basis cuts.
- `--cut`: cut of level 1.
- `--cap WIDTH,S,D`: fine windows [Å], CAP strength [eV], ramp width [Å].
- `--cap-split`: strips per periodic transverse axis.
- `--mu M1,M2[,NPTS[,W]]`: contour, relative to Ec.
- `--obc beyn|folded`.
- `--tol`, `--maxit`.
- `--cache DIR`: keeps Σ, the injection and the bases, so later runs skip the density matrices.

## daint.alps (GH200)

From the release folder:

```
srun -A <account> -p debug -N1 -t 20 bash scripts/daint/setup.sh                  # once: .venv-daint with cupy (CUDA 13)
SRC=/path/to/quatrex_device DST=$SCRATCH/inputs/dev sbatch -A <account> scripts/daint/prep.sbatch
INPUTS=$SCRATCH/inputs/dev DE=0.05,0.15,0.30 SPLIT=2,1 sbatch -A <account> scripts/daint/run.sbatch
```

`run.sbatch` takes its options from environment variables, with defaults in brackets:
- `INPUTS`
- `DE`
- `UP` [`y:1e-5,x:1e-5`]
- `SPLIT`
- `MU` [`-0.15,0.6`]
- `CUT` [`1e-5`]
- `CAP` [`16,2,2`]
- `OBC` [`beyn`]
- `MAXIT` [`100`]
- `CACHE` [`$SCRATCH/qtxmg_cache/<inputs>`]
- `OUT`
- `EXTRA`: any further solver options

The log and the JSON with T, iterations and residuals per energy are written to `qtxmg_<job>.{log,json}`.

Reference: 2808a MoS2/HfO2 device (22 848 orbitals) on one GH200, 11 energies at Ec … Ec + 0.50, cuts 1e-5,
N = 7335 / 4283 / 1778 per level:

| `--obc` | setup | per energy | iterations | T |
|---|---|---|---|---|
| `beyn` (default) | 104 s (Σ at 35 points 78 s) | 1.5–2.3 s | 13..25 | equal to the distributed development solver to ≤ 7e-12 |
| `folded` | 70 s | 1.7–2.6 s | 11..25 | exact T (folded lead) to 1.3e-6 |

GPU memory: 5–6 GB. Beyn periodizes the lead from its outermost primitive cell (as QuaTrEx); on this device the lead cell
is not exactly periodic, so Beyn and folded are different lead models: T differs by about 1e-3.

## Potential changes on fixed bases (`--dV`)

`--dV NEW.npy[,REF.npy]` (both solvers) adds a potential change, per atom or per orbital in the source order, to H as
(dV S + S dV)/2.
- With the default `--dV-bases reuse`, the bases and Σ of the inputs are reused from `--cache` (run once without `--dV`
  first). The Galerkin blocks project the change into the bases.
- dV must be 0 on the left lead face.
- On the right face it must be a constant c (another drain bias): the right lead moves with it, and Σ_R(z) = Σ_R(z − c) of
  the given lead.
- `--dV-bases build` builds new bases of H + dV in an empty cache folder, as a reference.

A potential that is part of the device itself belongs in the inputs: `qtxmg-prep-quatrex --potential`.

## Several GPUs

`qtxmg-multi` (`python -m qtxmg.hier.run`) is `qtx_ms.hier` of the development code. It runs one MPI rank per GPU
(`--dist nccl|mpi`). Its options are those of `qtxmg`, except: `--obc` defaults to `folded` (also `beyn`, `beyn-real`,
`closed`), Σ is cached in `--sigma-cache` and the bases in `--cache`, and `--schwarz`, `--slow`, `--bottom`, `--dev-store`
are extra:

```
pip install -e .[prep,gpu13,mpi,nccl13]
mpiexec -n 4 qtxmg-multi --inputs inputs/mos2_hfo2 --dE 0.05,0.15 --cap-split 2,1 --up y:1e-5,x:1e-5 --obc beyn --gpu --dist nccl \
    --cache $SCRATCH/qtxmg/dev --sigma-cache $SCRATCH/qtxmg/dev/sigma
```

Distribution:
- Level-1 blocks (fine cores) go to ranks in contiguous chunks of x positions of equal window cost. A merged block belongs
  to the owner of its first kid, and a window to the owner of its parents. Every rank holds the rows of its own blocks of
  every vector.
- The plan (`hier/plan.py`) lists once, from the structure, which bases and block pairs each rank needs (neighbours, parents,
  lead blocks, CAP windows); each block pair is computed once, and the bases and pairs go only to the ranks that need them,
  point to point (NCCL between the GPUs, or MPI in chunks below 2 GB).
- Σ and the injection are computed once, with the points dealt out to the ranks, and stored in the Σ cache. The caches must be
  on a file system that all ranks see.
- A_K⁻¹ uses a block-tridiagonal LU over domains of slabs separated by single slabs, one domain per rank. The separator
  system is factored on one rank.
- Dot products are summed in one message per Gram–Schmidt pass. A sum gives the same bits on every rank, so all ranks
  take the same branches.

On daint, `bash scripts/daint/setup_mpi.sh` (on a compute node, after `setup.sh`) adds mpi4py, MPICH and NCCL. Then:

```
INPUTS=$SCRATCH/inputs/dev DE=0.05,0.15 SPLIT=2,1 sbatch -A <account> scripts/daint/run_multi.sbatch                    # 4 GPUs of one node
INPUTS=... sbatch -A <account> --nodes=2 --uenv=prgenv-gnu/24.11:v1 scripts/daint/run_multi.sbatch                       # several nodes
```

`run_multi.sbatch` takes the variables of `run.sbatch` plus:
- `GPUS`: ranks per node [`4`]
- `DIST`: `nccl` or `mpi` [`nccl`]

Several nodes use cray-mpich of the uenv under mpi4py, and NCCL over Slingshot (`multinode_env.sh`).

## Tests

```
python -m pytest tests                                   # unit tests, CPU, thread ranks on a synthetic strip
srun -n 8 python tests/nccl_probe.py                     # NCCL init, sums and exchanges over real ranks (GPUs)
```

- **Unit tests** (`tests/test_hier.py`, `tests/test_dist.py`: the tests of `qtx_ms.hier` and of the parts of `qtx_ms.dist` it
  uses, from the development repo). The reference is the serial `hier` run, checked against the dense Caroli T; the
  development repo also compares it with `qtx_ms.mlsolve`, which is not in the release.
  - thread ranks (2, 3): the same T, N and iterations, every block pair and every Σ point once, the cache
  - injection, contour slices, the bottom LU and its domains, Schwarz only, slow modes, Beyn (also on ranks), closed
    boundaries, lead blocks, GPU (skipped without one)
  - `--dV` (bases reused, source order, drain bias)
  - ownership chunks, halos, slab domains, sharing of the contour points, LU host views
- **daint regressions** (`tests/regression/regress.sbatch`): 2808a, 9072a, 21600a, 6 nm (4 GPUs each) and 12 nm (2 nodes,
  8 GPUs), with the options of the references in `tests/regression/ref/`. 9072a and 21600a compare with the exact T only
  (tolerance 1e-7; N and iterations not checked). The job ends with the table and PASS / FAIL from
  `tests/regression/compare.py` (T to 1e-8 relative, N equal, iterations at most 2 more). Install pytest with `setup_mpi.sh`.
