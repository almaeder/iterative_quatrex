#!/bin/bash
# Potential sweep on fixed bases (--dV): the 12 nm OFF-state device (barrier Vb 0.2 eV, drain Vds 0.4 V) is solved once, which
# builds its bases; then every case of ref/12nm_off_sweep.json (gate Vb 0.1, 0.3, 0.0; drain Vds 0.1, 0.2, 0.3, 0.5 at Vb 0.2) is
# solved with dV = V(case) - V(base) on those bases (--dV-bases reuse; drain cases: Sigma_R at z - c). T and the iterations are
# compared with the development solver (qtx_ms.hier.run) at Ec + 0.10, 0.25, 0.50 (compare.py; T to 1e-8 relative or 1e-12).
#   INPUTS   prep folder of the base device                  STORE   its sliced device store (optional, --dev-store)
#   POTS     folder of <dV>/inputs/potential.npy of every case (dV of the reference) and of the base (<ref_potential>)
#   SIGMA    lead Sigma folder (default $CACHE/sigma)        CACHE   folder of the bases (a later job resumes from it)
#   CASES    subset of the cases (default: all)             DE      energies (default: those of the reference)
#   INPUTS=... POTS=... CACHE=... sbatch -A <account> --nodes=6 --uenv=prgenv-gnu/24.11:v1 tests/regression/sweep.sbatch
#SBATCH --job-name=qtxmg-sweep
#SBATCH --partition=debug
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=qtxmg-sweep-%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
: "${INPUTS:?INPUTS=prep folder}" "${POTS:?POTS=folder of the potentials}" "${CACHE:?CACHE=folder of the bases}"
REF=tests/regression/ref/12nm_off_sweep.json
source scripts/daint/activate.sh
eval "$(python - $REF <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); a = r["args"]
print(f"DE0={','.join(r['cases'][0]['E'])} UP={a['up']} SPLIT={a['cap_split']} MU={a['mu']} CUT={a['cut']} CAP={a['cap']} OBC={r['obc']} "
      f"MAXIT={a['maxit']} PREF={r['ref_potential']} ALL='{' '.join(c['name'] for c in r['cases'])}'")
print(f"REFX='--Ec {r['Ec']!r} --npts {a['npts']} --eta {a['eta']} --cap-spacing {a['cap_spacing']} --cap-tov {a['cap_tov']} "
      f"--halo {a['halo']} --tol {a['tol']} --chunk {a['chunk']}'")
PY
)"
export DE=${DE:-$DE0} UP SPLIT MU CUT CAP OBC MAXIT INPUTS CACHE SIGMA=${SIGMA:-$CACHE/sigma}
O=${OUT:-$PWD/qtxmg_sweep_$SLURM_JOB_ID}
echo "sweep 12nm_off: $INPUTS, potentials $POTS, bases $CACHE, Sigma $SIGMA, energies $DE"
rc=0
for c in ${CASES:-$ALL}; do
  x="$REFX ${STORE:+--dev-store $STORE}"
  d=$(python -c "import json, sys; print(next(c['dV'] or '' for c in json.load(open(sys.argv[1]))['cases'] if c['name'] == sys.argv[2]))" $REF $c)
  [ -z "$d" ] || x="$x --dV $POTS/$d/inputs/potential.npy,$POTS/$PREF/inputs/potential.npy --dV-bases reuse"
  echo "=== $c"
  OUT=${O}_$c EXTRA="$x" bash scripts/daint/run_multi.sbatch > ${O}_$c.out 2>&1 || { rc=1; tail -5 ${O}_$c.out; continue; }
  python - $REF $c ${O}_$c.json <<'PY' || rc=1
import json, sys
r, name, run = json.load(open(sys.argv[1])), sys.argv[2], json.load(open(sys.argv[3]))
c = next(c for c in r["cases"] if c["name"] == name)
ok = run["N"] == r["N"]
print(f"{name}: N {run['N']}" + ("" if ok else f"  ** reference {r['N']} **") + f"; setup {run['t_setup']}")
for k, v in c["E"].items():
    w = run["E"].get(k)
    if w is None:
        print(f"  {k} missing **"); ok = False; continue
    d = abs(w["T"] - v["T"])
    its = [min(w["its"] or [0]), max(w["its"] or [0])]
    good = w["converged"] and (d <= 1e-8 * abs(v["T"]) or d <= 1e-12) and its[1] - v["its"][1] <= 2
    ok &= good
    tb = c.get("T_build", {}).get(k)
    print(f"  {k} T {w['T']:.10e} ref {v['T']:.10e} dT {d:.1e}" + (f" (new bases {tb:.10e})" if tb is not None else "")
          + f"  its {its[0]}..{its[1]} ref {v['its'][0]}..{v['its'][1]}  {w.get('t_energy', 0):.0f} s" + ("" if good else "  **"))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
done
echo "sweep: $([ $rc = 0 ] && echo PASS || echo FAIL) after $SECONDS s"
exit $rc
