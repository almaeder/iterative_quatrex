"""Compare a qtxmg run (its --out JSON) with a reference of ref/: N per level, T and the iteration range per energy.
    python tests/regression/compare.py ref/2808a.json qtxmg_123.json [--tol 1e-8] [--its 2]
Exit code 1 when an energy is missing or not converged, N differs, |T / T_ref - 1| > tol, or the largest iteration count differs
by more than --its."""
import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref")
    p.add_argument("run")
    p.add_argument("--tol", type=float, default=1e-8, help="relative T tolerance (absolute where T_ref = 0)")
    p.add_argument("--its", type=int, default=2, help="allowed difference of the largest iteration count")
    a = p.parse_args(argv)
    ref, run = json.load(open(a.ref)), json.load(open(a.run))
    ok = run["N"] == ref["N"]
    print(f"{ref['device']}: reference {ref['source']}")
    print(f"N run {run['N']}, reference {ref['N']}" + ("" if ok else "  ** N DIFFERS **"))
    print(f"{'dE':>7} {'T':>15} {'T ref':>15} {'rel dT':>9} {'its':>8} {'ref its':>8} {'t [s]':>6} {'ref t':>6}")
    for k, v in ref["E"].items():
        w = run["E"].get(k)
        if w is None:
            print(f"{k:>7} missing  **")
            ok = False
            continue
        d = abs(w["T"] / v["T"] - 1) if v["T"] else abs(w["T"])
        its = [min(w["its"] or [0]), max(w["its"] or [0])]
        good = w["converged"] and d <= a.tol and abs(its[1] - v["its"][1]) <= a.its
        ok &= good
        print(f"{k:>7} {w['T']:15.10f} {v['T']:15.10f} {d:9.1e} {its[0]:>3}..{its[1]:<4} {v['its'][0]:>3}..{v['its'][1]:<4} "
              f"{w.get('t_energy', 0):6.1f} {v.get('t_energy') or 0:6.1f}" + ("" if good else "  **"))
    print("setup [s]:", {k: round(x, 1) for k, x in run.get("t_setup", {}).items()})
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
