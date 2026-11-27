"""Compare a qtmg run (its --out JSON) with a reference of ref/: N per level, T and the iteration range per energy.
    python tests/regression/compare.py ref/2808a.json qtxmg_123.json [--tol 1e-8] [--its 2]
Exit code 1 when an energy is missing or not converged, N differs, |T / T_ref - 1| > tol, or the largest iteration count exceeds the
reference's by more than --its. A reference without N or iterations (exact T only) checks T; its "tol" replaces the default."""
import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref")
    p.add_argument("run")
    p.add_argument("--tol", type=float, default=None, help="relative T tolerance (absolute where T_ref = 0) [the reference's tol, or 1e-8]")
    p.add_argument("--its", type=int, default=2, help="allowed excess of the largest iteration count over the reference (fewer: fine)")
    a = p.parse_args(argv)
    ref, run = json.load(open(a.ref)), json.load(open(a.run))
    tol = a.tol if a.tol is not None else ref.get("tol", 1e-8)
    ok = "N" not in ref or run["N"] == ref["N"]
    print(f"{ref['device']}: reference {ref['source']}; T tolerance {tol:g}")
    print(f"N run {run['N']}, reference {ref.get('N', '-')}" + ("" if ok else "  ** N DIFFERS **"))
    print(f"{'dE':>7} {'T':>15} {'T ref':>15} {'rel dT':>9} {'its':>8} {'ref its':>8} {'t [s]':>6} {'ref t':>6}")
    for k, v in ref["E"].items():
        w = run["E"].get(k)
        if w is None:
            print(f"{k:>7} missing  **")
            ok = False
            continue
        d = abs(w["T"] / v["T"] - 1) if v["T"] else abs(w["T"])
        its = [min(w["its"] or [0]), max(w["its"] or [0])]
        rits = v.get("its", [0, its[1]])
        good = w["converged"] and d <= tol and its[1] - rits[1] <= a.its
        ok &= good
        print(f"{k:>7} {w['T']:15.10f} {v['T']:15.10f} {d:9.1e} {its[0]:>3}..{its[1]:<4} {rits[0]:>3}..{rits[1]:<4} "
              f"{w.get('t_energy', 0):6.1f} {v.get('t_energy') or 0:6.1f}" + ("" if good else "  **"))
    print("setup [s]:", {k: round(x, 1) for k, x in run.get("t_setup", {}).items()})
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
