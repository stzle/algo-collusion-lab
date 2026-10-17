"""
Rung 2 — the Dou-Goldstein-Ji trading laboratory at laptop scale, symmetric
information (both speculators see v exactly). DGJ's two baseline environments:
low noise trading (sigma_u = 0.1) and high noise trading (sigma_u = 100),
with xi = 500, I = 2, rho = 0.95, theta = 0.1.

    python scripts/02_dgj_baseline.py                 # 30 sessions per environment
    python scripts/02_dgj_baseline.py --sessions 10   # quick look

Outputs in results/dgj_baseline/summary.json
"""
import argparse

import numpy as np

import _common as C
import _dgj_common as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-periods", type=int, default=G.LAPTOP["max_periods"])
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override any DGJParams field, e.g. --set alpha=0.05 n_p=37")
    args = ap.parse_args()
    threads = C.init_threads(args.threads)
    out = C.out_dir("dgj_baseline" + G.tag(args.set))
    seeds = np.arange(1, args.sessions + 1)

    results = {"machine": C.machine_info(threads), "points": {}}
    for su in (0.1, 100.0):
        prm = G.make_params(sigma_u=su, max_periods=args.max_periods, **G.parse_set(args.set))
        s, _ = G.run_point(prm, seeds)
        results["points"][str(su)] = s
        G.print_point(f"sigma_u = {su:g}", s)
        b = s["benchmarks"]
        print(f"{'':<34} benchmarks: chi_N = {b['chi_nash'][0]:.2f}, chi_M = {b['chi_cartel'][0]:.2f}, "
              f"profit_N = {b['profit_nash'][0]:.2f}, profit_M = {b['profit_cartel'][0]:.2f}")
    C.save_json(out / "summary.json", results)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
