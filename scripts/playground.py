"""
Playground: one setting, both speculators learning and one learning alone, in about a
minute. For practising and for answering "change a parameter now" at the stand.
Not part of any pre-registered experiment.

    python scripts/playground.py                         # exact, equal information
    python scripts/playground.py --q2 1                  # speculator 2 sees the value with noise 1
    python scripts/playground.py --q1 0.5 --q2 0.5 --sigma-u 100
    python scripts/playground.py --price-grid global     # the coarse price memory
    python scripts/playground.py --alpha 0.02 --sessions 8

Prints Delta, each speculator's aggressiveness (x Nash), restraint (x best response),
the collusive part and the learning bias.
"""
import argparse

import numpy as np

import _common as C
import _dgj_common as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1", type=float, default=0.0, help="signal noise of speculator 1")
    ap.add_argument("--q2", type=float, default=0.0, help="signal noise of speculator 2")
    ap.add_argument("--sigma-u", type=float, default=0.1, help="noise trading (0.1 low, 100 high)")
    ap.add_argument("--price-grid", default="per_value", choices=["per_value", "global"])
    ap.add_argument("--grid-width", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.01, help="learning rate")
    ap.add_argument("--beta", type=float, default=5e-7, help="exploration decay")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--max-periods", type=float, default=2e8)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    C.init_threads(a.threads)
    seeds = np.arange(1, a.sessions + 1)
    common = dict(sigma_u=a.sigma_u, sigma_eps=(a.q1, a.q2), grid="nash", grid_width=a.grid_width,
                  price_grid=a.price_grid, alpha=a.alpha, beta=a.beta, max_periods=int(a.max_periods))
    pair, _ = G.run_point(G.make_params(fixed=(None, None), **common), seeds)
    G.print_point("both learn", pair)
    alone = {}
    for i in (0, 1) if a.q1 != a.q2 else (0,):
        fixed = (None, 1.0) if i == 0 else (1.0, None)
        alone[i], _ = G.run_point(G.make_params(fixed=fixed, **common), seeds)
        G.print_point(f"speculator {i + 1} learns alone", alone[i])
    print()
    for i in (0, 1):
        al = alone.get(i, alone[0])
        j = i if i in alone else 0
        r_pair = pair["restraint_vs_best_response"][i]
        r_alone = al["restraint_vs_best_response"][j]
        print(f"speculator {i + 1}: collusive part {100 * (r_alone - r_pair):+5.1f} %   "
              f"learning bias {100 * (1 - r_alone):+5.1f} %")


if __name__ == "__main__":
    main()
