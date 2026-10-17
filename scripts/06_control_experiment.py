"""
Control experiment for the one-sided restraint found in the heterogeneity pilot.

In the pilot, under asymmetric precision (speculator 1 sees v exactly, speculator 2
sees v + noise), the informed speculator traded at its best response and only the noisy
one held back. Two things could produce that pattern without any coordination:

  1. A single-agent learning bias: a Q-learner with noisy rewards may under-trade even
     when nobody reacts to it.  -> test: one learner against a FIXED opponent that always
     plays its Nash strategy ("informed only", "noisy only", "alone").
  2. The action grid: DGJ's rule builds each agent's grid between its Nash and cartel
     levels. Under asymmetry that puts the informed agent's grid at 0.97-1.33 x Nash
     (it cannot hold back) and the noisy agent's at -0.10-1.10 x Nash.
     -> test: the same runs with a Nash-centred grid, 0.7-1.3 x own Nash for everyone.

For every learning speculator the script reports its restraint = aggressiveness divided
by its best response to what the other speculator actually does (1 = no restraint).

    python scripts/06_control_experiment.py
    python scripts/06_control_experiment.py --sigma-u 0.1
    python scripts/06_control_experiment.py --plot-only

Default: both noise-trading levels (0.1 and 100), 20 sessions, 32 points. At sigma_u = 100
prices are too noisy for punishment strategies, so a lone learner that still holds back
there shows DGJ's over-pruning bias without any interaction.

Outputs in results/control/: summary.json, control.png
"""
import argparse
import json

import numpy as np

import _common as C
import _dgj_common as G

WHO = {
    # name: (fixed tuple, which agent's restraint to read)
    "pair": ((None, None), None),
    "informed_only": ((None, 1.0), 0),
    "noisy_only": ((1.0, None), 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, nargs="+", default=[0.1, 100.0])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--grid-width", type=float, default=0.3)
    ap.add_argument("--max-periods", type=int, default=G.LAPTOP["max_periods"])
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    out = C.out_dir("control")
    if args.plot_only:
        plot(out)
        return
    threads = C.init_threads(args.threads)
    seeds = np.arange(1, args.sessions + 1)

    results = {"machine": C.machine_info(threads), "grid_width": args.grid_width, "points": []}
    for su in args.sigma_u:
        for grid in ("benchmarks", "nash"):
            # symmetric perfect information: the pair, and one learner alone
            plan = [(0.0, "pair"), (0.0, "alone")]
            plan += [(q, who) for q in args.noise for who in ("pair", "informed_only", "noisy_only")]
            for q, who in plan:
                fixed = (None, 1.0) if who == "alone" else WHO[who][0]
                prm = G.make_params(sigma_u=su, sigma_eps=(0.0, q), fixed=fixed, grid=grid,
                                    grid_width=args.grid_width, max_periods=args.max_periods)
                s, _ = G.run_point(prm, seeds)
                label = f"sigma_u={su:g} {grid:<10} q={q:<4} {who:<13}"
                G.print_point(label, s)
                results["points"].append({"sigma_u": su, "grid": grid, "noise": q, "who": who, **s})
                C.save_json(out / "summary.json", results)      # save as we go
    print()
    table(results)
    plot(out)
    print(f"Results in {out}")


def _learner_restraint(points, su, grid, q, who, agent):
    """Restraint of `agent` (0 informed, 1 noisy) in the given cell, or nan."""
    for p in points:
        if p["sigma_u"] == su and p["grid"] == grid and p["noise"] == q and p["who"] == who:
            r = p["restraint_vs_best_response"]
            return float(r[agent])
    return float("nan")


def series(points, su, grid):
    """x = noise of speculator 2; four lines: informed / noisy, in a pair / alone."""
    qs = sorted({p["noise"] for p in points if p["sigma_u"] == su and p["grid"] == grid})
    out = {k: [] for k in ("inf_pair", "inf_alone", "noisy_pair", "noisy_alone")}
    for q in qs:
        if q == 0.0:
            pair = np.mean([_learner_restraint(points, su, grid, 0.0, "pair", a) for a in (0, 1)])
            alone = _learner_restraint(points, su, grid, 0.0, "alone", 0)
            for k, v in (("inf_pair", pair), ("noisy_pair", pair), ("inf_alone", alone), ("noisy_alone", alone)):
                out[k].append(v)
        else:
            out["inf_pair"].append(_learner_restraint(points, su, grid, q, "pair", 0))
            out["noisy_pair"].append(_learner_restraint(points, su, grid, q, "pair", 1))
            out["inf_alone"].append(_learner_restraint(points, su, grid, q, "informed_only", 0))
            out["noisy_alone"].append(_learner_restraint(points, su, grid, q, "noisy_only", 1))
    return qs, {k: np.array(v) for k, v in out.items()}


def table(results):
    pts = results["points"]
    print("Restraint relative to best response (1 = none). 'alone' = the other speculator is fixed at Nash.")
    print(f"{'sigma_u':>8} {'grid':<11} {'q':>4} | {'informed: pair':>14} {'alone':>6} | {'noisy: pair':>11} {'alone':>6}")
    for su in sorted({p["sigma_u"] for p in pts}):
        for grid in ("benchmarks", "nash"):
            qs, s = series(pts, su, grid)
            for k, q in enumerate(qs):
                print(f"{su:>8g} {grid:<11} {q:>4} | {s['inf_pair'][k]:>14.3f} {s['inf_alone'][k]:>6.3f} | "
                      f"{s['noisy_pair'][k]:>11.3f} {s['noisy_alone'][k]:>6.3f}")


def plot(out):
    res = json.loads((out / "summary.json").read_text())
    pts = res["points"]
    sigmas = sorted({p["sigma_u"] for p in pts})
    grids = ("benchmarks", "nash")
    titles = {"benchmarks": "DGJ action grid (between Nash and cartel)",
              "nash": f"Equal action grid ({1 - res['grid_width']:.1f}–{1 + res['grid_width']:.1f} × own Nash)"}
    n_sess = pts[0]["sessions"]
    plt = C.plot_style()
    fig, axes = plt.subplots(len(sigmas), 2, figsize=(12, 4.4 * len(sigmas)), squeeze=False,
                             sharey=True)
    for i, su in enumerate(sigmas):
        for j, grid in enumerate(grids):
            ax = axes[i, j]
            qs, s = series(pts, su, grid)
            if not qs:
                continue
            ax.plot(qs, s["inf_pair"], color=C.SERIES[2], marker="o", ms=6, label="informed, both learn")
            ax.plot(qs, s["inf_alone"], color=C.SERIES[2], ls="--", marker="o", ms=6, mfc="none",
                    label="informed, opponent fixed at Nash")
            ax.plot(qs, s["noisy_pair"], color=C.SERIES[3], marker="s", ms=6, label="noisy, both learn")
            ax.plot(qs, s["noisy_alone"], color=C.SERIES[3], ls="--", marker="s", ms=6, mfc="none",
                    label="noisy, opponent fixed at Nash")
            ax.axhline(1, color=C.INK_2, lw=1)
            ax.set_title(f"{titles[grid]}, σᵤ = {su:g}", fontsize=11)
            ax.set_xlabel("Signal noise q of speculator 2 (speculator 1 sees v exactly)")
            if j == 0:
                ax.set_ylabel("χ ÷ own best response (1 = no restraint)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Control experiment: who holds back, and does it need a learning opponent? "
                 f"({n_sess} sessions per point)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "control.png", dpi=160)


if __name__ == "__main__":
    main()
