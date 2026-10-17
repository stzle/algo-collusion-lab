"""
Rung 3 pilot — the research question: does heterogeneous signal precision make
algorithmic collusion among informed speculators more fragile or more robust?

Speculator i observes the grid point nearest to v + eps_i, eps_i ~ N(0, q_i^2).
Two treatments, each in both monitoring variants (A and B):
  symmetric-noisy  (q, q)  — both signals degrade; benchmarks stay symmetric
  asymmetric       (0, q)  — speculator 1 sees v exactly, speculator 2 does not
  Variant A ("realized"):   the state contains the realised past value v_{t-1}
                            -> only the trading channel is affected
  Variant B ("own_signal"): the state contains the agent's own past signal
                            -> trading and monitoring channels are both affected
Benchmarks (Nash, cartel) are recomputed for every precision profile; Delta is
measured on joint profits (see TECHNICAL.md section 5).

THIS IS A PILOT. Treat the numbers as a first look, not results.

    python scripts/05_heterogeneity_pilot.py                       # both sigma_u, 20 sessions
    python scripts/05_heterogeneity_pilot.py --sigma-u 0.1 --sessions 10

Outputs in results/heterogeneity_pilot/: summary.json, delta_vs_signal_noise.png
"""
import argparse

import numpy as np

import _common as C
import _dgj_common as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, nargs="+", default=[0.1, 100.0])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--max-periods", type=int, default=G.LAPTOP["max_periods"])
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override any DGJParams field, e.g. --set alpha=0.05")
    ap.add_argument("--plot-only", action="store_true", help="redraw the chart from summary.json")
    args = ap.parse_args()
    out = C.out_dir("heterogeneity_pilot" + G.tag(args.set))
    if args.plot_only:
        plot(out)
        return
    threads = C.init_threads(args.threads)
    seeds = np.arange(1, args.sessions + 1)

    results = {"machine": C.machine_info(threads), "points": []}
    for su in args.sigma_u:
        extra = G.parse_set(args.set)
        base = G.make_params(sigma_u=su, max_periods=args.max_periods, **extra)
        s, _ = G.run_point(base, seeds)
        G.print_point(f"sigma_u={su:g}  baseline (0, 0)", s)
        results["points"].append({"sigma_u": su, "treatment": "baseline", "variant": "-",
                                  "noise": 0.0, **s})
        for treat in ("symmetric", "asymmetric"):
            for variant in ("realized", "own_signal"):
                for q in args.noise:
                    eps = (q, q) if treat == "symmetric" else (0.0, q)
                    prm = G.make_params(sigma_u=su, sigma_eps=eps, monitor=variant,
                                        max_periods=args.max_periods, **extra)
                    s2, _ = G.run_point(prm, seeds)
                    tag = "A" if variant == "realized" else "B"
                    G.print_point(f"sigma_u={su:g}  {treat[:4]} {tag}  eps={eps}", s2)
                    rel = s2["profit_mean"] / np.asarray(s2["benchmarks"]["profit_nash"])
                    print(f"{'':<34} profit / own Nash profit per agent: {np.round(rel, 3)}"
                          f"   static deviation gain: {np.round(100 * s2['static_deviation_gain'], 1)}%"
                          f"   cartel chi: {np.round(s2['benchmarks']['chi_cartel'], 1)}")
                    results["points"].append({"sigma_u": su, "treatment": treat, "variant": variant,
                                              "noise": q, "profit_vs_own_nash": rel, **s2})
    C.save_json(out / "summary.json", results)

    plot(out)
    print(f"Results in {out}")


def _restraint(p):
    """Restraint per agent from a summary point; recomputed from the stored
    aggregates for summary files written before this field existed."""
    if "restraint_vs_best_response" in p:
        return np.array(p["restraint_vs_best_response"])
    from lab import dgj_benchmarks as bmk
    prm = p["params"]
    vg = bmk.value_grid(prm["n_v"], prm["v_bar"], prm["sigma_v"])
    b = bmk.benchmarks(vg, prm["sigma_eps"], prm["xi"], prm["theta"], prm["sigma_u"], prm["v_bar"])
    chi = np.array(p["aggressiveness_vs_nash"]) * b["nash"]["chi"]
    return bmk.restraint(chi, b["C"], b["S"], b["nash"]["lam"])["restraint"]


def plot(out):
    """Row 1: Delta per treatment and variant. Row 2: each speculator's restraint
    relative to its best response (1 = no restraint). Row 3: in the asymmetric
    treatment, each speculator's profit relative to its own Nash profit."""
    import json
    res = json.loads((out / "summary.json").read_text())
    pts = res["points"]
    sigmas = sorted({p["sigma_u"] for p in pts})
    n_sess = pts[0]["sessions"]
    plt = C.plot_style()
    n = len(sigmas)
    fig, axes = plt.subplots(3, n, figsize=(max(6.4 * n, 7.2), 11), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 2.4, 2]})
    style = {("symmetric", "realized"): (C.SERIES[0], "-", "symmetric (q, q), Variant A"),
             ("symmetric", "own_signal"): (C.SERIES[0], "--", "symmetric (q, q), Variant B"),
             ("asymmetric", "realized"): (C.SERIES[1], "-", "asymmetric (0, q), Variant A"),
             ("asymmetric", "own_signal"): (C.SERIES[1], "--", "asymmetric (0, q), Variant B")}
    for j, su in enumerate(sigmas):
        base = [p for p in pts if p["sigma_u"] == su and p["treatment"] == "baseline"][0]
        rb = float(np.mean(_restraint(base)))
        ax = axes[0, j]
        for (treat, variant), (col, ls, lab) in style.items():
            c = [(0.0, base["delta_all"]["mean"], base["delta_all"]["se"])]
            c += [(p["noise"], p["delta_all"]["mean"], p["delta_all"]["se"]) for p in pts
                  if p["sigma_u"] == su and p["treatment"] == treat and p["variant"] == variant]
            c = np.array(sorted(c))
            ax.errorbar(c[:, 0], c[:, 1], yerr=1.96 * c[:, 2], color=col, ls=ls, marker="o",
                        ms=5, capsize=3, label=lab)
        ax.axhline(0, color=C.INK_2, lw=1, ls=":")
        ax.set_title(f"Profit gain Δ, σᵤ = {su:g}")
        ax.set_ylabel("Δ (joint, own benchmarks)")

        # row 2: who holds back
        ax = axes[1, j]
        for variant, ls in (("realized", "-"), ("own_signal", "--")):
            tag = "A" if variant == "realized" else "B"
            sym = sorted((p["noise"], float(np.mean(_restraint(p)))) for p in pts
                         if p["sigma_u"] == su and p["treatment"] == "symmetric" and p["variant"] == variant)
            sym = np.array([(0.0, rb)] + sym)
            ax.plot(sym[:, 0], sym[:, 1], color=C.SERIES[0], ls=ls, marker="o", ms=5,
                    label=f"symmetric, both speculators, Variant {tag}")
            asy = np.array(sorted((p["noise"], *_restraint(p)) for p in pts
                                  if p["sigma_u"] == su and p["treatment"] == "asymmetric" and p["variant"] == variant))
            asy = np.vstack([[0.0, rb, rb], asy])
            ax.plot(asy[:, 0], asy[:, 1], color=C.SERIES[2], ls=ls, marker="o", ms=5,
                    label=f"asymmetric, informed (q = 0), Variant {tag}")
            ax.plot(asy[:, 0], asy[:, 2], color=C.SERIES[3], ls=ls, marker="s", ms=5,
                    label=f"asymmetric, noisy (q), Variant {tag}")
        ax.axhline(1, color=C.INK_2, lw=1)
        ax.text(ax.get_xlim()[0], 1.0, "  best response (no restraint)", color=C.INK_2, ha="left",
                va="bottom", fontsize=9)
        ax.set_title("Who holds back: aggressiveness ÷ own best response")
        ax.set_ylabel("restraint (1 = none)")

        # row 3: who gains under asymmetry
        ax = axes[2, j]
        for variant, ls in (("realized", "-"), ("own_signal", "--")):
            c = sorted((p["noise"], p["profit_vs_own_nash"][0], p["profit_vs_own_nash"][1]) for p in pts
                       if p["sigma_u"] == su and p["treatment"] == "asymmetric" and p["variant"] == variant)
            c = np.array(c)
            tag = "A" if variant == "realized" else "B"
            ax.plot(c[:, 0], 100 * (c[:, 1] - 1), color=C.SERIES[2], ls=ls, marker="o", ms=5,
                    label=f"informed (q = 0), Variant {tag}")
            ax.plot(c[:, 0], 100 * (c[:, 2] - 1), color=C.SERIES[3], ls=ls, marker="s", ms=5,
                    label=f"noisy (q), Variant {tag}")
        ax.axhline(0, color=C.INK_2, lw=1)
        ax.set_title("Asymmetric treatment: profit vs own Nash profit")
        ax.set_ylabel("% above own Nash profit")
        ax.set_xlabel("Signal noise q (std. dev. of εᵢ)")
    axes[0, 0].legend(fontsize=9, loc="best")
    axes[1, 0].legend(fontsize=7.5, loc="lower left", ncol=2)
    axes[2, 0].legend(fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.36), ncol=2)
    fig.suptitle(f"PILOT: heterogeneous signal precision ({n_sess} sessions per point; "
                 f"bars = 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "delta_vs_signal_noise.png", dpi=160)


if __name__ == "__main__":
    main()
