"""
Rung 1 — validation against a published number.

Replicates the baseline of Calvano, Calzolari, Denicolò & Pastorello (2020, AER):
two Q-learning firms, logit demand, 15 prices, alpha = 0.15, beta = 4e-6,
delta = 0.95, convergence = strategies unchanged for 100,000 periods,
1,000 independent sessions. Published result: average profit gain Delta ~ 0.85.

    python scripts/01_calvano_baseline.py               # full: 1000 sessions
    python scripts/01_calvano_baseline.py --sessions 100

Outputs in results/calvano/: summary.json, delta_distribution.png, deviation_irf.png
"""
import argparse

import numpy as np

import _common as C
from lab import calvano as cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--grid", choices=["aer", "wp"], default="aer")
    ap.add_argument("--seed0", type=int, default=1)
    args = ap.parse_args()
    threads = C.init_threads(args.threads)
    out = C.out_dir("calvano")

    prm = cv.CalvanoParams(grid=args.grid)
    bm = cv.benchmarks(prm)
    prices = cv.price_grid(prm)
    PI = cv.profit_matrix(prices, prm)
    Q0 = cv.initial_Q(PI, prm.delta)
    seeds = np.arange(args.seed0, args.seed0 + args.sessions, dtype=np.int64)

    print(f"Nash price {bm['p_nash']:.4f}, monopoly price {bm['p_monopoly']:.4f}")
    print(f"Running {args.sessions} sessions on {threads} threads ...")
    cv.run_sessions(seeds[:2], PI, Q0, prm.alpha, prm.beta, prm.delta, 1000, 5000)  # compile
    with C.Timer() as tm:
        Qs, conv, periods, s_fin = cv.run_sessions(
            seeds, PI, Q0, prm.alpha, prm.beta, prm.delta, prm.conv_window, prm.max_periods)

    outs = [cv.session_outcome(Qs[k], s_fin[k], PI, bm) for k in range(len(seeds))]
    delta = np.array([o["delta"] for o in outs])
    cyc = np.array([o["cycle_len"] for o in outs])
    ok = conv
    # deviation experiment on sessions that settle on a constant price pair
    irf = [cv.impulse_response(Qs[k], outs[k]["cycle"][0], prices, PI)
           for k in range(len(seeds)) if ok[k] and outs[k]["cycle_len"] == 1]
    irf = np.mean(irf, axis=0) if irf else None

    summary = {
        "machine": C.machine_info(threads),
        "params": prm.to_dict(),
        "benchmarks": bm,
        "price_grid": prices,
        "sessions": int(len(seeds)),
        "converged_share": float(ok.mean()),
        "periods_to_convergence": C.summary_stats(periods[ok]),
        "delta": C.summary_stats(delta[ok]),
        "published_delta": 0.85,
        "cycle_length_shares": {int(L): float((cyc[ok] == L).mean()) for L in np.unique(cyc[ok])},
        "wall_clock_seconds": tm.seconds,
        "seconds_per_session_per_core": tm.seconds * min(threads, len(seeds)) / len(seeds),
        "irf_prices": irf,
    }
    C.save_json(out / "summary.json", summary)

    plt = C.plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delta[ok], bins=np.linspace(0.3, 1.05, 31), color=C.SERIES[0], rwidth=0.92)
    ax.axvline(delta[ok].mean(), color=C.INK, lw=2, label=f"mean, this run: {delta[ok].mean():.3f}")
    ax.axvline(0.85, color=C.SERIES[1], lw=2, ls=(0, (3, 3)), label="published: 0.85")
    ax.legend(loc="upper left")
    ax.set_xlabel("Profit gain Δ per session  (0 = Nash, 1 = monopoly)")
    ax.set_ylabel("Sessions")
    ax.set_title(f"Calvano et al. (2020) baseline — {ok.sum()} converged sessions")
    fig.tight_layout()
    fig.savefig(out / "delta_distribution.png", dpi=160)

    if irf is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        tau = np.arange(len(irf))
        ax.plot(tau, irf[:, 0], color=C.SERIES[0], marker="o", ms=4, label="Firm 1 (deviates at τ=1)")
        ax.plot(tau, irf[:, 1], color=C.SERIES[1], marker="o", ms=4, label="Firm 2")
        ax.axhline(bm["p_nash"], color=C.INK_2, lw=1, ls=":")
        ax.axhline(bm["p_monopoly"], color=C.INK_2, lw=1, ls=":")
        ax.text(len(irf) - 1, bm["p_nash"], "Nash", color=C.INK_2, ha="right", va="bottom")
        ax.text(len(irf) - 1, bm["p_monopoly"], "Monopoly", color=C.INK_2, ha="right", va="bottom")
        ax.set_xlabel("Period after deviation (τ)")
        ax.set_ylabel("Average price")
        ax.set_title("Deviation and punishment (Calvano et al. Fig. 5)")
        ax.legend(loc="center right")
        fig.tight_layout()
        fig.savefig(out / "deviation_irf.png", dpi=160)

    s = summary
    print(f"\nConverged: {s['converged_share']:.1%} of sessions, "
          f"median {s['periods_to_convergence']['median'] / 1e6:.2f} M periods")
    print(f"Delta = {s['delta']['mean']:.3f} (s.e. {s['delta']['se']:.3f}); published ≈ 0.85")
    print(f"Wall clock {tm.seconds:.0f} s on {threads} threads "
          f"= {s['seconds_per_session_per_core']:.2f} s per session per core")
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
