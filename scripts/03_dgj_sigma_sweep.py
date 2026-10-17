"""
Profit gain across noise-trading risk — the laptop analogue of DGJ Figure 2,
Panel A (they find a U-shape in log sigma_u: price-trigger collusion at low
noise, over-pruning collusion at high noise). DGJ's x-axis is the natural log
of sigma_u from -5 to 5.

    python scripts/03_dgj_sigma_sweep.py                  # 6 points x 30 sessions
    python scripts/03_dgj_sigma_sweep.py --sessions 10

Outputs in results/dgj_sigma_sweep/: summary.json, delta_vs_sigma.png
"""
import argparse

import numpy as np

import _common as C
import _dgj_common as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--log-sigma", type=float, nargs="+", default=[-5, -3, -1, 1, 3, 5])
    ap.add_argument("--max-periods", type=int, default=G.LAPTOP["max_periods"])
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override any DGJParams field, e.g. --set alpha=0.05 n_p=37")
    ap.add_argument("--plot-only", action="store_true", help="redraw the chart from summary.json")
    args = ap.parse_args()
    out = C.out_dir("dgj_sigma_sweep" + G.tag(args.set))
    if args.plot_only:
        plot(out)
        return
    threads = C.init_threads(args.threads)
    seeds = np.arange(1, args.sessions + 1)

    results = {"machine": C.machine_info(threads), "points": {}}
    rows = []
    for ls in args.log_sigma:
        prm = G.make_params(sigma_u=float(np.exp(ls)), max_periods=args.max_periods,
                            **G.parse_set(args.set))
        s, res = G.run_point(prm, seeds)
        s["log_sigma_u"] = ls
        s["delta_sessions"] = res["delta"]
        s["converged_sessions"] = res["converged"]
        results["points"][str(ls)] = s
        G.print_point(f"ln sigma_u = {ls:+.1f} (sigma_u = {np.exp(ls):.3g})", s)
    C.save_json(out / "summary.json", results)
    plot(out)
    print(f"Results in {out}")


def plot(out):
    import json
    pts = json.loads((out / "summary.json").read_text())["points"].values()
    rows = []
    for p in pts:
        d = np.array(p["delta_sessions"])
        rows.append((p["log_sigma_u"], d.mean(), np.percentile(d, 25), np.percentile(d, 75),
                     d.min(), d.max(), p["converged_share"]))
    rows = np.array(sorted(rows))
    n_sess = len(d)
    plt = C.plot_style()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.fill_between(rows[:, 0], rows[:, 4], rows[:, 5], color=C.SERIES[0], alpha=0.12, lw=0,
                    label="min–max across sessions")
    ax.fill_between(rows[:, 0], rows[:, 2], rows[:, 3], color=C.SERIES[0], alpha=0.28, lw=0,
                    label="interquartile range")
    ax.plot(rows[:, 0], rows[:, 1], color=C.SERIES[0], marker="o", ms=6, label="mean Δ")
    ax.axhline(0, color=C.INK_2, lw=1, ls=":")
    ax.axhline(1, color=C.INK_2, lw=1, ls=":")
    ax.text(rows[-1, 0], 0.02, "Nash", color=C.INK_2, ha="right", va="bottom")
    ax.text(rows[-1, 0], 0.98, "Cartel", color=C.INK_2, ha="right", va="top")
    ax.set_ylabel("Profit gain Δ")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"Collusion vs noise trading risk ({n_sess} sessions per point)")
    ax.legend(loc="upper left", fontsize=9)
    ax2.bar(rows[:, 0], 100 * rows[:, 6], width=0.8, color=C.SERIES[2])
    for xv, cv in zip(rows[:, 0], rows[:, 6]):
        ax2.text(xv, 100 * cv + 4, f"{100 * cv:.0f}%", ha="center", va="bottom",
                 color=C.INK_2, fontsize=9)
    ax2.set_ylabel("% converged")
    ax2.set_ylim(0, 135)
    ax2.set_yticks([0, 50, 100])
    ax2.set_xlabel("ln σᵤ  (noise trading risk)")
    fig.tight_layout()
    fig.savefig(out / "delta_vs_sigma.png", dpi=160)


if __name__ == "__main__":
    main()
