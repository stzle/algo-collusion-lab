"""
Is it price-trigger collusion? The impulse-response test of DGJ Section 5.3 /
Figure 3, done with common random numbers.

After learning, a one-off adverse noise-trade shock hits the price at t = 3.
Shock sizes are 0.25%, 2.5%, 11.5% and 15% of the speculators' average absolute
order (DGJ's small / medium / large / ultra-large). Price-trigger strategies
show a threshold: no reaction to a small shock, a similar punishment (more
aggressive trading at t = 4) for all shocks above the threshold.

Each curve is the average difference between the shocked path and the same path
without the shock (same draws of v, signals and u), so even small effects are
measured precisely.

    python scripts/04_dgj_impulse_response.py                   # sigma_u = 0.1, 30 sessions
    python scripts/04_dgj_impulse_response.py --sigma-u 100

Outputs in results/dgj_irf_sigma<sigma>/: summary.json, irf.png
"""
import argparse

import numpy as np

import _common as C
import _dgj_common as G
from lab import dgj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, default=0.1)
    ap.add_argument("--paths", type=int, default=5000, help="paired paths per session")
    ap.add_argument("--max-periods", type=int, default=G.LAPTOP["max_periods"])
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override any DGJParams field, e.g. --set alpha=0.05 n_p=37")
    args = ap.parse_args()
    threads = C.init_threads(args.threads)
    out = C.out_dir(f"dgj_irf_sigma{args.sigma_u:g}" + G.tag(args.set))
    seeds = np.arange(1, args.sessions + 1)

    prm = G.make_params(sigma_u=args.sigma_u, max_periods=args.max_periods, **G.parse_set(args.set))
    s, res = G.run_point(prm, seeds)
    G.print_point(f"sigma_u = {args.sigma_u:g}", s)

    irfs = [dgj.session_irf(res, k, prm, n_paths=args.paths) for k in range(len(seeds))]
    # relative to each session's own average aggressiveness / price deviation
    dchi = np.array([I["dchi"].mean(axis=2) / I["base_chi"].mean() for I in irfs])   # (sess, K, H)
    dpt = np.array([I["dptil"] / I["base_ptil"].mean() for I in irfs])
    labels = ["small 0.25%", "medium 2.5%", "large 11.5%", "ultra-large 15%"]
    H = dchi.shape[2]
    t = np.arange(1, H + 1)

    summary = {"machine": C.machine_info(threads), "point": s,
               "shock_fracs": irfs[0]["shock_fracs"], "labels": labels,
               "aggressiveness_dev_mean": dchi.mean(axis=0),
               "aggressiveness_dev_se": dchi.std(axis=0, ddof=1) / np.sqrt(len(seeds)),
               "price_dev_mean": dpt.mean(axis=0),
               "price_dev_se": dpt.std(axis=0, ddof=1) / np.sqrt(len(seeds))}
    C.save_json(out / "summary.json", summary)

    plt = C.plot_style()
    colors = [C.INK_2, C.SERIES[0], C.SERIES[1], C.SERIES[2]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, lab in enumerate(labels):
        m = 100 * dpt.mean(axis=0)[k]
        axes[0].plot(t, m, color=colors[k], marker="o", ms=4, label=lab)
        m = 100 * dchi.mean(axis=0)[k]
        se = 100 * summary["aggressiveness_dev_se"][k]
        axes[1].plot(t, m, color=colors[k], marker="o", ms=4, label=lab)
        axes[1].fill_between(t, m - 1.96 * se, m + 1.96 * se, color=colors[k], alpha=0.12, lw=0)
    axes[0].set_title("Price deviation (% of mean)")
    axes[1].set_title("Speculators' aggressiveness (% of mean)")
    for ax in axes:
        ax.axvline(3, color=C.INK_2, lw=1, ls=":")
        ax.axhline(0, color=C.INK_2, lw=0.8)
        ax.set_xlabel("Period (shock at t = 3)")
        ax.set_xticks(t)
    axes[1].legend(title="Shock size", fontsize=9)
    fig.suptitle(f"Impulse responses, σᵤ = {args.sigma_u:g}, {len(seeds)} sessions "
                 f"× {args.paths} paired paths (bands: 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "irf.png", dpi=160)

    print("\nAggressiveness deviation at t = 4 (% of mean, ± 1.96 s.e.):")
    for k, lab in enumerate(labels):
        print(f"  {lab:<18} {100 * summary['aggressiveness_dev_mean'][k][3]:+.2f} "
              f"± {196 * summary['aggressiveness_dev_se'][k][3]:.2f}")
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
