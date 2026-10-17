"""Shared runner for the DGJ scripts: one parameter point -> summary dict."""
from __future__ import annotations

import numpy as np

import _common as C
from lab import dgj
from lab import dgj_benchmarks as bmk

# Laptop-scale defaults (see TECHNICAL.md, "What is scaled down and why").
# Environment and learning rates are DGJ's; grids, convergence window and the
# period cap are reduced.
LAPTOP = dict(n_v=6, n_x=13, n_p=37, alpha=0.01, beta=5e-7,
              conv_window=100_000, max_periods=200_000_000, eval_periods=100_000, Tm=10_000)


def parse_set(items) -> dict:
    """--set alpha=0.05 n_p=37  ->  {'alpha': 0.05, 'n_p': 37}"""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        default = getattr(dgj.DGJParams(), k)          # raises if the name is wrong
        out[k] = type(default)(float(v)) if isinstance(default, (int, float)) else v
    return out


def tag(items) -> str:
    """Folder suffix for runs with overrides, so they never overwrite the defaults."""
    return "".join("_" + it.replace("=", "") for it in (items or []))


def make_params(**kw) -> dgj.DGJParams:
    p = dict(LAPTOP)
    p.update(kw)
    return dgj.DGJParams(**p)


def run_point(prm: dgj.DGJParams, seeds) -> tuple[dict, dict]:
    env = dgj.setup(prm)
    with C.Timer() as tm:
        res = dgj.simulate(prm, seeds, env)
    bm = env["bm"]
    conv = res["converged"]
    chiN = bm["nash"]["chi"]
    # per-session restraint relative to the static best response (TECHNICAL.md 5.5)
    rr = [bmk.restraint(res["chi"][k], bm["C"], bm["S"], bm["nash"]["lam"]) for k in range(len(seeds))]
    restr = np.array([r["restraint"] for r in rr])
    devg = np.array([r["deviation_gain"] for r in rr])
    n = len(seeds)
    busy = min(_threads(), n)                       # cores actually used
    periods = float(np.sum(res["t_learn"]) + n * prm.eval_periods)
    summary = {
        "params": prm.to_dict(),
        "benchmarks": {
            "lambda_nash": bm["nash"]["lam"], "lambda_cartel": bm["cartel"]["lam"],
            "chi_nash": chiN, "chi_cartel": bm["cartel"]["chi"],
            "profit_nash": bm["nash"]["profit"], "profit_cartel": bm["cartel"]["profit"],
        },
        "sessions": len(seeds),
        "converged_share": float(conv.mean()),
        "periods_learning": C.summary_stats(res["t_learn"]),
        "delta_all": C.summary_stats(res["delta"]),
        "delta_converged": C.summary_stats(res["delta"][conv]),
        "delta_capped": C.summary_stats(res["delta"][~conv]),
        "delta_agent_mean": np.nanmean(res["delta_agent"], axis=0),
        "profit_mean": res["profit"].mean(axis=0),
        "aggressiveness_vs_nash": (res["chi"] / chiN).mean(axis=0),
        "informativeness": C.summary_stats(res["informativeness"]),
        "mispricing": C.summary_stats(res["mispricing"]),
        "restraint_vs_best_response": restr.mean(axis=0),
        "restraint_vs_best_response_se": restr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(restr.shape[1], np.nan),
        "static_deviation_gain": devg.mean(axis=0),
        "wall_clock_seconds": tm.seconds,
        # approximate: on Apple M-series, efficiency cores are slower than performance cores
        "seconds_per_session_per_core": tm.seconds * busy / n,
        "periods_per_second_per_core": periods / (tm.seconds * busy),
    }
    return summary, res


def _threads() -> int:
    import numba
    return numba.get_num_threads()


def print_point(label: str, s: dict) -> None:
    d = s["delta_all"]
    print(f"{label:<34} Δ = {d['mean']:.3f} (s.e. {d['se']:.3f}, p1–p99 {d['p01']:.2f}–{d['p99']:.2f})"
          f"  converged {s['converged_share']:.0%}  "
          f"aggressiveness/Nash {np.round(s['aggressiveness_vs_nash'], 3)}  "
          f"restraint/best-response {np.round(s['restraint_vs_best_response'], 3)}  [{s['wall_clock_seconds']:.0f} s]",
          flush=True)
