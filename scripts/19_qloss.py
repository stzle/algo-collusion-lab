"""
Q-loss of the learned strategies (PREREGISTRATION.md, pre-registration 13). For every saved
session and every learning speculator: the exact value of its learned greedy policy, of its
dynamic best response to the frozen rival and pricing rule, and of the myopic best
response, weighted by how often the learned play visits each state. Method: the "Q-loss"
check of Calvano, Calzolari, Denicolò & Pastorello (2020, AER) (section number unverified);
model and derivation in lab/qloss.py.

Pair minus lone learner gives the part of the Q-loss that needs a learning partner; the
lone learner (rival fixed at Nash) gives the single-agent part. Each session also gets a
simulation check of the model (model vs simulated long-run average profit, common random
numbers), and optionally the same deviation against a market maker that keeps learning.

    python scripts/19_qloss.py                                    # all saved sessions
    python scripts/19_qloss.py --max-sessions 2 --robustness      # quick test
    python scripts/19_qloss.py --analyze-only                     # verdicts from points.json

Inputs in results/qloss_data/ (scripts/save_qloss_data.py).
Outputs in results/<out>/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import _common as C
from lab import dgj
from lab import qloss as QL

SCALARS = ["q_loss", "value_learned", "value_br", "value_myopic", "myopic_gain_static",
           "myopic_gain_dynamic", "avg_profit_learned", "avg_profit_br", "avg_profit_myopic",
           "br_differs", "br_is_myopic", "chi_learned_vs_nash", "chi_br_vs_nash",
           "bellman_residual", "pi_iterations"]


def load(path: Path):
    z = np.load(path)
    p = json.loads(str(z["params"]))
    p["sigma_eps"], p["fixed"] = tuple(p["sigma_eps"]), tuple(p["fixed"])
    prm = dgj.DGJParams(**p)
    return z, prm, dgj.setup(prm)


def mean_se(x) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")}


def run_point(path: Path, max_sessions, sim_periods, robustness) -> dict:
    z, prm, env = load(path)
    n = len(z["seeds"]) if max_sessions is None else min(max_sessions, len(z["seeds"]))
    learners = [i for i in range(prm.I) if not env["fixed_mask"][i]]
    rec = {"file": path.name, "sessions": n, "seeds": z["seeds"][:n], "learners": learners,
           "converged": z["ps_converged"][:n], "delta": z["ps_delta"][:n], "spec": {}}
    for i in learners:
        cols = {k: [] for k in SCALARS}
        cols["static_deviation_gain"] = list(z["ps_deviation_gain"][:n, i])     # cross-check only
        cols["seconds"] = []
        sim, rob = [], []
        for k in range(n):
            Q, mm = z["Q"][k], z["mm"][k]
            with C.Timer() as tm:
                m = QL.build(env, prm, mm, Q, i)
                r = QL.analyze(env, prm, mm, Q, i, model=m)
            cols["seconds"].append(tm.seconds)
            for key in SCALARS:
                cols[key].append(r[key])
            if sim_periods:
                sim.append(QL.model_vs_sim(env, prm, mm, Q, i, res=r, periods=sim_periods,
                                           seed=int(z["seeds"][k]) + 50_000))
            if robustness:
                rob.append(QL.learning_mm_check(env, prm, Q, i, r["policy_br"], seed=int(z["seeds"][k]) + 60_000))
        spec = {"per_session": cols, "summary": {k: mean_se(v) for k, v in cols.items()}}
        if sim:
            spec["sim_check"] = sim
        if rob:
            spec["robustness"] = rob
            spec["summary"]["robust_gain_settled"] = mean_se([x["settled"]["gain"] for x in rob])
        rec["spec"][f"spec{i + 1}"] = spec
        print(f"  {path.stem:<12} spec{i + 1}: Q-loss {100 * spec['summary']['q_loss']['mean']:6.2f} % "
              f"(s.e. {100 * spec['summary']['q_loss']['se']:.2f}), myopic one-period "
              f"{100 * spec['summary']['myopic_gain_static']['mean']:.2f} %, dynamic "
              f"{100 * spec['summary']['myopic_gain_dynamic']['mean']:.2f} %  "
              f"[{np.mean(cols['seconds']):.2f} s/session]", flush=True)
    return rec


def session_mean(rec) -> np.ndarray:
    """Per-session Q-loss, averaged over the learners of that session."""
    return np.mean([rec["spec"][s]["per_session"]["q_loss"] for s in rec["spec"]], axis=0)


def parts(points: dict) -> dict:
    """Lone learner = single-agent part; pair minus lone = the part that needs a partner.
    s.e. from session-level values (independent runs, so they add in quadrature)."""
    out = {}
    for prof in sorted({k.rsplit("_", 1)[0] for k in points}):
        pr, al = points.get(f"{prof}_pair"), points.get(f"{prof}_only0")
        if pr is None or al is None:
            continue
        a, b = mean_se(session_mean(pr)), mean_se(al["spec"]["spec1"]["per_session"]["q_loss"])
        out[prof] = {"pair": a, "lone": b, "pair_part": {"mean": a["mean"] - b["mean"],
                                                         "se": math.hypot(a["se"], b["se"])}}
    return out


# ------------------------------------------------------------------ verdicts

def verdicts(points) -> dict:
    """STUB: descriptive numbers only. The decision rules come from pre-registration 13 and
    are added once it is committed."""
    v = {"rules": "not yet added (pre-registration 13)", "parts": parts(points), "points": {}}
    for name, rec in points.items():
        v["points"][name] = {s: {k: d["summary"][k] for k in ("q_loss", "myopic_gain_static",
                                                              "myopic_gain_dynamic", "static_deviation_gain")}
                             for s, d in rec["spec"].items()}
    return v


def report_md(v) -> str:
    L = ["# Q-loss of the learned strategies", "",
         f"Decision rules: {v['rules']}. Gains in % of the learned policy's value, weighted by the "
         "learned play's visits; ± = session-level s.e.", "",
         "| point | speculator | Q-loss | ± | myopic, one period | myopic, dynamic | static deviation gain |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for name, d in v["points"].items():
        for s, x in d.items():
            L.append(f"| {name} | {s} | {100 * x['q_loss']['mean']:.2f} | {100 * x['q_loss']['se']:.2f} | "
                     f"{100 * x['myopic_gain_static']['mean']:.2f} | {100 * x['myopic_gain_dynamic']['mean']:.2f} | "
                     f"{100 * x['static_deviation_gain']['mean']:.3f} |")
    if v["parts"]:
        L += ["", "## Pair part and single-agent part", "",
              "| profile | pair | lone (single-agent part) | pair − lone | ± |", "|---|---:|---:|---:|---:|"]
        for prof, d in v["parts"].items():
            L.append(f"| {prof} | {100 * d['pair']['mean']:.2f} | {100 * d['lone']['mean']:.2f} | "
                     f"{100 * d['pair_part']['mean']:+.2f} | {100 * d['pair_part']['se']:.2f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.RESULTS / "qloss_data"))
    ap.add_argument("--out", default="qloss")
    ap.add_argument("--max-sessions", type=int, default=None, help="quick tests")
    ap.add_argument("--sim-periods", type=float, default=1e6, help="simulation check per session (0 = off)")
    ap.add_argument("--robustness", action="store_true", help="also the learning market maker check")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out = C.out_dir(args.out)
    path = out / "points.json"
    if args.analyze_only:
        points = json.loads(path.read_text())["points"]
    else:
        threads = C.init_threads(args.threads)
        files = sorted(Path(args.data_dir).glob("*_*.npz"))
        if not files:
            raise SystemExit(f"no saved sessions in {args.data_dir}")
        points = {}
        for f in files:
            points[f.stem] = run_point(f, args.max_sessions, int(args.sim_periods), args.robustness)
            C.save_json(path, {"machine": C.machine_info(threads), "data_dir": str(args.data_dir),
                               "points": points, "parts": parts(points)})
    v = verdicts(points)
    C.save_json(out / "verdicts.json", v)
    md = report_md(v)
    (out / "verdicts.md").write_text(md)
    print()
    print(md)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
