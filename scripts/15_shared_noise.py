"""
Shared versus private noise (PREREGISTRATION.md, "Pre-registration 7", with its
amendments). Both speculators get noise 0.5; only the shared part w of the noise
variance changes, w = 0, 0.25, 0.5, 0.75, 1. Price grid per lagged value, variant A.

For each sigma_u and w: the pair and one lone learner (opponent fixed at Nash). At
sigma_u = 0.1 every run also gets the shock test. At w = 0 and w = 1 the collusive part is
also measured directly (opponent fixed at the level it reached alone, as in
pre-registration 6).

    python scripts/15_shared_noise.py               # 24 runs, about 45 min on an M4
    python scripts/15_shared_noise.py --analyze-only

Saved after every run; rerunning skips finished runs.
Outputs in results/shared_noise/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

import _common as C
import _dgj_common as G

_spec = importlib.util.spec_from_file_location("shock", Path(__file__).with_name("10_shock_test.py"))
SH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SH)

Q = 0.5
SHARES = [0.0, 0.25, 0.5, 0.75, 1.0]
DIRECT = [0.0, 1.0]
# V1: Renat's prediction (pre-registration 7), sigma_u = 0.1, w = 1
V1_CONFIRMED, V1_WRONG = 0.06, 0.03
# V2: secondary, sigma_u = 100, w = 1 (as amended on 12 November 2026)
V2_SHARE_OF_EXACT = 0.6
C_EXACT_FINE = {0.1: 0.124, 100.0: 0.111}   # collusive part at exact information, fine grid (pre-reg 9)
MIN_P = 0.005


def plan(sigma_u):
    pts = []
    for su in sigma_u:
        for w in SHARES:
            pts += [dict(sigma_u=su, w=w, who="pair"), dict(sigma_u=su, w=w, who="only0")]
        pts += [dict(sigma_u=su, w=w, who="solo_level0") for w in DIRECT]
    return pts


def key(p):
    return (float(p["sigma_u"]), float(p["w"]), p["who"])


def params_for(p, records):
    fixed = (None, None) if p["who"] == "pair" else (None, 1.0)
    if p["who"] == "solo_level0":
        alone = {key(r): r for r in records}[(float(p["sigma_u"]), float(p["w"]), "only0")]
        fixed = (None, float(alone["restraint"][0]))   # symmetric: the other's solo level
    return G.make_params(sigma_u=p["sigma_u"], sigma_eps=(Q, Q), noise_shared=p["w"], fixed=fixed,
                         grid="nash", grid_width=0.3, monitor="realized", price_grid="per_value")


# ------------------------------------------------------------------ analysis

def collusive_part(idx, su, w):
    pr, al = idx.get((float(su), float(w), "pair")), idx.get((float(su), float(w), "only0"))
    if pr is None or al is None:
        return None
    rp = (pr["restraint"][0] + pr["restraint"][1]) / 2
    rp_se = math.hypot(*pr["restraint_se"]) / 2
    c, se = al["restraint"][0] - rp, math.hypot(al["restraint_se"][0], rp_se)
    out = {"c": c, "c_se": se, "b": 1 - al["restraint"][0], "converged": pr["converged_share"],
           "delta": pr["delta"]["mean"]}
    dr = idx.get((float(su), float(w), "solo_level0"))
    if dr is not None:
        out["c_direct"] = dr["restraint"][0] - pr["restraint"][0]
        out["c_direct_se"] = math.hypot(dr["restraint_se"][0], pr["restraint_se"][0])
    return out


def v1(d1):
    c, se = d1["c"], d1["c_se"]
    if c >= V1_CONFIRMED and c > 2 * se:
        return "confirmed"
    if c < V1_WRONG:
        return "wrong"
    return "partly right / inconclusive"


def v2(d1, d0, su):
    rise, rise_se = d1["c"] - d0["c"], math.hypot(d1["c_se"], d0["c_se"])
    if rise <= 2 * rise_se:
        return "wrong"
    if d1["c"] >= V2_SHARE_OF_EXACT * C_EXACT_FINE[su]:
        return "confirmed"
    return "partly right / inconclusive"


def punishment(idx, su, w):
    """Rule of pre-registration 4 plus the amendment: the pair's own reaction must also be
    >= MIN_P and > 2 SE at both large shocks."""
    pr, al = idx.get((float(su), float(w), "pair")), idx.get((float(su), float(w), "only0"))
    if pr is None or al is None or "R_t4" not in pr or "R_t4" not in al:
        return None
    A = np.array(al["R_t4"])[:, :, 0]
    out = {}
    for i in (0, 1):
        R = np.array(pr["R_t4"])[:, :, i]
        m, se = R.mean(0), R.std(0, ddof=1) / math.sqrt(R.shape[0])
        am, ase = A.mean(0), A.std(0, ddof=1) / math.sqrt(A.shape[0])
        P, Pse = m - am, np.hypot(se, ase)
        ok = all(P[k] > 2 * Pse[k] and P[k] >= MIN_P and m[k] >= MIN_P and m[k] > 2 * se[k] for k in SH.LARGE)
        out[f"spec{i + 1}"] = {"pair": m.tolist(), "pair_se": se.tolist(), "alone": am.tolist(),
                               "P": P.tolist(), "P_se": Pse.tolist(), "punishes": bool(ok)}
    return out


def analyze(records, sigma_u) -> dict:
    idx = {key(r): r for r in records}
    v = {"V1": "not run", "V2": "not run", "V3": "not run", "curve": {}, "shock": {}}
    for su in sigma_u:
        v["curve"][str(su)] = {f"{w:g}": collusive_part(idx, su, w) for w in SHARES}
    if 0.1 in sigma_u:
        d1 = v["curve"]["0.1"]["1"]
        if d1 is not None:
            v["V1"] = v1(d1)
        for w in SHARES:
            v["shock"][f"{w:g}"] = punishment(idx, 0.1, w)
        p1 = v["shock"]["1"]
        if p1 is not None:
            v["V3"] = "punishment returns" if any(x["punishes"] for x in p1.values()) else "no punishment"
    if 100.0 in sigma_u:
        d1, d0 = v["curve"]["100.0"]["1"], v["curve"]["100.0"]["0"]
        if d1 is not None and d0 is not None:
            v["V2"] = v2(d1, d0, 100.0)
    return v


def report_md(v) -> str:
    L = ["# Shared versus private noise", "",
         "Rules: PREREGISTRATION.md, pre-registration 7 and its amendments. Both speculators have noise "
         "0.5; w = shared part of the noise variance. c = collusive part (% of best response).", "",
         f"- **V1 (Renat's prediction, σᵤ = 0.1, w = 1):** {v['V1']}",
         f"- **V2 (σᵤ = 100, w = 1):** {v['V2']}",
         f"- **V3 (punishment, σᵤ = 0.1, w = 1):** {v['V3']}", ""]
    for su, curve in v["curve"].items():
        L += [f"## σᵤ = {su}", "", "| w | c | ± | direct | ± | learning bias | converged | Δ |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for w, d in curve.items():
            if d is None:
                continue
            direct = (f"{100 * d['c_direct']:+.1f} | {100 * d['c_direct_se']:.1f}" if "c_direct" in d else "– | –")
            L.append(f"| {w} | {100 * d['c']:+.1f} | {100 * d['c_se']:.1f} | {direct} | {100 * d['b']:+.1f} | "
                     f"{d['converged']:.0%} | {d['delta']:.3f} |")
        L.append("")
    if any(x is not None for x in v["shock"].values()):
        L += ["## Shock test, σᵤ = 0.1 (15 % shock)", "",
              "| w | speculator | pair | ± | alone | P | ± | punishes |", "|---:|---|---:|---:|---:|---:|---:|---|"]
        for w, d in v["shock"].items():
            if d is None:
                continue
            for spec, x in d.items():
                L.append(f"| {w} | {spec} | {100 * x['pair'][3]:+.2f} | {100 * x['pair_se'][3]:.2f} | "
                         f"{100 * x['alone'][3]:+.2f} | {100 * x['P'][3]:+.2f} | {100 * x['P_se'][3]:.2f} | "
                         f"{'yes' if x['punishes'] else 'no'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, nargs="+", default=[0.1, 100.0])
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--out", default="shared_noise")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--max-periods", type=float, default=None, help="smoke tests only")
    args = ap.parse_args()
    if args.max_periods:
        G.LAPTOP["max_periods"] = int(args.max_periods)

    out = C.out_dir(args.out)
    path = out / "points.json"
    records = json.loads(path.read_text())["points"] if path.exists() else []
    if not args.analyze_only:
        threads = C.init_threads(args.threads)
        seeds = np.arange(1, args.sessions + 1)
        todo = [p for p in plan(args.sigma_u) if key(p) not in {key(r) for r in records}]
        print(f"{len(todo)} runs to do ({len(records)} already saved), {args.sessions} sessions each", flush=True)
        for n, p in enumerate(todo, 1):
            prm = params_for(p, records)
            s, res = G.run_point(prm, seeds)
            rec = {**p, "sessions": len(seeds), "converged_share": s["converged_share"],
                   "restraint": s["restraint_vs_best_response"], "restraint_se": s["restraint_vs_best_response_se"],
                   "delta": s["delta_all"], "aggressiveness_vs_nash": s["aggressiveness_vs_nash"],
                   "benchmarks": s["benchmarks"], "wall_clock_seconds": s["wall_clock_seconds"]}
            if p["sigma_u"] == 0.1 and p["who"] != "solo_level0":
                R = SH.responses(res, prm, len(seeds), args.paths)
                rec["R_t4"] = R[:, :, SH.T_AFTER, :]
                rec["paths"] = args.paths
            G.print_point(f"[{n}/{len(todo)}] σu={p['sigma_u']:g} w={p['w']:g} {p['who']}", s)
            records.append(rec)
            C.save_json(path, {"machine": C.machine_info(threads), "points": records})
    v = analyze(records, args.sigma_u)
    C.save_json(out / "verdicts.json", v)
    md = report_md(v)
    (out / "verdicts.md").write_text(md)
    print()
    print(md)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
