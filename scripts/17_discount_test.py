"""
Which channel is it? (PREREGISTRATION.md, "Pre-registration 11".) Agents with discount 0
cannot punish, so coordination that survives at discount 0 speaks against a punishment
channel. Coordination that falls is not proof of punishment: the discount also changes
learning (initial Q-values scale with 1 / (1 - rho), convergence changes).

Four profiles (exact information and noise 0.5 with w = 0 and w = 1 at sigma_u = 0.1;
exact information at sigma_u = 100), each pair + lone learner, discounts 0 and 0.5, plus
0.95 rerun on the same seeds 1-20 as the paired reference (checked against the saved
means of results/precision_fine and results/shared_noise).

    python scripts/17_discount_test.py                # 24 points, about 40 min on an M4
    python scripts/17_discount_test.py --analyze-only
    python scripts/17_discount_test.py --sessions 2 --max-periods 1e6   # smoke -> discount_test_smoke

Saved after every point; rerunning skips finished points.
Outputs in results/discount_test/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

import _common as C
import _dgj_common as G

_spec = importlib.util.spec_from_file_location("measure", Path(__file__).with_name("16_measure_checks.py"))
MC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MC)

SEEDS = np.arange(1, 21)
CAP = int(2e8)
PROFILES = {"exact": dict(sigma_u=0.1, q=0.0, w=0.0), "w0": dict(sigma_u=0.1, q=0.5, w=0.0),
            "w1": dict(sigma_u=0.1, q=0.5, w=1.0), "exact100": dict(sigma_u=100.0, q=0.0, w=0.0)}
RHO_REF = 0.95
RHOS = [0.0, 0.5]
GATE, CHANNEL = "exact", "w1"
MIN_CHANGE, FALL_RATIO = 0.01, 0.5
REF_ATOL = 1e-9


def plan() -> list[dict]:
    return [dict(profile=pr, rho=rho, who=who) for rho in [RHO_REF] + RHOS for pr in PROFILES
            for who in ("pair", "only0")]


def key(p):
    return (p["profile"], float(p["rho"]), p["who"])


def params_for(p, records=None, max_periods=None):
    f = PROFILES[p["profile"]]
    return G.make_params(sigma_u=f["sigma_u"], sigma_eps=(f["q"], f["q"]), noise_shared=f["w"],
                         fixed=(None, None) if p["who"] == "pair" else (None, 1.0), rho=float(p["rho"]),
                         grid="nash", grid_width=0.3, monitor="realized", price_grid="per_value",
                         max_periods=int(max_periods or CAP))


# ------------------------------------------------------------------ analysis

def discount_rule(c_new, c_ref, se) -> str:
    d, tol = c_new - c_ref, max(MIN_CHANGE, 2 * se)
    if abs(d) < tol:
        return "unchanged"
    if c_new < FALL_RATIO * c_ref:
        return "falls"
    if d > tol:
        return "rises"
    return "in between"


def channel(gate: str | None, w1: str | None) -> str:
    if gate is None or w1 is None:
        return "not run"
    if gate != "falls":
        return "test not informative"
    if w1 == "unchanged":
        return "evidence against a punishment channel"
    if w1 == "falls":
        return "consistent with punishment, not proof (discount also changes learning)"
    return "inconclusive"


def reference(p, results=C.RESULTS) -> dict | None:
    """Saved record of the same point at rho = 0.95, seeds 1-20, from the earlier experiments."""
    f = PROFILES[p["profile"]]
    if f["q"] == 0:
        path, k = results / "precision_fine" / "points.json", (f["sigma_u"], 0.0, 0.0, p["who"])
        kf = lambda r: (float(r["sigma_u"]), float(r["q1"]), float(r["q2"]), r["who"])
    else:
        path, k = results / "shared_noise" / "points.json", (f["sigma_u"], f["w"], p["who"])
        kf = lambda r: (float(r["sigma_u"]), float(r["w"]), r["who"])
    if not path.exists():
        return None
    return next((r for r in json.loads(path.read_text())["points"] if kf(r) == k), None)


def check_reference(rec, ref) -> dict:
    if ref is None:
        return {"status": "no reference"}
    if rec["sessions"] != ref["sessions"] or rec["max_periods"] != CAP:
        return {"status": "not comparable (sessions or cap differ)"}
    diff = float(np.max(np.abs(np.array(rec["restraint"]) - np.array(ref["restraint"]))))
    return {"status": "match" if diff <= REF_ATOL else "mismatch", "max_abs_diff": diff}


def analyze(records, results=C.RESULTS) -> dict:
    idx = {key(r): r for r in records}
    v = {"rules": {"MIN_CHANGE": MIN_CHANGE, "FALL_RATIO": FALL_RATIO}, "points": {}, "reference": {},
         "profiles": {}, "channel": {}}
    for k, r in idx.items():
        v["points"][f"{k[0]} ρ={k[1]:g} {k[2]}"] = {**MC.conv_learn(r), "restraint": r["restraint"]}
        if k[1] == RHO_REF:
            v["reference"][f"{k[0]} {k[2]}"] = check_reference(r, reference(r, results))
    cs = {}
    for pr in PROFILES:
        for rho in [RHO_REF] + RHOS:
            a, b = idx.get((pr, rho, "pair")), idx.get((pr, rho, "only0"))
            if a is not None and b is not None:
                cs[pr, rho] = MC.c_sessions(a, b)
                c, se = MC.mean_se(cs[pr, rho])
                v["profiles"].setdefault(pr, {})[f"{rho:g}"] = {"c": c, "c_se": se, "c_se_old": MC.old_se(a, b)}
        for rho in RHOS:
            if (pr, rho) in cs and (pr, RHO_REF) in cs:
                if not MC.same_seeds(idx[pr, rho, "pair"], idx[pr, RHO_REF, "pair"]):
                    raise ValueError("discounts run on different seeds")
                d, se = MC.paired(cs[pr, rho], cs[pr, RHO_REF])
                e = v["profiles"][pr][f"{rho:g}"]
                e.update(dc=d, dc_se=se, label=discount_rule(e["c"], v["profiles"][pr][f"{RHO_REF:g}"]["c"], se))
    lab = lambda pr, rho: v["profiles"].get(pr, {}).get(f"{rho:g}", {}).get("label")
    for rho in RHOS:   # gate at the same discount; rho = 0 is the main test, 0.5 secondary
        v["channel"][f"{rho:g}"] = channel(lab(GATE, rho), lab(CHANNEL, rho))
    return v


def report_md(v) -> str:
    L = ["# Discount test", "",
         "Rules: PREREGISTRATION.md, pre-registration 11. c = collusive part (% of best response), seeds 1–20; "
         f"Δc = c(ρ′) − c({RHO_REF}), paired SE by seed; old ± = formula of 15_shared_noise.py.", "",
         f"- **Channel (ρ = 0, main):** {v['channel']['0']}  (gate: exact information σᵤ = 0.1 must fall)",
         f"- Channel (ρ = 0.5, secondary): {v['channel']['0.5']}", "",
         "| profile | ρ | c | SE | old ± | Δc | SE | label |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for pr, e in v["profiles"].items():
        for rho, d in e.items():
            dc = f"{100 * d['dc']:+.2f} | {100 * d['dc_se']:.2f} | {d['label']}" if "label" in d else "– | – | reference"
            L.append(f"| {pr} | {rho} | {100 * d['c']:+.2f} | {100 * d['c_se']:.2f} | {100 * d['c_se_old']:.2f} | {dc} |")
    L += ["", f"## Reference check (ρ = {RHO_REF} rerun against saved means, atol {REF_ATOL:g})", ""]
    L += [f"- {k}: {x['status']}" + (f" (max |diff| {x['max_abs_diff']:.2e})" if "max_abs_diff" in x else "")
          for k, x in v["reference"].items()]
    L += ["", "## Convergence per point", "", "| point | converged | mean learning periods |", "|---|---:|---:|"]
    L += [f"| {k} | {x['converged']:.0%} | {x['t_learn_mean']:.3g} |" for k, x in v["points"].items()]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=len(SEEDS))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="discount_test")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--max-periods", type=float, default=None, help="smoke tests only")
    args = ap.parse_args()

    out = C.out_dir(MC.smoke_out(args))
    path = out / "points.json"
    records = json.loads(path.read_text())["points"] if path.exists() else []
    if not args.analyze_only:
        threads = C.init_threads(args.threads)
        seeds = SEEDS[:args.sessions]
        todo = MC.pending(plan(), key, records, len(seeds))
        print(f"{len(todo)} points to run ({len(records)} already saved), {len(seeds)} sessions each", flush=True)
        records = MC.run(todo, key, lambda p, rs: params_for(p, rs, args.max_periods), records, path, seeds,
                         threads, lambda p: f"{p['profile']} ρ={p['rho']:g} {p['who']}")
    v = analyze(records)
    C.save_json(out / "verdicts.json", v)
    md = report_md(v)
    (out / "verdicts.md").write_text(md)
    print()
    print(md)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
