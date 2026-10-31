"""
Shock test: is the collusive part sustained by punishment?
Rules: PREREGISTRATION.md, "Pre-registration 4".

For each profile, learn (pair and lone-learner controls), then hit the market with a
one-off adverse noise-trade shock and measure how much more aggressively each speculator
trades in the next period, relative to the same path without the shock. Punishment =
reaction in the pair minus the reaction of a lone learner (which needs no coordination).

    python scripts/10_shock_test.py                    # 24 runs, ~45 min on an M4
    python scripts/10_shock_test.py --sigma-u 0.1      # the rule-relevant half
    python scripts/10_shock_test.py --analyze-only

Saved after every run; rerunning skips finished runs.
Outputs in results/shock/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

import _common as C
import _dgj_common as G
from lab import dgj

PROFILES = [(0.0, 0.0), (0.0, 0.25), (0.0, 1.0), (0.5, 0.5), (1.0, 1.0)]
FIXED = {"pair": (None, None), "only0": (None, 1.0), "only1": (1.0, None)}
SHOCKS = (0.0025, 0.025, 0.115, 0.15)
LABELS = ["0.25%", "2.5%", "11.5%", "15%"]
T_AFTER = 3          # index of t = 4 (shock at index 2, t = 3)
MIN_P = 0.005
LARGE = (2, 3)       # 11.5 % and 15 %
SMALL = 0


def cells(prof):
    return ["pair", "only0"] if prof[0] == prof[1] else ["pair", "only0", "only1"]


def plan(sigma_u):
    return [dict(sigma_u=su, q1=p[0], q2=p[1], who=w) for su in sigma_u for p in PROFILES for w in cells(p)]


def key(p):
    return (float(p["sigma_u"]), float(p["q1"]), float(p["q2"]), p["who"])


def responses(res, prm, n, paths) -> np.ndarray:
    """(sessions, shocks, horizon, speculators): aggressiveness change in % of own mean."""
    out = []
    for k in range(n):
        I = dgj.session_irf(res, k, prm, shock_fracs=SHOCKS, n_paths=paths)
        base = I["base_chi"].mean(axis=0)                      # per speculator
        out.append(I["dchi"] / base[None, None, :])
    return np.array(out)


# ------------------------------------------------------------------ analysis

def _R(idx, su, prof, who, agent):
    rec = idx.get((float(su), float(prof[0]), float(prof[1]), who))
    if rec is None:
        return None
    r = np.array(rec["R_t4"])[:, :, agent]                      # (sessions, shocks)
    return r.mean(axis=0), r.std(axis=0, ddof=1) / math.sqrt(r.shape[0])


def punishment(idx, su, prof) -> dict | None:
    """Per speculator: P, SE per shock, and the verdicts."""
    sym = prof[0] == prof[1]
    out = {}
    for a in ((0, 1) if not sym else (0, 1)):
        pr = _R(idx, su, prof, "pair", a)
        ctrl = _R(idx, su, prof, "only0" if sym else f"only{a}", 0 if sym else a)
        if pr is None or ctrl is None:
            return None
        P = pr[0] - ctrl[0]
        se = np.hypot(pr[1], ctrl[1])
        punishes = all(P[k] > 2 * se[k] and P[k] >= MIN_P for k in LARGE)
        trigger = punishes and P[SMALL] < 0.25 * P[LARGE[0]]
        out[f"spec{a + 1}"] = {"R_pair": pr[0].tolist(), "R_alone": ctrl[0].tolist(),
                               "P": P.tolist(), "se": se.tolist(),
                               "punishes": bool(punishes), "trigger_like": bool(trigger)}
    return out


def any_punish(d):
    return any(v["punishes"] for v in d.values())


def analyze(records, sigma_u, s1_held=True) -> dict:
    idx = {key(r): r for r in records}
    v = {"rules": {"MIN_P": MIN_P}, "per_sigma": {}}
    for su in sigma_u:
        v["per_sigma"][str(su)] = {f"{p[0]:g},{p[1]:g}": punishment(idx, su, p) for p in PROFILES}
    low = v["per_sigma"].get("0.1", {})
    t1 = low.get("0,0")
    if t1 is None:
        v.update(T1="not run", T2="not run", T3="not run", T4="not run")
        return v
    v["T1"] = "yes" if any_punish(t1) else "no"
    if v["T1"] == "no":
        v.update(T2="inconclusive", T3="inconclusive", T4="inconclusive")
        return v
    d2, d3 = low.get("1,1"), low.get("0,1")
    v["T2"] = "not run" if d2 is None else ("punishment-based" if any_punish(d2) else "not punishment-based")
    if d3 is None:
        v["T3"] = "not run"
    elif not any_punish(d3):
        v["T3"] = "inequality removes punishment"
    else:
        v["T3"] = "punishment survives (" + ", ".join(k for k, x in d3.items() if x["punishes"]) + ")"
    v["T4"] = ("supported" if v["T3"] == "inequality removes punishment" and s1_held else
               "not supported" if d3 is not None else "not run")
    return v


def report_md(v) -> str:
    L = ["# Shock test verdicts", "",
         "Rules: PREREGISTRATION.md, pre-registration 4. Response = change in aggressiveness at t = 4 "
         "in % of the speculator's average; P = pair − lone learner.", "",
         f"- **T1 (the test detects punishment at equal exact information):** {v['T1']}",
         f"- **T2 (rebound at equal noise m = 1):** {v['T2']}",
         f"- **T3 (inequality, one exact vs q = 1):** {v['T3']}",
         f"- **T4 (H3, main test):** {v['T4']}", ""]
    for su, profs in v["per_sigma"].items():
        L += [f"## σᵤ = {su}", "",
              "| profile | speculator | shock | pair | alone | P | ± | punishes | trigger-like |",
              "|---|---|---:|---:|---:|---:|---:|---|---|"]
        for prof, d in profs.items():
            if d is None:
                continue
            for spec, x in d.items():
                for k, lab in enumerate(LABELS):
                    L.append(f"| {prof} | {spec} | {lab} | {100 * x['R_pair'][k]:+.2f} | {100 * x['R_alone'][k]:+.2f} | "
                             f"{100 * x['P'][k]:+.2f} | {100 * x['se'][k]:.2f} | "
                             f"{('yes' if x['punishes'] else 'no') if k == 3 else ''} | "
                             f"{('yes' if x['trigger_like'] else 'no') if k == 3 else ''} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, nargs="+", default=[0.1, 100.0])
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--max-periods", type=float, default=G.LAPTOP["max_periods"])
    ap.add_argument("--out", default="shock")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out = C.out_dir(args.out)
    path = out / "points.json"
    records = json.loads(path.read_text())["points"] if path.exists() else []
    if not args.analyze_only:
        threads = C.init_threads(args.threads)
        seeds = np.arange(1, args.sessions + 1)
        done = {key(r) for r in records if r["sessions"] == args.sessions and r["paths"] == args.paths}
        todo = [p for p in plan(args.sigma_u) if key(p) not in done]
        print(f"{len(todo)} runs to do ({len(done)} already saved), {args.sessions} sessions each", flush=True)
        for n, p in enumerate(todo, 1):
            prm = G.make_params(sigma_u=p["sigma_u"], sigma_eps=(p["q1"], p["q2"]), fixed=FIXED[p["who"]],
                                grid="nash", grid_width=0.3, monitor="realized",
                                max_periods=int(args.max_periods))
            s, res = G.run_point(prm, seeds)
            with C.Timer() as tm:
                R = responses(res, prm, len(seeds), args.paths)
            label = f"[{n}/{len(todo)}] σu={p['sigma_u']:g} q=({p['q1']:g},{p['q2']:g}) {p['who']}"
            G.print_point(label, s)
            m = 100 * R[:, :, T_AFTER, :].mean(axis=0)
            print(f"    response at t=4 (% of mean), shocks {LABELS}: "
                  + "; ".join(f"spec{i + 1} " + " ".join(f"{x:+.2f}" for x in m[:, i]) for i in range(m.shape[1]))
                  + f"  [IRF {tm.seconds:.0f} s]", flush=True)
            records = [r for r in records if key(r) != key(p)]
            records.append({**p, "sessions": len(seeds), "paths": args.paths,
                            "converged_share": s["converged_share"], "restraint": s["restraint_vs_best_response"],
                            "R_t4": R[:, :, T_AFTER, :], "R_mean": R.mean(axis=0),
                            "wall_clock_seconds": s["wall_clock_seconds"] + tm.seconds})
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
