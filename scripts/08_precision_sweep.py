"""
Precision sweep: how the collusive part changes with the size of the precision gap.
Rules: PREREGISTRATION.md, "Pre-registration 2".

Profiles (signal noise of speculator 1, speculator 2):
  (0, 0)                         reference: equal, exact information
  (0, q), q = 0.1 ... 2          one speculator exact, the other noisy      -> rule P1
  (0.5, 0.5)                     reference: equal, noisy information
  (0.25, 0.75)                   both noisy, unequal                        -> rule P2

For each profile: the pair (both learn) and each speculator learning alone against the
other fixed at its Nash strategy. Equal grid +-30 %, variant A, cap 2 x 10^8.

    python scripts/08_precision_sweep.py                  # 44 points, ~1 h on an M4
    python scripts/08_precision_sweep.py --sigma-u 0.1    # half
    python scripts/08_precision_sweep.py --analyze-only

Saved after every point; rerunning skips finished points.
Outputs in results/precision/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

import _common as C
import _dgj_common as G

SWEEP = [0.1, 0.25, 0.5, 1.0, 2.0]
REF_EXACT = (0.0, 0.0)
REF_NOISY = (0.5, 0.5)
BOTH_NOISY = (0.25, 0.75)
FIXED = {"pair": (None, None), "only0": (None, 1.0), "only1": (1.0, None)}
MIN_C = 0.03
RATIO = 0.5


def profiles(sweep=SWEEP) -> list[tuple[float, float]]:
    return [REF_EXACT] + [(0.0, q) for q in sweep] + [REF_NOISY, BOTH_NOISY]


def cells(prof) -> list[str]:
    return ["pair", "only0"] if prof[0] == prof[1] else ["pair", "only0", "only1"]


def plan(sigma_u, sweep=SWEEP) -> list[dict]:
    return [dict(sigma_u=su, q1=p[0], q2=p[1], who=w)
            for su in sigma_u for p in profiles(sweep) for w in cells(p)]


def key(p) -> tuple:
    return (float(p["sigma_u"]), float(p["q1"]), float(p["q2"]), p["who"])


def _r(idx, su, prof, who, agent):
    rec = idx.get((float(su), float(prof[0]), float(prof[1]), who))
    if rec is None:
        return None
    return float(rec["restraint"][agent]), float(rec["restraint_se"][agent]), float(rec["converged_share"])


def decompose(idx, su, prof) -> dict | None:
    """Per speculator: collusive part c and learning bias b (symmetric: one entry)."""
    if prof[0] == prof[1]:
        pa = [_r(idx, su, prof, "pair", a) for a in (0, 1)]
        al = _r(idx, su, prof, "only0", 0)
        if None in pa or al is None:
            return None
        pm = (pa[0][0] + pa[1][0]) / 2
        pm_se = math.hypot(pa[0][1], pa[1][1]) / 2
        return {"both": {"c": al[0] - pm, "c_se": math.hypot(al[1], pm_se), "b": 1 - al[0],
                         "b_se": al[1], "converged": min(pa[0][2], al[2])}}
    out = {}
    for a in (0, 1):
        pr = _r(idx, su, prof, "pair", a)
        fx = _r(idx, su, prof, f"only{a}", a)
        if pr is None or fx is None:
            return None
        out[f"spec{a + 1}"] = {"c": fx[0] - pr[0], "c_se": math.hypot(fx[1], pr[1]), "b": 1 - fx[0],
                               "b_se": fx[1], "converged": min(pr[2], fx[2])}
    return out


def informative(ref: dict) -> bool:
    r = ref["both"]
    return r["c"] >= MIN_C and r["c"] > 2 * r["c_se"]


def rho(d: dict, ref: dict) -> float:
    return max(v["c"] for v in d.values()) / ref["both"]["c"]


def p1_cell(rhos: dict) -> str:
    """rhos: {q: rho} for one sigma_u."""
    if rhos[0.1] <= RATIO:
        return "collapse"
    if rhos[1.0] <= RATIO and rhos[2.0] <= RATIO:
        return "gradual"
    return "mixed"


def combine(vs: list[str], disagree: str) -> str:
    vs = [v for v in vs if v != "uninformative"]
    if not vs:
        return "uninformative"
    return vs[0] if all(v == vs[0] for v in vs) else disagree


def analyze(records, sigma_u, sweep=SWEEP) -> dict:
    idx = {key(r): r for r in records}
    v = {"rules": {"MIN_C": MIN_C, "RATIO": RATIO}, "per_sigma": {}, "P1": None, "P2": None}
    p1s, p2s = [], []
    for su in sigma_u:
        e = {"profiles": {}}
        for prof in profiles(sweep):
            d = decompose(idx, su, prof)
            if d is not None:
                e["profiles"][f"{prof[0]:g},{prof[1]:g}"] = d
        ref_e = decompose(idx, su, REF_EXACT)
        if ref_e is not None and all(decompose(idx, su, (0.0, q)) for q in sweep):
            rh = {q: rho(decompose(idx, su, (0.0, q)), ref_e) for q in sweep}
            e["rho_exact"] = {f"{q:g}": x for q, x in rh.items()}
            e["P1"] = p1_cell(rh) if informative(ref_e) else "uninformative"
            p1s.append(e["P1"])
        ref_n = decompose(idx, su, REF_NOISY)
        dn = decompose(idx, su, BOTH_NOISY)
        if ref_n is not None and dn is not None:
            e["rho_noisy"] = rho(dn, ref_n)
            e["P2"] = (("holds" if e["rho_noisy"] <= RATIO else "fails")
                       if informative(ref_n) else "uninformative")
            p2s.append(e["P2"])
        v["per_sigma"][str(su)] = e
    v["P1"] = combine(p1s, "depends on noise trading") if p1s else "not run"
    if not p2s:
        v["P2"] = "not run"
    else:
        inf = [x for x in p2s if x != "uninformative"]
        v["P2"] = "uninformative" if not inf else ("holds" if all(x == "holds" for x in inf) else "fails")
    return v


def report_md(v) -> str:
    L = ["# Precision sweep verdicts", "",
         "Rules: PREREGISTRATION.md, pre-registration 2. c = collusive part, b = learning bias "
         "(% of best response); ρ = larger collusive part ÷ reference.", "",
         f"- **P1 (shape):** {v['P1']}", f"- **P2 (no perfectly informed speculator needed):** {v['P2']}", ""]
    for su, e in v["per_sigma"].items():
        L.append(f"## σᵤ = {su}  —  P1: {e.get('P1', '–')}, P2: {e.get('P2', '–')}")
        L += ["", "| profile (q₁, q₂) | speculator | c | ± | b | ± | conv | ρ |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for prof, d in e["profiles"].items():
            r = e.get("rho_exact", {}).get(prof.split(",")[1]) if prof.startswith("0,") and prof != "0,0" else (
                e.get("rho_noisy") if prof == "0.25,0.75" else None)
            for k, (spec, x) in enumerate(d.items()):
                L.append(f"| {prof} | {spec} | {100 * x['c']:+.1f} | {100 * x['c_se']:.1f} | {100 * x['b']:+.1f} | "
                         f"{100 * x['b_se']:.1f} | {x['converged']:.0%} | {'' if r is None or k else f'{r:.2f}'} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, nargs="+", default=[0.1, 100.0])
    ap.add_argument("--max-periods", type=float, default=G.LAPTOP["max_periods"])
    ap.add_argument("--out", default="precision")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out = C.out_dir(args.out)
    path = out / "points.json"
    records = json.loads(path.read_text())["points"] if path.exists() else []
    if not args.analyze_only:
        threads = C.init_threads(args.threads)
        seeds = np.arange(1, args.sessions + 1)
        done = {key(r) for r in records if r["sessions"] == args.sessions}
        todo = [p for p in plan(args.sigma_u) if key(p) not in done]
        print(f"{len(todo)} points to run ({len(done)} already saved), {args.sessions} runs each", flush=True)
        for n, p in enumerate(todo, 1):
            prm = G.make_params(sigma_u=p["sigma_u"], sigma_eps=(p["q1"], p["q2"]), fixed=FIXED[p["who"]],
                                grid="nash", grid_width=0.3, monitor="realized",
                                max_periods=int(args.max_periods))
            s, _ = G.run_point(prm, seeds)
            G.print_point(f"[{n}/{len(todo)}] σu={p['sigma_u']:g} q=({p['q1']:g},{p['q2']:g}) {p['who']}", s)
            records = [r for r in records if key(r) != key(p)]
            records.append({**p, "sessions": s["sessions"], "converged_share": s["converged_share"],
                            "restraint": s["restraint_vs_best_response"],
                            "restraint_se": s["restraint_vs_best_response_se"],
                            "delta": s["delta_all"], "benchmarks": s["benchmarks"],
                            "wall_clock_seconds": s["wall_clock_seconds"]})
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
