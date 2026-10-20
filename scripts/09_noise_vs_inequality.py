"""
Noise versus inequality: is the drop in collusion under unequal precision caused by
inequality, or by noise as such? Rules: PREREGISTRATION.md, "Pre-registration 3".

Compares equal-noise profiles (m, m) with unequal profiles of the same average noise m:
one speculator exact (0, 2m), or both noisy (m/2, 3m/2). Shares results/precision/
points.json with 08_precision_sweep.py, so points from the sweep are reused.

    python scripts/09_noise_vs_inequality.py              # 28 new points, ~30 min on an M4
    python scripts/09_noise_vs_inequality.py --analyze-only

Outputs in results/precision/: points.json (shared), noise_vs_inequality.json / .md
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

_spec = importlib.util.spec_from_file_location("sweep", Path(__file__).with_name("08_precision_sweep.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

MS = [0.05, 0.125, 0.25, 0.5, 1.0]
MS_BOTH = [0.25, 0.5, 1.0]
MIN_E = 0.01
MIN_TOTAL = 0.02


def families():
    """m -> profiles: equal, one exact, both noisy (or None)."""
    return {m: {"equal": (m, m), "exact": (0.0, 2 * m),
                "both": (m / 2, 3 * m / 2) if m in MS_BOTH else None} for m in MS}


def all_profiles():
    out = [(0.0, 0.0)]
    for f in families().values():
        out += [p for p in f.values() if p is not None]
    return out


def plan(sigma_u):
    return [dict(sigma_u=su, q1=p[0], q2=p[1], who=w)
            for su in sigma_u for p in all_profiles() for w in S.cells(p)]


def c_of(idx, su, prof):
    """(c, se) for a profile: symmetric -> its c; unequal -> mean over speculators."""
    d = S.decompose(idx, su, prof)
    if d is None:
        return None
    if "both" in d:
        return d["both"]["c"], d["both"]["c_se"]
    a, b = d["spec1"], d["spec2"]
    return (a["c"] + b["c"]) / 2, math.hypot(a["c_se"], b["c_se"]) / 2


def effects(idx, su, family: str) -> dict | None:
    c0 = c_of(idx, su, (0.0, 0.0))
    if c0 is None:
        return None
    rows = {}
    for m, f in families().items():
        if f[family] is None:
            continue
        eq, un = c_of(idx, su, f["equal"]), c_of(idx, su, f[family])
        if eq is None or un is None:
            return None
        N, N_se = c0[0] - eq[0], math.hypot(c0[1], eq[1])
        E, E_se = eq[0] - un[0], math.hypot(eq[1], un[1])
        tot = N + E
        rows[m] = {"c_equal": eq[0], "c_unequal": un[0], "N": N, "N_se": N_se, "E": E, "E_se": E_se,
                   "detected": E >= MIN_E and E > 2 * E_se,
                   "share": E / tot if tot >= MIN_TOTAL else None}
    return {"c0": c0[0], "rows": rows}


def i1(counts: list[int], n: int) -> str:
    yes_at, no_at = (3, 1) if n == 5 else (2, 0)
    if all(k >= yes_at for k in counts):
        return "yes"
    if all(k <= no_at for k in counts):
        return "no"
    return "partial"


def i2_cell(rows: dict) -> str:
    sh = [r["share"] for r in rows.values() if r["share"] is not None]
    if not sh:
        return "undefined"
    if all(x < 0.5 for x in sh):
        return "mainly noise"
    if all(x > 0.5 for x in sh):
        return "mainly inequality"
    return "mixed"


def analyze(records, sigma_u) -> dict:
    idx = {S.key(r): r for r in records}
    v = {"rules": {"MIN_E": MIN_E, "MIN_TOTAL": MIN_TOTAL}, "per_sigma": {}}
    for family, n in (("exact", len(MS)), ("both", len(MS_BOTH))):
        counts, i2s = [], []
        for su in sigma_u:
            e = effects(idx, su, family)
            if e is None:
                continue
            e["I2"] = i2_cell(e["rows"])
            e["detected"] = sum(r["detected"] for r in e["rows"].values())
            v["per_sigma"].setdefault(str(su), {})[family] = e
            counts.append(e["detected"])
            i2s.append(e["I2"])
        v[f"I1_{family}"] = i1(counts, n) if len(counts) == len(sigma_u) else "not run"
        v[f"I2_{family}"] = S.combine(i2s, "depends on noise trading") if i2s else "not run"
    return v


def report_md(v) -> str:
    L = ["# Noise versus inequality", "",
         "Rules: PREREGISTRATION.md, pre-registration 3. c = collusive part (% of best response); "
         "N = drop from (0,0) to equal noise m; E = further drop from equal to unequal at the same m; "
         "s = E ÷ (N + E).", "",
         f"- **I1 (inequality has its own effect), one exact:** {v['I1_exact']}",
         f"- **I2 (which is larger), one exact:** {v['I2_exact']}",
         f"- I1 / I2, both noisy (secondary): {v['I1_both']} / {v['I2_both']}", ""]
    for su, fams in v["per_sigma"].items():
        for family, e in fams.items():
            L.append(f"## σᵤ = {su}, unequal = {'one exact (0, 2m)' if family == 'exact' else 'both noisy (m/2, 3m/2)'}"
                     f" — c at (0,0) = {100 * e['c0']:.1f} %, E detected at {e['detected']}, I2: {e['I2']}")
            L += ["", "| m | c equal | c unequal | N | E | ± | detected | s |", "|---:|---:|---:|---:|---:|---:|---|---:|"]
            for m, r in e["rows"].items():
                s = "–" if r["share"] is None else f"{r['share']:.2f}"
                L.append(f"| {m:g} | {100 * r['c_equal']:.1f} | {100 * r['c_unequal']:.1f} | {100 * r['N']:+.1f} | "
                         f"{100 * r['E']:+.1f} | {100 * r['E_se']:.1f} | {'yes' if r['detected'] else 'no'} | {s} |")
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
        done = {S.key(r) for r in records if r["sessions"] == args.sessions}
        todo = [p for p in plan(args.sigma_u) if S.key(p) not in done]
        print(f"{len(todo)} points to run ({len(done)} already saved), {args.sessions} runs each", flush=True)
        for n, p in enumerate(todo, 1):
            prm = G.make_params(sigma_u=p["sigma_u"], sigma_eps=(p["q1"], p["q2"]), fixed=S.FIXED[p["who"]],
                                grid="nash", grid_width=0.3, monitor="realized",
                                max_periods=int(args.max_periods))
            s, _ = G.run_point(prm, seeds)
            G.print_point(f"[{n}/{len(todo)}] σu={p['sigma_u']:g} q=({p['q1']:g},{p['q2']:g}) {p['who']}", s)
            records = [r for r in records if S.key(r) != S.key(p)]
            records.append({**p, "sessions": s["sessions"], "converged_share": s["converged_share"],
                            "restraint": s["restraint_vs_best_response"],
                            "restraint_se": s["restraint_vs_best_response_se"],
                            "delta": s["delta_all"], "benchmarks": s["benchmarks"],
                            "wall_clock_seconds": s["wall_clock_seconds"]})
            C.save_json(path, {"machine": C.machine_info(threads), "points": records})
    v = analyze(records, args.sigma_u)
    C.save_json(out / "noise_vs_inequality.json", v)
    md = report_md(v)
    (out / "noise_vs_inequality.md").write_text(md)
    print()
    print(md)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
