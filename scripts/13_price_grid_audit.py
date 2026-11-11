"""
Audit run: price grid per lagged value (PREREGISTRATION.md, "Pre-registration 8").

  P  baseline at the paper's sizes with price_grid="per_value", cap 1e10
  L  laptop sizes with price_grid="per_value": profiles (0,0), (0,1), (0.5,0.5),
     pair and lone-learner controls, both noise-trading levels

    python scripts/13_price_grid_audit.py              # 16 runs, about 1 h 15 min on an M4
    python scripts/13_price_grid_audit.py --part L     # one part only
    python scripts/13_price_grid_audit.py --analyze-only

Saved after every run; rerunning skips finished runs.
Outputs in results/price_grid/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

import _common as C
import _dgj_common as G

_spec = importlib.util.spec_from_file_location("sweep", Path(__file__).with_name("08_precision_sweep.py"))
SW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SW)

L_PROFILES = [(0.0, 0.0), (0.0, 1.0), (0.5, 0.5)]
PAPER = dict(n_v=10, n_x=15, n_p=31, grid="benchmarks", conv_window=1_000_000,
             conv_max_explore=0.0, max_periods=10_000_000_000, price_grid="per_value")
P_SESSIONS = {"pair": 12, "only0": 6}
SINGLE_GRID_C = {0.1: 0.102, 100.0: 0.072}       # collusive part at (0,0), earlier runs
RATIO = 0.5


def plan(parts, sigma_u):
    pts = []
    if "L" in parts:
        pts += [dict(part="L", sigma_u=su, q1=p[0], q2=p[1], who=w)
                for su in sigma_u for p in L_PROFILES for w in SW.cells(p)]
    if "P" in parts:
        pts += [dict(part="P", sigma_u=0.1, q1=0.0, q2=0.0, who=w) for w in ("pair", "only0")]
    return pts


def key(p):
    return (p["part"], float(p["sigma_u"]), float(p["q1"]), float(p["q2"]), p["who"])


def params_for(p):
    kw = dict(sigma_u=p["sigma_u"], sigma_eps=(p["q1"], p["q2"]), monitor="realized", fixed=SW.FIXED[p["who"]])
    if p["part"] == "L":
        kw.update(grid="nash", grid_width=0.3, price_grid="per_value")
    else:
        kw.update(PAPER)
    return G.make_params(**kw)


def analyze(records, sigma_u) -> dict:
    v = {"P1": "not run", "L1": "not run", "L": {}}
    idx = {r["who"]: r for r in records if r["part"] == "P"}
    if "pair" in idx:
        pr = idx["pair"]
        d = pr["delta"]["mean"]
        v["P1"] = "reproduced" if d >= 0.65 else "partly" if d >= 0.50 else "not reproduced"
        v["P"] = {"delta": d, "delta_se": pr["delta"]["se"], "converged": pr["converged_share"],
                  "sessions": pr["sessions"], "profit": list(pr["profit"])}
        if "only0" in idx:
            al = idx["only0"]
            v["P"]["collusive_part"] = al["restraint"][0] - (pr["restraint"][0] + pr["restraint"][1]) / 2
            v["P"]["learning_bias"] = 1 - al["restraint"][0]
    lidx = {SW.key(r): r for r in records if r["part"] == "L"}
    cells = []
    for su in sigma_u:
        ref = SW.decompose(lidx, su, (0.0, 0.0))
        d1, d2 = SW.decompose(lidx, su, (0.0, 1.0)), SW.decompose(lidx, su, (0.5, 0.5))
        if ref is None or d1 is None or d2 is None:
            continue
        inf = SW.informative(ref)
        rho1, rho2 = SW.rho(d1, ref), SW.rho(d2, ref)
        cell = ("holds" if rho1 <= RATIO else "fails") if inf else "uninformative"
        cells.append(cell)
        v["L"][str(su)] = {"c_ref": ref["both"]["c"], "c_ref_se": ref["both"]["c_se"],
                           "bias_ref": ref["both"]["b"], "converged_ref": ref["both"]["converged"],
                           "c_single_grid": SINGLE_GRID_C.get(float(su)),
                           "rho_unequal": rho1, "rho_noisy": rho2, "L1": cell,
                           "unequal": d1, "noisy": d2}
    inf = [c for c in cells if c != "uninformative"]
    if cells:
        v["L1"] = "uninformative" if not inf else ("holds" if all(c == "holds" for c in inf) else "fails")
    return v


def report_md(v) -> str:
    L = ["# Price grid per lagged value", "", "Rules: PREREGISTRATION.md, pre-registration 8.", "",
         f"- **P1 (baseline at the paper's sizes):** {v['P1']}"
         + (f" — Δ = {v['P']['delta']:.3f} ± {v['P']['delta_se']:.3f} (paper ≈ 0.75), "
            f"{v['P']['converged']:.0%} of {v['P']['sessions']} runs converged" if "P" in v else ""),
         f"- **L1 (main result with the finer price grid):** {v['L1']}", ""]
    if "P" in v and "collusive_part" in v["P"]:
        L += [f"Paper's sizes: collusive part {100 * v['P']['collusive_part']:.1f} %, "
              f"learning bias {100 * v['P']['learning_bias']:.1f} %, profit per speculator "
              f"{v['P']['profit'][0]:.1f} / {v['P']['profit'][1]:.1f}.", ""]
    if v["L"]:
        L += ["| σᵤ | c at (0,0) | ± | single grid | converged | (0,1) spec1 | (0,1) spec2 | (0.5,0.5) | ρ (0,1) | ρ (0.5,0.5) | L1 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for su, x in v["L"].items():
            L.append(f"| {su} | {100 * x['c_ref']:.1f} | {100 * x['c_ref_se']:.1f} | {100 * x['c_single_grid']:.1f} | "
                     f"{x['converged_ref']:.0%} | {100 * x['unequal']['spec1']['c']:+.1f} | "
                     f"{100 * x['unequal']['spec2']['c']:+.1f} | {100 * x['noisy']['both']['c']:+.1f} | "
                     f"{x['rho_unequal']:.2f} | {x['rho_noisy']:.2f} | {x['L1']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["L", "P"], choices=["L", "P"])
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, nargs="+", default=[0.1, 100.0])
    ap.add_argument("--out", default="price_grid")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--max-periods", type=float, default=None, help="smoke tests only")
    args = ap.parse_args()
    smoke = args.max_periods is not None
    if smoke:
        G.LAPTOP["max_periods"] = int(args.max_periods)
        PAPER["max_periods"] = int(args.max_periods)

    out = C.out_dir(args.out)
    path = out / "points.json"
    records = json.loads(path.read_text())["points"] if path.exists() else []
    if not args.analyze_only:
        threads = C.init_threads(args.threads)
        todo = [p for p in plan(args.part, args.sigma_u) if key(p) not in {key(r) for r in records}]
        print(f"{len(todo)} runs to do ({len(records)} already saved)", flush=True)
        for n, p in enumerate(todo, 1):
            n_s = args.sessions if (p["part"] == "L" or smoke) else P_SESSIONS[p["who"]]
            s, _ = G.run_point(params_for(p), np.arange(1, n_s + 1))
            G.print_point(f"[{n}/{len(todo)}] {p['part']} σu={p['sigma_u']:g} q=({p['q1']:g},{p['q2']:g}) {p['who']}", s)
            records.append({**p, "sessions": s["sessions"], "converged_share": s["converged_share"],
                            "restraint": s["restraint_vs_best_response"],
                            "restraint_se": s["restraint_vs_best_response_se"],
                            "delta": s["delta_all"], "periods_learning": s["periods_learning"],
                            "profit": s["profit_mean"], "aggressiveness_vs_nash": s["aggressiveness_vs_nash"],
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
