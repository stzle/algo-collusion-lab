"""
Final checks before writing (PREREGISTRATION.md, "Pre-registration 5").

  M  market quality for all 14 precision profiles (pairs): informativeness, price impact,
     mispricing, profits, aggressiveness, plus each profile's Nash and cartel benchmarks
  P  shock test at equal noise (0.125, 0.125), sigma_u = 0.1
  G  finer value grid (n_v = 10) for the four key profiles, pair + lone-learner controls

    python scripts/11_final_checks.py               # 50 runs, ~1.5 h on an M4
    python scripts/11_final_checks.py --part M      # one part only
    python scripts/11_final_checks.py --analyze-only

Saved after every run; rerunning skips finished runs.
Outputs in results/final/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

import _common as C
import _dgj_common as G


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SW = _load("sweep", "08_precision_sweep.py")
NI = _load("noise", "09_noise_vs_inequality.py")
SH = _load("shock", "10_shock_test.py")

M_PROFILES = sorted(set(NI.all_profiles()) | set(SW.profiles()))
P_PROFILE = (0.125, 0.125)
G_PROFILES = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.5, 0.5)]
N_V_FINE = 10


def plan(parts, sigma_u):
    pts = []
    if "M" in parts:
        pts += [dict(part="M", sigma_u=su, q1=p[0], q2=p[1], who="pair", n_v=6)
                for su in sigma_u for p in M_PROFILES]
    if "P" in parts:
        pts += [dict(part="P", sigma_u=0.1, q1=P_PROFILE[0], q2=P_PROFILE[1], who=w, n_v=6)
                for w in ("pair", "only0")]
    if "G" in parts:
        pts += [dict(part="G", sigma_u=su, q1=p[0], q2=p[1], who=w, n_v=N_V_FINE)
                for su in sigma_u for p in G_PROFILES for w in SW.cells(p)]
    return pts


def key(p):
    return (p["part"], float(p["sigma_u"]), float(p["q1"]), float(p["q2"]), p["who"], int(p["n_v"]))


def run_one(p, seeds, paths):
    prm = G.make_params(sigma_u=p["sigma_u"], sigma_eps=(p["q1"], p["q2"]), fixed=SW.FIXED[p["who"]],
                        grid="nash", grid_width=0.3, monitor="realized", n_v=p["n_v"])
    s, res = G.run_point(prm, seeds)
    rec = {**p, "sessions": len(seeds), "converged_share": s["converged_share"],
           "restraint": s["restraint_vs_best_response"], "restraint_se": s["restraint_vs_best_response_se"],
           "delta": s["delta_all"], "benchmarks": s["benchmarks"],
           "informativeness": s["informativeness"], "mispricing": s["mispricing"],
           "price_impact": C.summary_stats(np.array([mm[1] for mm in res["mm"]])),
           "profit_mean": s["profit_mean"], "chi_mean": res["chi"].mean(axis=0),
           "aggressiveness_vs_nash": s["aggressiveness_vs_nash"],
           "wall_clock_seconds": s["wall_clock_seconds"]}
    if p["part"] == "P":
        R = SH.responses(res, prm, len(seeds), paths)
        rec["R_t4"] = R[:, :, SH.T_AFTER, :]
        rec["paths"] = paths
    return s, rec


# ------------------------------------------------------------------ analysis

def _as_sweep(records, part, n_v):
    """Records of one part in the format of 08_precision_sweep.decompose."""
    return {SW.key(r): r for r in records if r["part"] == part and r["n_v"] == n_v}


def _verdict(cells):
    inf = [c for c in cells if c != "uninformative"]
    if not inf:
        return "uninformative"
    return "holds" if all(c == "holds" for c in inf) else "fails"


def analyze(records, sigma_u) -> dict:
    v = {"G1": "not run", "G2": "not run", "P5": "not run", "G_detail": {}, "M": {}}
    idx = _as_sweep(records, "G", N_V_FINE)
    g1, g2 = [], []
    for su in sigma_u:
        ref = SW.decompose(idx, su, (0.0, 0.0))
        ds = {p: SW.decompose(idx, su, p) for p in G_PROFILES[1:]}
        if ref is None or any(d is None for d in ds.values()):
            continue
        rho = {f"{p[0]:g},{p[1]:g}": SW.rho(d, ref) for p, d in ds.items()}
        informative = SW.informative(ref)
        c1 = ("holds" if max(rho["0,0.5"], rho["0,1"]) <= SW.RATIO else "fails") if informative else "uninformative"
        c2 = ("holds" if rho["0.5,0.5"] <= SW.RATIO else "fails") if informative else "uninformative"
        g1.append(c1); g2.append(c2)
        v["G_detail"][str(su)] = {"c_ref": ref["both"]["c"], "c_ref_se": ref["both"]["c_se"], "rho": rho,
                                  "G1": c1, "G2": c2,
                                  "cases": {f"{p[0]:g},{p[1]:g}": d for p, d in ds.items()}}
    if g1:
        v["G1"], v["G2"] = _verdict(g1), _verdict(g2)
    pidx = {SH.key(r): r for r in records if r["part"] == "P"}
    d = SH.punishment(pidx, 0.1, P_PROFILE)
    if d is not None:
        n = sum(x["punishes"] for x in d.values())
        v["P5"] = {2: "inequality specifically weakens punishment", 0: "noise alone removes punishment",
                   1: "mixed"}[n]
        v["P_detail"] = d
    for r in records:
        if r["part"] == "M":
            v["M"].setdefault(str(r["sigma_u"]), []).append({
                "profile": [r["q1"], r["q2"]], "delta": r["delta"]["mean"],
                "informativeness": r["informativeness"]["mean"], "price_impact": r["price_impact"]["mean"],
                "mispricing": r["mispricing"]["mean"], "profit": list(r["profit_mean"]),
                "lambda_nash": r["benchmarks"]["lambda_nash"], "lambda_cartel": r["benchmarks"]["lambda_cartel"]})
    return v


def report_md(v) -> str:
    L = ["# Final checks", "", "Rules: PREREGISTRATION.md, pre-registration 5.", "",
         f"- **G1 (main result with 10 value levels):** {v['G1']}",
         f"- **G2 (noise effect with 10 value levels):** {v['G2']}",
         f"- **P5 (punishment at equal noise 0.125):** {v['P5']}", ""]
    for su, g in v["G_detail"].items():
        L.append(f"## Finer grid, σᵤ = {su}: collusive part at (0,0) = {100 * g['c_ref']:.1f} % "
                 f"(± {100 * g['c_ref_se']:.1f}); ρ = " + ", ".join(f"{k}: {x:.2f}" for k, x in g["rho"].items()))
    if "P_detail" in v:
        L += ["", "## Punishment at (0.125, 0.125), σᵤ = 0.1", "",
              "| speculator | shock | pair | alone | P | ± | punishes |", "|---|---:|---:|---:|---:|---:|---|"]
        for spec, x in v["P_detail"].items():
            for k, lab in enumerate(SH.LABELS):
                L.append(f"| {spec} | {lab} | {100 * x['R_pair'][k]:+.2f} | {100 * x['R_alone'][k]:+.2f} | "
                         f"{100 * x['P'][k]:+.2f} | {100 * x['se'][k]:.2f} | "
                         f"{('yes' if x['punishes'] else 'no') if k == 3 else ''} |")
    for su, rows in v["M"].items():
        L += ["", f"## Market quality, σᵤ = {su} (pairs)", "",
              "| profile | Δ | informativeness | price impact | λ Nash | λ cartel | mispricing | profit 1 | profit 2 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sorted(rows, key=lambda r: (r["profile"][0] != 0 or r["profile"][1] == 0, r["profile"])):
            L.append(f"| {r['profile'][0]:g}, {r['profile'][1]:g} | {r['delta']:.3f} | {r['informativeness']:.3f} | "
                     f"{r['price_impact']:.5f} | {r['lambda_nash']:.5f} | {r['lambda_cartel']:.5f} | "
                     f"{r['mispricing']:.4f} | {r['profit'][0]:.2f} | {r['profit'][1]:.2f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["M", "P", "G"], choices=["M", "P", "G"])
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, nargs="+", default=[0.1, 100.0])
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--out", default="final")
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
        done = {key(r) for r in records if r["sessions"] == args.sessions}
        todo = [p for p in plan(args.part, args.sigma_u) if key(p) not in done]
        print(f"{len(todo)} runs to do ({len(done)} already saved), {args.sessions} sessions each", flush=True)
        for n, p in enumerate(todo, 1):
            s, rec = run_one(p, seeds, args.paths)
            G.print_point(f"[{n}/{len(todo)}] {p['part']} σu={p['sigma_u']:g} q=({p['q1']:g},{p['q2']:g}) "
                          f"{p['who']} n_v={p['n_v']}", s)
            records = [r for r in records if key(r) != key(p)]
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
