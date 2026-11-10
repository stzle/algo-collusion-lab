"""
Three checks of weak points (PREREGISTRATION.md, "Pre-registration 6").

  S  opponent fixed at its lone-learner level: measures the collusive part without the
     assumption behind the exploratory "phantom" correction
  H  learning rate alpha and exploration decay beta halved and doubled
  B  symmetric baseline at the paper's grid sizes and convergence rule

    python scripts/12_hygiene_checks.py              # 46 runs, about 1.5 h on an M4
    python scripts/12_hygiene_checks.py --part S     # one part only
    python scripts/12_hygiene_checks.py --analyze-only

Saved after every run; rerunning skips finished runs.
Outputs in results/hygiene/: points.json, verdicts.json, verdicts.md
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

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sweep", Path(__file__).with_name("08_precision_sweep.py"))
SW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SW)

S_PROFILES = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0), (0.0, 0.25), (0.0, 1.0)]
H_PROFILES = [(0.0, 0.0), (0.0, 1.0), (0.5, 0.5)]
H_SETTINGS = [("alpha", 0.005), ("alpha", 0.02), ("beta", 2.5e-7), ("beta", 1e-6)]
PAPER = dict(n_v=10, n_x=15, n_p=31, grid="benchmarks", conv_window=1_000_000,
             conv_max_explore=0.0, max_periods=2_000_000_000)
B_SESSIONS = {"pair": 50, "only0": 20}
DGJ_DELTA = 0.75
MIN_C, RATIO, MIN_DIFF, REBOUND = 0.03, 0.5, 0.01, 0.02


def reference_index() -> dict:
    pts = json.loads((ROOT / "reference_results" / "precision" / "points.json").read_text())["points"]
    return {SW.key(p): p for p in pts}


def solo_level(ref, su, prof, j) -> float:
    """Lone-learner restraint of speculator j = its aggressiveness as a multiple of Nash
    (the best response to a Nash opponent is the Nash strategy)."""
    sym = prof[0] == prof[1]
    rec = ref[(float(su), float(prof[0]), float(prof[1]), "only0" if sym else f"only{j}")]
    return float(rec["restraint"][0 if sym else j])


def plan(parts, sigma_u) -> list[dict]:
    pts = []
    if "S" in parts:
        for su in sigma_u:
            for p in S_PROFILES:
                for i in ((0,) if p[0] == p[1] else (0, 1)):
                    pts.append(dict(part="S", sigma_u=su, q1=p[0], q2=p[1], who=f"solo_level{i}", setting="base"))
    if "H" in parts:
        for name, val in H_SETTINGS:
            for p in H_PROFILES:
                for w in SW.cells(p):
                    pts.append(dict(part="H", sigma_u=0.1, q1=p[0], q2=p[1], who=w, setting=f"{name}={val:g}"))
    if "B" in parts:
        pts += [dict(part="B", sigma_u=0.1, q1=0.0, q2=0.0, who=w, setting="paper") for w in ("pair", "only0")]
    return pts


def key(p) -> tuple:
    return (p["part"], float(p["sigma_u"]), float(p["q1"]), float(p["q2"]), p["who"], p["setting"])


def params_for(p, ref):
    kw = dict(sigma_u=p["sigma_u"], sigma_eps=(p["q1"], p["q2"]), monitor="realized")
    if p["part"] == "S":
        i = int(p["who"][-1])
        m = solo_level(ref, p["sigma_u"], (p["q1"], p["q2"]), 1 - i)
        kw.update(grid="nash", grid_width=0.3, fixed=(None, m) if i == 0 else (m, None))
    elif p["part"] == "H":
        name, val = p["setting"].split("=")
        kw.update(grid="nash", grid_width=0.3, fixed=SW.FIXED[p["who"]], **{name: float(val)})
    else:
        kw.update(fixed=SW.FIXED[p["who"]], **PAPER)
    return G.make_params(**kw)


# ------------------------------------------------------------------ analysis

def _pair_r(ref, su, prof, i):
    rec = ref[(float(su), float(prof[0]), float(prof[1]), "pair")]
    if prof[0] == prof[1]:
        r, se = rec["restraint"], rec["restraint_se"]
        return (r[0] + r[1]) / 2, math.hypot(se[0], se[1]) / 2
    return rec["restraint"][i], rec["restraint_se"][i]


def clean_parts(records, ref, sigma_u) -> list[dict]:
    """Per (sigma_u, profile, speculator): measured, corrected and clean collusive part."""
    from lab import dgj_benchmarks as bmk
    idx = {key(r): r for r in records}
    v = bmk.value_grid(6, 1.0, 1.0)
    out = []
    for su in sigma_u:
        for prof in S_PROFILES:
            sym = prof[0] == prof[1]
            bm = bmk.benchmarks(v, prof, 500.0, 0.1, su, 1.0)
            Cm, Sm, lam, chiN = bm["C"], bm["S"], bm["nash"]["lam"], np.asarray(bm["nash"]["chi"])
            alone = [solo_level(ref, su, prof, j) for j in (0, 1)]
            r_mech = bmk.restraint(np.array(alone) * chiN, Cm, Sm, lam)["restraint"]
            for i in ((0,) if sym else (0, 1)):
                rec = idx.get(("S", float(su), float(prof[0]), float(prof[1]), f"solo_level{i}", "base"))
                if rec is None:
                    continue
                rp, rp_se = _pair_r(ref, su, prof, i)
                measured = alone[i] - rp
                corrected = measured - (alone[i] - r_mech[i])
                clean = rec["restraint"][i] - rp
                out.append({"sigma_u": su, "profile": list(prof), "speculator": "both" if sym else f"spec{i + 1}",
                            "measured": measured, "corrected": corrected, "clean": clean,
                            "clean_se": math.hypot(rec["restraint_se"][i], rp_se)})
    return out


def s_verdicts(rows, sigma_u) -> dict:
    v = {}
    dif = [r for r in rows if abs(r["measured"] - r["corrected"]) >= MIN_DIFF]
    if dif:
        near = sum(abs(r["clean"] - r["corrected"]) < abs(r["clean"] - r["measured"]) for r in dif)
        share = near / len(dif)
        v["S1"] = ("correction supported" if share >= 2 / 3 else
                   "correction rejected" if share <= 1 / 3 else "mixed")
        v["S1_detail"] = f"closer to corrected in {near} of {len(dif)} cases"
    else:
        v["S1"] = "not run" if not rows else "no case differs"

    def get(su, prof):
        return [r for r in rows if r["sigma_u"] == su and r["profile"] == list(prof)]
    s2, s3 = [], []
    for su in sigma_u:
        ref, asym, reb = get(su, (0.0, 0.0)), get(su, (0.0, 1.0)), get(su, (1.0, 1.0))
        if ref and len(asym) == 2:
            s2.append(ref[0]["clean"] >= MIN_C and max(a["clean"] for a in asym) <= RATIO * ref[0]["clean"])
        if reb:
            c, se = reb[0]["clean"], reb[0]["clean_se"]
            s3.append("real" if (c >= REBOUND and c > 2 * se) else "gone" if c < REBOUND else "unclear")
    v["S2"] = ("holds" if all(s2) else "fails") if len(s2) == len(sigma_u) else "not run"
    v["S3"] = (s3[0] if all(x == s3[0] for x in s3) and s3[0] != "unclear" else "unclear") if len(s3) == len(sigma_u) else "not run"
    return v


def h_verdicts(records) -> dict:
    out, cells = {"settings": {}}, []
    for name, val in H_SETTINGS:
        setting = f"{name}={val:g}"
        idx = {SW.key(r): r for r in records if r["part"] == "H" and r["setting"] == setting}
        ref = SW.decompose(idx, 0.1, (0.0, 0.0))
        d1, d2 = SW.decompose(idx, 0.1, (0.0, 1.0)), SW.decompose(idx, 0.1, (0.5, 0.5))
        if ref is None or d1 is None or d2 is None:
            continue
        inf = SW.informative(ref)
        rho1, rho2 = SW.rho(d1, ref), SW.rho(d2, ref)
        cell = ("holds" if rho1 <= RATIO else "fails") if inf else "uninformative"
        cells.append(cell)
        out["settings"][setting] = {"c_ref": ref["both"]["c"], "c_ref_se": ref["both"]["c_se"],
                                    "converged_ref": ref["both"]["converged"],
                                    "rho_unequal": rho1, "rho_noisy": rho2, "H1": cell}
    inf = [c for c in cells if c != "uninformative"]
    if not cells:
        out["H1"] = "not run"
    elif "fails" in inf:
        out["H1"] = "fails"
    elif len(inf) >= 3:
        out["H1"] = "robust"
    else:
        out["H1"] = "insufficient"
    return out


def b_verdict(records) -> dict:
    idx = {r["who"]: r for r in records if r["part"] == "B"}
    if "pair" not in idx:
        return {"B1": "not run"}
    pr = idx["pair"]
    d = pr["delta"]["mean"]
    out = {"B1": "reproduced" if d >= 0.65 else "partly" if d >= 0.50 else "not reproduced",
           "delta": d, "delta_se": pr["delta"]["se"], "converged": pr["converged_share"],
           "paper_delta": DGJ_DELTA, "sessions": pr["sessions"],
           "periods_median": pr["periods_learning"]["median"]}
    if "only0" in idx:
        al = idx["only0"]
        out["collusive_part"] = al["restraint"][0] - (pr["restraint"][0] + pr["restraint"][1]) / 2
        out["learning_bias"] = 1 - al["restraint"][0]
    return out


def analyze(records, sigma_u, ref=None) -> dict:
    ref = ref if ref is not None else reference_index()
    rows = clean_parts(records, ref, sigma_u)
    return {"S": {**s_verdicts(rows, sigma_u), "rows": rows}, "H": h_verdicts(records), "B": b_verdict(records)}


def report_md(v) -> str:
    S, H, B = v["S"], v["H"], v["B"]
    L = ["# Checks of weak points", "", "Rules: PREREGISTRATION.md, pre-registration 6.", "",
         f"- **S1 (phantom correction):** {S['S1']}" + (f" — {S['S1_detail']}" if "S1_detail" in S else ""),
         f"- **S2 (main result with the measured clean part):** {S['S2']}",
         f"- **S3 (rebound at equal noise 1):** {S['S3']}",
         f"- **H1 (main result under other α, β):** {H['H1']}",
         f"- **B1 (baseline at the paper's grid sizes):** {B['B1']}"
         + (f" — Δ = {B['delta']:.3f} ± {B['delta_se']:.3f} (paper ≈ {B['paper_delta']}), "
            f"{B['converged']:.0%} of {B['sessions']} runs converged" if "delta" in B else ""), ""]
    if "collusive_part" in B:
        L += [f"At the paper's scale: collusive part {100 * B['collusive_part']:.1f} %, "
              f"learning bias {100 * B['learning_bias']:.1f} %.", ""]
    if S["rows"]:
        L += ["| σᵤ | profile | speculator | measured | corrected | clean | ± |", "|---:|---|---|---:|---:|---:|---:|"]
        for r in S["rows"]:
            L.append(f"| {r['sigma_u']:g} | {r['profile'][0]:g}, {r['profile'][1]:g} | {r['speculator']} | "
                     f"{100 * r['measured']:+.1f} | {100 * r['corrected']:+.1f} | {100 * r['clean']:+.1f} | {100 * r['clean_se']:.1f} |")
        L.append("")
    if H["settings"]:
        L += ["| setting | c at (0,0) | ± | converged | ρ (0,1) | ρ (0.5,0.5) | H1 |", "|---|---:|---:|---:|---:|---:|---|"]
        for k, x in H["settings"].items():
            L.append(f"| {k} | {100 * x['c_ref']:.1f} | {100 * x['c_ref_se']:.1f} | {x['converged_ref']:.0%} | "
                     f"{x['rho_unequal']:.2f} | {x['rho_noisy']:.2f} | {x['H1']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["S", "H", "B"], choices=["S", "H", "B"])
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, nargs="+", default=[0.1, 100.0])
    ap.add_argument("--out", default="hygiene")
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
    ref = reference_index()
    if not args.analyze_only:
        threads = C.init_threads(args.threads)
        todo = [p for p in plan(args.part, args.sigma_u) if key(p) not in {key(r) for r in records}]
        print(f"{len(todo)} runs to do ({len(records)} already saved)", flush=True)
        for n, p in enumerate(todo, 1):
            n_s = args.sessions if (p["part"] != "B" or smoke) else B_SESSIONS[p["who"]]
            s, _ = G.run_point(params_for(p, ref), np.arange(1, n_s + 1))
            G.print_point(f"[{n}/{len(todo)}] {p['part']} σu={p['sigma_u']:g} q=({p['q1']:g},{p['q2']:g}) "
                          f"{p['who']} {p['setting']}", s)
            records.append({**p, "sessions": s["sessions"], "converged_share": s["converged_share"],
                            "restraint": s["restraint_vs_best_response"],
                            "restraint_se": s["restraint_vs_best_response_se"],
                            "delta": s["delta_all"], "periods_learning": s["periods_learning"],
                            "aggressiveness_vs_nash": s["aggressiveness_vs_nash"],
                            "wall_clock_seconds": s["wall_clock_seconds"]})
            C.save_json(path, {"machine": C.machine_info(threads), "points": records})
    v = analyze(records, args.sigma_u, ref)
    C.save_json(out / "verdicts.json", v)
    md = report_md(v)
    (out / "verdicts.md").write_text(md)
    print()
    print(md)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
