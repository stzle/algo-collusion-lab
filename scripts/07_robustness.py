"""
Robustness of the first results, with the decision rules of PREREGISTRATION.md.

Three stages (see PREREGISTRATION.md, "Design"):

  grid     equal action grid at widths 0.2, 0.3, 0.4 x own Nash, variant A
  monitor  width 0.3, variant B (speculators monitor their own past signal)
  caps     width 0.3, variant A, period caps 1, 2, 4 x 10^8, same seeds

Every stage runs, per noise-trading level sigma_u, the pair and the fixed-opponent
controls for equal precision (q = 0) and for each unequal profile q. From those it
splits each speculator's restraint into a collusive part (only when both learn) and a
learning bias (also against a fixed Nash opponent), then checks rules R1, R2, S1 and
the H3 proxy.

    python scripts/07_robustness.py                      # everything: 92 points, ~2.5 h on an M4
    python scripts/07_robustness.py --stage grid
    python scripts/07_robustness.py --stage caps --caps 1e8 2e8 4e8
    python scripts/07_robustness.py --analyze-only       # verdicts from saved results

Results are saved after every point; a rerun skips points that are already there, so an
interrupted run can simply be started again.

Outputs in results/robustness/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

import _common as C
import _dgj_common as G

# who learns: (fixed tuple, agent whose restraint the control measures)
WHO = {
    "pair": (None, None),
    "alone": (None, 1.0),            # equal precision: speculator 1 learns, 2 fixed at Nash
    "informed_only": (None, 1.0),    # unequal: informed (1) learns, noisy (2) fixed
    "noisy_only": (1.0, None),       # unequal: noisy (2) learns, informed (1) fixed
}
DEFAULT_CAP = G.LAPTOP["max_periods"]

# thresholds from PREREGISTRATION.md
MIN_C_SYM = 0.03         # R1: a symmetric collusive part must exist
FRAGILE_RATIO = 0.5      # R1: asymmetric collusive part at most half the symmetric one
CAP_TOL = 0.01           # R2: max move in mean restraint between caps
MIN_BIAS = 0.03          # S1
AB_MIN = 0.02            # A vs B: minimum difference worth reporting


# --------------------------------------------------------------------------- plan

def cells(noise) -> list[tuple[float, str]]:
    plan = [(0.0, "pair"), (0.0, "alone")]
    plan += [(q, who) for q in noise for who in ("pair", "informed_only", "noisy_only")]
    return plan


def plan_points(stages, sigma_u, noise, widths, caps) -> list[dict]:
    pts = []
    for st in stages:
        for su in sigma_u:
            if st == "grid":
                groups = [(w, "realized", DEFAULT_CAP) for w in widths]
            elif st == "monitor":
                groups = [(0.3, "own_signal", DEFAULT_CAP)]
            elif st == "caps":
                groups = [(0.3, "realized", int(c)) for c in caps]
            else:
                raise ValueError(st)
            for w, mon, cap in groups:
                for q, who in cells(noise):
                    if mon == "own_signal" and q == 0.0:
                        continue        # identical to variant A when the signal is exact
                    pts.append(dict(sigma_u=su, width=w, monitor=mon, cap=cap, noise=q, who=who))
    # drop duplicates (e.g. width 0.3 / cap 2e8 / variant A appears in grid and caps)
    seen, out = set(), []
    for p in pts:
        k = key(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def key(p: dict) -> tuple:
    return (float(p["sigma_u"]), round(float(p["width"]), 4), p["monitor"], int(p["cap"]),
            float(p["noise"]), p["who"])


# ----------------------------------------------------------------------- analysis

def _index(records):
    return {key(r): r for r in records}


def _r(idx, su, w, mon, cap, q, who, agent):
    if mon == "own_signal" and q == 0.0:
        mon = "realized"        # with an exact signal, own past signal = realized value
    rec = idx.get((float(su), round(float(w), 4), mon, int(cap), float(q), who))
    if rec is None:
        return None
    return float(rec["restraint"][agent]), float(rec["restraint_se"][agent]), float(rec["converged_share"])


def _diff(a, b):
    """(a - b, se) for two (mean, se) pairs, treated as independent."""
    return a[0] - b[0], math.hypot(a[1], b[1])


def decompose(idx, su, w, mon, cap, noise) -> dict | None:
    """Collusive part c and learning bias b per case; None if a cell is missing."""
    pa = [_r(idx, su, w, mon, cap, 0.0, "pair", a) for a in (0, 1)]
    al = _r(idx, su, w, mon, cap, 0.0, "alone", 0)
    if None in pa or al is None:
        return None
    pair_mean = ((pa[0][0] + pa[1][0]) / 2, math.hypot(pa[0][1], pa[1][1]) / 2)
    c, c_se = _diff(al, pair_mean)
    out = {"sym": {"c": c, "c_se": c_se, "b": 1 - al[0], "b_se": al[1],
                   "converged": min(pa[0][2], al[2])}}
    for q in noise:
        for name, who, agent in (("informed", "informed_only", 0), ("noisy", "noisy_only", 1)):
            pr = _r(idx, su, w, mon, cap, q, "pair", agent)
            fx = _r(idx, su, w, mon, cap, q, who, agent)
            if pr is None or fx is None:
                return None
            c, c_se = _diff(fx, pr)
            out[f"{name}_q{q:g}"] = {"c": c, "c_se": c_se, "b": 1 - fx[0], "b_se": fx[1],
                                     "converged": min(pr[2], fx[2]), "r_pair": pr[0], "r_fixed": fx[0]}
    return out


def r1_cell(d: dict) -> str:
    """'holds' / 'fails' / 'uninformative' for one (sigma_u, width)."""
    s = d["sym"]
    if not (s["c"] >= MIN_C_SYM and s["c"] > 2 * s["c_se"]):
        return "uninformative"
    asym = [v["c"] for k, v in d.items() if k != "sym"]
    return "holds" if all(c <= FRAGILE_RATIO * s["c"] for c in asym) else "fails"


def r1_overall(per_width: dict) -> str:
    """per_width: {width: [cell verdicts over sigma_u]} -> overall R1 verdict."""
    width_v = {}
    for w, vs in per_width.items():
        if "fails" in vs:
            width_v[w] = "fails"
        elif "holds" in vs:
            width_v[w] = "holds"
        else:
            width_v[w] = "uninformative"
    if width_v.get(0.3) == "fails":
        return "fails"
    informative = [v for v in width_v.values() if v != "uninformative"]
    if len(informative) >= 2 and all(v == "holds" for v in informative):
        return "robust"
    if "holds" in informative and "fails" in informative:
        return "grid-dependent"
    return "insufficient"


def s1_ok(d: dict) -> bool:
    return all(v["b"] >= MIN_BIAS and v["b"] > 2 * v["b_se"]
               for k, v in d.items() if k.startswith("noisy_"))


def h3_ok(d: dict) -> bool:
    b_ref = max(d["sym"]["b"], MIN_BIAS)
    return r1_cell(d) == "holds" and all(v["b"] >= 0.5 * b_ref for k, v in d.items() if k.startswith("noisy_"))


def r2_check(idx, su, noise, cap_lo, cap_hi) -> dict | None:
    """Max move in mean restraint between two caps over all capped cells."""
    moves, n_capped = [], 0
    for q, who in cells(noise):
        lo = idx.get((float(su), 0.3, "realized", int(cap_lo), float(q), who))
        hi = idx.get((float(su), 0.3, "realized", int(cap_hi), float(q), who))
        if lo is None or hi is None:
            return None
        if hi["converged_share"] >= 0.5:
            continue
        n_capped += 1
        agents = (0, 1) if who == "pair" else ((0,) if who in ("alone", "informed_only") else (1,))
        for a in agents:
            moves.append(abs(hi["restraint"][a] - lo["restraint"][a]))
    return {"max_move": max(moves) if moves else 0.0, "capped_cells": n_capped,
            "ok": (max(moves) if moves else 0.0) <= CAP_TOL}


def analyze(records, sigma_u, noise, widths, caps) -> dict:
    idx = _index(records)
    v = {"rules": {"MIN_C_SYM": MIN_C_SYM, "FRAGILE_RATIO": FRAGILE_RATIO, "CAP_TOL": CAP_TOL,
                   "MIN_BIAS": MIN_BIAS, "AB_MIN": AB_MIN},
         "decomposition": [], "R1": {}, "S1": None, "H3_proxy": None, "R2": {}, "AB": []}

    # grid stage: R1, S1, H3
    per_width, s1, h3 = {}, [], []
    for w in widths:
        for su in sigma_u:
            d = decompose(idx, su, w, "realized", DEFAULT_CAP, noise)
            if d is None:
                continue
            cell = r1_cell(d)
            per_width.setdefault(round(w, 4), []).append(cell)
            v["decomposition"].append({"sigma_u": su, "width": w, "monitor": "realized",
                                       "cap": DEFAULT_CAP, "R1": cell, "cases": d})
            s1.append(s1_ok(d))
            if round(w, 4) == 0.3:
                h3.append(h3_ok(d))
    v["R1"] = {"per_width": {str(k): x for k, x in per_width.items()},
               "verdict": r1_overall(per_width) if per_width else "not run"}
    v["S1"] = ("holds" if all(s1) else "fails") if s1 else "not run"
    v["H3_proxy"] = ("supported" if len(h3) == len(sigma_u) and all(h3) else
                     "not supported" if h3 else "not run")

    # caps stage: R2
    cs = sorted(int(c) for c in caps)
    for su in sigma_u:
        res = {}
        for lo, hi in zip(cs, cs[1:]):
            chk = r2_check(idx, su, noise, lo, hi)
            if chk is None:
                continue
            d_lo = decompose(idx, su, 0.3, "realized", lo, noise)
            d_hi = decompose(idx, su, 0.3, "realized", hi, noise)
            chk["R1_same"] = (d_lo is not None and d_hi is not None and r1_cell(d_lo) == r1_cell(d_hi))
            res[f"{lo:.0e}->{hi:.0e}"] = chk
            for cap, d in ((lo, d_lo), (hi, d_hi)):
                listed = {(e["sigma_u"], e["width"], e["monitor"], e["cap"]) for e in v["decomposition"]}
                if d is not None and (su, 0.3, "realized", cap) not in listed:
                    v["decomposition"].append({"sigma_u": su, "width": 0.3, "monitor": "realized",
                                               "cap": cap, "R1": r1_cell(d), "cases": d})
        if res:
            last_k = list(res)[-1]
            final = res[last_k]
            top = max(cs)
            status = ("pass" if final["ok"] and final["R1_same"] else "fail")
            if status == "pass" and top < 4e8:
                status = "provisional pass"
            res["verdict"] = status
            v["R2"][str(su)] = res

    # monitor stage: A vs B
    for su in sigma_u:
        da = decompose(idx, su, 0.3, "realized", DEFAULT_CAP, noise)
        db = decompose(idx, su, 0.3, "own_signal", DEFAULT_CAP, noise)
        if da is None or db is None:
            continue
        v["decomposition"].append({"sigma_u": su, "width": 0.3, "monitor": "own_signal",
                                   "cap": DEFAULT_CAP, "R1": r1_cell(db), "cases": db})
        for case in da:
            diff = db[case]["c"] - da[case]["c"]
            se = math.hypot(db[case]["c_se"], da[case]["c_se"])
            v["AB"].append({"sigma_u": su, "case": case, "c_A": da[case]["c"], "c_B": db[case]["c"],
                            "diff": diff, "se": se,
                            "differs": abs(diff) > max(AB_MIN, 2 * se)})
    return v


# ------------------------------------------------------------------------ report

def _pct(x):
    return f"{100 * x:+.1f}"


def report_md(v: dict) -> str:
    L = ["# Robustness verdicts", "",
         "Rules: PREREGISTRATION.md. c = collusive part, b = learning bias, both in % of the "
         "best response; conv = share of runs that converged (the rest hit the cap).", ""]
    L += [f"- **R1 (H1 survives grid width):** {v['R1']['verdict']}  — per width: {v['R1']['per_width']}",
          f"- **S1 (bias of the worse-informed speculator):** {v['S1']}",
          f"- **H3 proxy:** {v['H3_proxy']}"]
    for su, res in v["R2"].items():
        L.append(f"- **R2 (caps), σᵤ = {su}:** {res['verdict']}  — " + "; ".join(
            f"{k}: max move {x['max_move']:.3f} over {x['capped_cells']} capped cells, R1 same: {x['R1_same']}"
            for k, x in res.items() if k != "verdict"))
    if v["AB"]:
        n = sum(a["differs"] for a in v["AB"])
        L.append(f"- **A vs B:** difference in {n} of {len(v['AB'])} cases")
    L += ["", "| σᵤ | width | monitor | cap | case | c | ± | b | ± | conv | R1 |",
          "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for e in v["decomposition"]:
        for case, d in e["cases"].items():
            L.append(f"| {e['sigma_u']:g} | {e['width']:g} | {e['monitor']} | {e['cap']:.0e} | {case} | "
                     f"{_pct(d['c'])} | {100 * d['c_se']:.1f} | {_pct(d['b'])} | {100 * d['b_se']:.1f} | "
                     f"{d['converged']:.0%} | {e['R1'] if case == 'sym' else ''} |")
    if v["AB"]:
        L += ["", "| σᵤ | case | c (A) | c (B) | B − A | ± | differs |", "|---:|---|---:|---:|---:|---:|---|"]
        for a in v["AB"]:
            L.append(f"| {a['sigma_u']:g} | {a['case']} | {_pct(a['c_A'])} | {_pct(a['c_B'])} | "
                     f"{_pct(a['diff'])} | {100 * a['se']:.1f} | {'yes' if a['differs'] else 'no'} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["grid", "monitor", "caps"],
                    choices=["grid", "monitor", "caps"])
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sigma-u", type=float, nargs="+", default=[0.1, 100.0])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--widths", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    ap.add_argument("--caps", type=float, nargs="+", default=[1e8, 2e8, 4e8])
    ap.add_argument("--max-periods", type=float, default=None,
                    help="cap for the grid and monitor stages (default 2e8; smaller only for smoke tests)")
    ap.add_argument("--only", nargs="+", default=None, choices=list(WHO),
                    help="run only these cell types (e.g. --only pair)")
    ap.add_argument("--out", default="robustness", help="folder name under results/")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.max_periods:
        global DEFAULT_CAP
        DEFAULT_CAP = int(args.max_periods)

    out = C.out_dir(args.out)
    path = out / "points.json"
    records = json.loads(path.read_text())["points"] if path.exists() else []

    if not args.analyze_only:
        threads = C.init_threads(args.threads)
        seeds = np.arange(1, args.sessions + 1)
        done = {key(r) for r in records if r["sessions"] == args.sessions}
        todo = [p for p in plan_points(args.stage, args.sigma_u, args.noise, args.widths, args.caps)
                if key(p) not in done and (args.only is None or p["who"] in args.only)]
        print(f"{len(todo)} points to run ({len(done)} already saved), {args.sessions} runs each", flush=True)
        for n, p in enumerate(todo, 1):
            prm = G.make_params(sigma_u=p["sigma_u"], sigma_eps=(0.0, p["noise"]), fixed=WHO[p["who"]],
                                grid="nash", grid_width=p["width"], monitor=p["monitor"],
                                max_periods=p["cap"])
            s, _ = G.run_point(prm, seeds)
            label = (f"[{n}/{len(todo)}] σu={p['sigma_u']:g} w={p['width']:g} {p['monitor'][:4]} "
                     f"cap={p['cap']:.0e} q={p['noise']:g} {p['who']}")
            G.print_point(label, s)
            records = [r for r in records if key(r) != key(p)]
            records.append({**p, "sessions": s["sessions"], "converged_share": s["converged_share"],
                            "restraint": s["restraint_vs_best_response"],
                            "restraint_se": s["restraint_vs_best_response_se"],
                            "delta": s["delta_all"], "periods_learning": s["periods_learning"],
                            "wall_clock_seconds": s["wall_clock_seconds"]})
            C.save_json(path, {"machine": C.machine_info(threads), "points": records})

    v = analyze(records, args.sigma_u, args.noise, args.widths, args.caps)
    v["decomposition"].sort(key=lambda e: (e["sigma_u"], e["monitor"] != "realized", e["width"], e["cap"]))
    C.save_json(out / "verdicts.json", v)
    md = report_md(v)
    (out / "verdicts.md").write_text(md)
    print()
    print(md)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
