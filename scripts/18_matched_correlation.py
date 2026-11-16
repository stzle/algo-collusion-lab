"""
Correlation is not enough (PREREGISTRATION.md, "Pre-registration 12"). At the same measured
signal correlation, does unequal precision (0, q) give less coordination than equal
precision with partly shared noise?

The two levels are the measured correlations of the equal profile (0.5, 0.5) at w = 0.5 and
w = 0.75. For each, q is chosen so that (0, q) with independent noise has the same measured
correlation. Only the unequal arm is run (pair + both lone learners, sigma_u = 0.1, seeds
21-40); the equal arm is read from results/measure_checks (same seeds).

    python scripts/18_matched_correlation.py                # 6 points, about 8 min on an M4
    python scripts/18_matched_correlation.py --analyze-only
    python scripts/18_matched_correlation.py --sessions 2 --max-periods 1e6   # smoke -> matched_correlation_smoke

Saved after every point; rerunning skips finished points.
Outputs in results/matched_correlation/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import _common as C
import _dgj_common as G
from lab import dgj, dgj_benchmarks as bmk


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC = _load("measure", "16_measure_checks.py")
S = _load("sweep", "08_precision_sweep.py")

SEEDS = MC.SEEDS
SIGMA_U = 0.1
Q_EQUAL = 0.5
LEVELS = [0.5, 0.75]           # shared parts w of the equal arm whose correlations are matched
CAP = int(2e8)
_P = dgj.DGJParams()
V_GRID = bmk.value_grid(G.LAPTOP["n_v"], _P.v_bar, _P.sigma_v)


def measured_correlation(v_grid, q1, q2, shared) -> float:
    """Correlation of the two discretised signal values v_grid[signal] under a uniform value prior."""
    P = bmk.joint_signal_matrix(v_grid, q1, q2, shared).mean(0)     # P[a, b]
    p1, p2 = P.sum(1), P.sum(0)
    m1, m2 = p1 @ v_grid, p2 @ v_grid
    cov = v_grid @ P @ v_grid - m1 * m2
    return float(cov / np.sqrt((p1 @ v_grid ** 2 - m1 ** 2) * (p2 @ v_grid ** 2 - m2 ** 2)))


def match_q(v_grid, target, lo=1e-3, hi=3.0) -> float:
    """q such that (0, q) with independent noise has the measured correlation `target`."""
    return brentq(lambda q: measured_correlation(v_grid, 0.0, q, 0.0) - target, lo, hi, xtol=1e-10)


def levels(v_grid=V_GRID) -> dict:
    out = {}
    for w in LEVELS:
        target = measured_correlation(v_grid, Q_EQUAL, Q_EQUAL, w)
        q = match_q(v_grid, target)
        out[f"{w:g}"] = {"w": w, "corr_target": target, "q": q, "corr_achieved": measured_correlation(v_grid, 0.0, q, 0.0)}
    return out


def plan(lv=None) -> list[dict]:
    lv = lv or levels()
    return [dict(sigma_u=SIGMA_U, level=x["w"], q1=0.0, q2=x["q"], corr_target=x["corr_target"],
                 corr_achieved=x["corr_achieved"], who=who) for x in lv.values() for who in ("pair", "only0", "only1")]


def key(p):
    return (float(p["level"]), p["who"])


def params_for(p, records=None, max_periods=None):
    return G.make_params(sigma_u=p["sigma_u"], sigma_eps=(p["q1"], p["q2"]), fixed=S.FIXED[p["who"]],
                         grid="nash", grid_width=0.3, monitor="realized", price_grid="per_value",
                         max_periods=int(max_periods or CAP))


# ------------------------------------------------------------------ analysis

def c_unequal_sessions(pair, only0, only1) -> np.ndarray:
    """Per seed: mean over the two speculators of r_only_i[i] - r_pair[i]."""
    if not MC.same_seeds(pair, only0, only1):
        raise ValueError("unequal arm run on different seeds")
    P = MC.restr(pair)
    return ((MC.restr(only0)[:, 0] - P[:, 0]) + (MC.restr(only1)[:, 1] - P[:, 1])) / 2


def a1(d, se) -> str:
    return "lower" if d > 2 * se else "not lower"


def overall(labels: list[str]) -> str:
    n = sum(x == "lower" for x in labels)
    return {2: "correlation is not enough", 1: "partly", 0: "not supported"}[n]


def analyze(records, equal_records) -> dict:
    idx = {key(r): r for r in records}
    eq = {MC.key(r): r for r in equal_records}
    v = {"levels": {}, "A1": "not run"}
    labels = []
    for w in LEVELS:
        un = [idx.get((w, who)) for who in ("pair", "only0", "only1")]
        ep, ea = eq.get((SIGMA_U, w, "pair", MC.CAP)), eq.get((SIGMA_U, w, "only0", MC.CAP))
        e = {}
        if None not in un:
            cu = c_unequal_sessions(*un)
            e.update(q=un[0]["q2"], corr_target=un[0]["corr_target"], corr_achieved=un[0]["corr_achieved"])
            e["c_unequal"], e["c_unequal_se"] = MC.mean_se(cu)
            e["c_unequal_spec"] = [MC.mean_se(MC.restr(un[1 + i])[:, i] - MC.restr(un[0])[:, i])[0] for i in (0, 1)]
        if ep is not None and ea is not None:
            ce = MC.c_sessions(ep, ea)
            e["c_equal"], e["c_equal_se"] = MC.mean_se(ce)
        if None not in un and ep is not None and ea is not None:
            if not MC.same_seeds(un[0], ep):
                raise ValueError("equal and unequal arm run on different seeds")
            e["d"], e["d_se"] = MC.paired(ce, cu)
            e["A1"] = a1(e["d"], e["d_se"])
            labels.append(e["A1"])
        v["levels"][f"{w:g}"] = e
    if len(labels) == len(LEVELS):
        v["A1"] = overall(labels)
    return v


def report_md(v) -> str:
    L = ["# Matched correlation", "",
         "Rules: PREREGISTRATION.md, pre-registration 12. c = collusive part (% of best response), σᵤ = 0.1, "
         "seeds 21–40; d = c(equal) − c(unequal), paired SE by seed.", "",
         f"- **A1 (correlation is not enough):** {v['A1']}", "",
         "| level w | corr target | q | corr achieved | c equal | SE | c unequal | SE | spec 1 / spec 2 | d | SE | A1 |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    f = lambda e, k: f"{100 * e[k]:+.2f}" if k in e else "–"
    g = lambda e, k: f"{100 * e[k]:.2f}" if k in e else "–"
    for w, e in v["levels"].items():
        sp = " / ".join(f"{100 * x:+.2f}" for x in e["c_unequal_spec"]) if "c_unequal_spec" in e else "–"
        q = f"{e['corr_target']:.4f} | {e['q']:.4f} | {e['corr_achieved']:.4f}" if "q" in e else "– | – | –"
        L.append(f"| {w} | {q} | {f(e, 'c_equal')} | {g(e, 'c_equal_se')} | {f(e, 'c_unequal')} | "
                 f"{g(e, 'c_unequal_se')} | {sp} | {f(e, 'd')} | {g(e, 'd_se')} | {e.get('A1', '–')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=len(SEEDS))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="matched_correlation")
    ap.add_argument("--equal-from", default=None, help="folder of the equal arm (default measure_checks[_smoke])")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--max-periods", type=float, default=None, help="smoke tests only")
    args = ap.parse_args()

    name = MC.smoke_out(args)
    out = C.out_dir(name)
    path = out / "points.json"
    records = json.loads(path.read_text())["points"] if path.exists() else []
    lv = levels()
    for r in records:     # q is recomputed each time; it must not have moved
        if not np.isclose(r["q2"], lv[f"{r['level']:g}"]["q"], atol=1e-8):
            raise ValueError(f"saved q {r['q2']} differs from the matched q now")
    print("Levels: " + ", ".join(f"w = {w}: corr {x['corr_target']:.4f} -> q = {x['q']:.4f} "
                                  f"(corr {x['corr_achieved']:.4f})" for w, x in lv.items()), flush=True)
    if not args.analyze_only:
        threads = C.init_threads(args.threads)
        seeds = SEEDS[:args.sessions]
        todo = MC.pending(plan(lv), key, records, len(seeds))
        print(f"{len(todo)} points to run ({len(records)} already saved), {len(seeds)} sessions each", flush=True)
        records = MC.run(todo, key, lambda p, rs: params_for(p, rs, args.max_periods), records, path, seeds,
                         threads, lambda p: f"level w={p['level']:g} q=(0,{p['q2']:.3f}) {p['who']}")
    eq_dir = C.RESULTS / (args.equal_from or ("measure_checks_smoke" if name.endswith("_smoke") else "measure_checks"))
    eq_path = eq_dir / "points.json"
    equal = json.loads(eq_path.read_text())["points"] if eq_path.exists() else []
    if not equal:
        print(f"No equal arm in {eq_dir}: run 16_measure_checks.py first")
    v = analyze(records, equal)
    v["equal_from"] = str(eq_dir.relative_to(C.ROOT))
    C.save_json(out / "verdicts.json", v)
    md = report_md(v)
    (out / "verdicts.md").write_text(md)
    print()
    print(md)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
