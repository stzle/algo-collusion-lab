"""
Small gap check (PREREGISTRATION.md, pre-registration 14): why the baseline at the paper's
sizes stays at Δ = 0.42 against the paper's ≈ 0.75. Runs part P's baseline again with the
same seeds (per-session values; must reproduce the saved mean) and one candidate that
changes a single detail. Candidates (ranked list: private/reports/2026-11-16 DGJ gap
candidates.md); only those that need no change to lab/dgj.py:

  learn_longer   every session keeps learning for exactly 10^10 periods (the paper's p. 24
                 read as "all sessions run until all have converged")
  cliff          diagnostic: σᵤ = 0.368 (ln σᵤ = -1), where the paper's Fig. 2A has fallen
                 to about 0.45; flat here = the gap is in learning, a cliff = in monitoring
  explore_more   β = 1e-7 instead of 5e-7 (exploration lasts about 5x longer)

    python scripts/20_gap_check.py --candidate learn_longer    # baseline + candidate, ~1.5 h
    python scripts/20_gap_check.py --candidate cliff           # ~30-45 min
    python scripts/20_gap_check.py --analyze-only

Outputs in results/gap_check/: points.json, verdicts.json, verdicts.md
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
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC = _load("measure_checks", "16_measure_checks.py")
PA = _load("price_grid_audit", "13_price_grid_audit.py")

SEEDS = np.arange(1, 13)
CANDIDATES = {"learn_longer": dict(conv_window=20_000_000_000),   # never reached: learning stops at the cap
              "cliff": dict(sigma_u=0.368),
              "explore_more": dict(beta=1e-7)}
G1_FULL, G1_PART = 0.65, 0.50
CLIFF_DROP = 0.10              # diagnostic only: "cliff" if Δ falls by more than max(0.10, 2 SE)


def params_for(name, max_periods=None):
    kw = dict(PA.PAPER, sigma_u=0.1, sigma_eps=(0.0, 0.0), monitor="realized", fixed=(None, None))
    kw.update(CANDIDATES.get(name, {}))
    if max_periods:
        kw["max_periods"] = int(max_periods)
    return G.make_params(**kw)


def reference_check(rec) -> str:
    if rec["sessions"] != len(SEEDS) or rec["max_periods"] != PA.PAPER["max_periods"]:
        return "not comparable (sessions or cap differ)"
    ref = Path(C.RESULTS / "price_grid" / "points.json")
    if not ref.exists():
        return "reference file missing"
    hits = [r for r in json.loads(ref.read_text())["points"] if r["part"] == "P" and r["who"] == "pair"]
    if not hits:
        return "reference point missing"
    ok = abs(rec["delta"]["mean"] - hits[0]["delta"]["mean"]) < 1e-9
    return "matches" if ok else f"MISMATCH ({rec['delta']['mean']:.6f} vs {hits[0]['delta']['mean']:.6f})"


def g1(level, d, se) -> str:
    if d > 2 * se and level >= G1_FULL:
        return "reproduced"
    if d > 2 * se and level >= G1_PART:
        return "partly"
    return "not reproduced"


def cliff(d, se) -> str:
    return "cliff" if -d > max(CLIFF_DROP, 2 * se) else "flat"


def delta_sessions(rec) -> np.ndarray:
    return np.asarray(rec["per_session"]["delta"], dtype=float)


def analyze(records) -> dict:
    idx = {r["candidate"]: r for r in records}
    base = idx.get("baseline")
    v = {"baseline": None, "candidates": {}}
    if base is None:
        return v
    v["baseline"] = {"delta": MC.mean_se(delta_sessions(base)), "reference": reference_check(base),
                     **MC.conv_learn(base)}
    for name in CANDIDATES:
        rec = idx.get(name)
        if rec is None:
            continue
        if not MC.same_seeds(base, rec):
            raise ValueError("candidate and baseline were run on different seeds")
        level = MC.mean_se(delta_sessions(rec))
        d, se = MC.paired(delta_sessions(rec), delta_sessions(base))
        out = {"delta": level, "change": d, "change_se": se, "moves_delta": bool(abs(d) > 2 * se),
               **MC.conv_learn(rec)}
        out["verdict"] = cliff(d, se) if name == "cliff" else g1(level[0], d, se)
        v["candidates"][name] = out
    return v


def report_md(v) -> str:
    L = ["# Small gap check", "", "Rules: PREREGISTRATION.md, pre-registration 14. Paper's sizes, σᵤ = 0.1 unless "
         "stated, seeds 1–12; changes paired by seed.", ""]
    b = v["baseline"]
    if b is None:
        return "\n".join(L + ["Baseline not run yet."]) + "\n"
    L.append(f"- **Baseline (part P rerun):** Δ = {b['delta'][0]:.3f} ± {b['delta'][1]:.3f}; "
             f"reference check: {b['reference']}")
    for name, x in v["candidates"].items():
        L.append(f"- **{name}: {x['verdict']}** — Δ = {x['delta'][0]:.3f} ± {x['delta'][1]:.3f}, change "
                 f"{x['change']:+.3f} ± {x['change_se']:.3f}{' (moves Δ)' if x['moves_delta'] else ''}, "
                 f"converged {x['converged']:.0%}, mean learning {x['t_learn_mean']:.2e} periods")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", nargs="+", choices=list(CANDIDATES), default=[])
    ap.add_argument("--sessions", type=int, default=len(SEEDS))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="gap_check")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--max-periods", type=float, default=None, help="smoke tests only")
    args = ap.parse_args()

    out = C.out_dir(MC.smoke_out(args))
    path = out / "points.json"
    records = json.loads(path.read_text())["points"] if path.exists() else []
    if not args.analyze_only:
        if not args.candidate:
            raise SystemExit("name the pre-registered candidate with --candidate")
        threads = C.init_threads(args.threads)
        seeds = SEEDS[:args.sessions]
        plan = [{"candidate": "baseline"}] + [{"candidate": c} for c in args.candidate]
        key = lambda p: p["candidate"]
        todo = MC.pending(plan, key, records, len(seeds))
        print(f"{len(todo)} points to run ({len(records)} already saved), {len(seeds)} sessions each", flush=True)
        records = MC.run(todo, key, lambda p, rs: params_for(p["candidate"], args.max_periods), records, path,
                         seeds, threads, lambda p: p["candidate"])
    v = analyze(records)
    C.save_json(out / "verdicts.json", v)
    md = report_md(v)
    (out / "verdicts.md").write_text(md)
    print()
    print(md)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
