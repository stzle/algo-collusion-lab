"""
Is the measure sound? (PREREGISTRATION.md, "Pre-registration 10".) A new sample of the
shared-noise curve at sigma_u = 0.1 (seeds 21-40), with per-session arrays, so the
standard errors are session-level and differences are paired by seed. Also a cap test at
w = 0.75 (8 x 10^8 against 2 x 10^8 periods, same seeds).

Plan, parameters and the old analysis come from 15_shared_noise.py (restricted to
sigma_u = 0.1). The helpers for per-session records, paired errors and the run loop are
also used by 17_discount_test.py and 18_matched_correlation.py.

    python scripts/16_measure_checks.py               # 14 points, about 35 min on an M4
    python scripts/16_measure_checks.py --analyze-only
    python scripts/16_measure_checks.py --sessions 2 --max-periods 1e6   # smoke -> measure_checks_smoke

Saved after every point; rerunning skips finished points.
Outputs in results/measure_checks/: points.json, verdicts.json, verdicts.md
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

import _common as C
import _dgj_common as G

_spec = importlib.util.spec_from_file_location("shared", Path(__file__).with_name("15_shared_noise.py"))
SN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SN)

SEEDS = np.arange(21, 41)
SIGMA_U = 0.1
CAP, CAP_LONG = int(2e8), int(8e8)
CAP_W = 0.75
BOOT_B, BOOT_SEED = 10_000, 12345


# ------------------------------------------------------------------ shared helpers (16, 17, 18)

def smoke_out(args) -> str:
    """Smoke runs never write to the real output folder."""
    return args.out + "_smoke" if args.max_periods and not args.out.endswith("_smoke") else args.out


def record(p, prm, seeds, s, res) -> dict:
    ps = G.session_arrays(res)
    return {**p, "sessions": len(seeds), "seeds": np.asarray(seeds).tolist(), "max_periods": prm.max_periods,
            "rho": prm.rho, "converged_share": s["converged_share"],
            "restraint": s["restraint_vs_best_response"], "restraint_se": s["restraint_vs_best_response_se"],
            "delta": s["delta_all"], "periods_learning": s["periods_learning"],
            "aggressiveness_vs_nash": s["aggressiveness_vs_nash"], "benchmarks": s["benchmarks"],
            "wall_clock_seconds": s["wall_clock_seconds"],
            "per_session": {k: ps[k] for k in ("restraint", "converged", "t_learn", "delta")}}


def pending(plan, key, records, n) -> list[dict]:
    done = {key(r) for r in records if r["sessions"] == n}
    return [p for p in plan if key(p) not in done]


def run(todo, key, params_for, records, path, seeds, threads, label) -> list[dict]:
    """Run the points one by one; save after every point (replacing an old record of the same key)."""
    for n, p in enumerate(todo, 1):
        prm = params_for(p, records)
        s, res = G.run_point(prm, seeds)
        G.print_point(f"[{n}/{len(todo)}] {label(p)}", s)
        records = [r for r in records if key(r) != key(p)]
        records.append(record(p, prm, seeds, s, res))
        C.save_json(path, {"machine": C.machine_info(threads), "points": records})
    return records


def restr(rec) -> np.ndarray:
    return np.asarray(rec["per_session"]["restraint"], dtype=float)


def same_seeds(*recs) -> bool:
    return all(r["seeds"] == recs[0]["seeds"] for r in recs)


def mean_se(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), (float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"))


def paired(a, b) -> tuple[float, float]:
    """Mean difference a - b and its paired SE (sd of per-seed differences / sqrt(n))."""
    return mean_se(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def c_sessions(pair, alone) -> np.ndarray:
    """Per-seed collusive part, symmetric profile: lone learner - mean of the pair."""
    if not same_seeds(pair, alone):
        raise ValueError("pair and lone learner were run on different seeds")
    return restr(alone)[:, 0] - restr(pair).mean(1)


def c_direct_sessions(pair, solo) -> np.ndarray:
    """Per-seed direct measure (pre-registration 6): opponent fixed at the lone learner's level."""
    if not same_seeds(pair, solo):
        raise ValueError("pair and direct run were run on different seeds")
    return restr(solo)[:, 0] - restr(pair)[:, 0]


def old_se(pair, alone) -> float:
    """The ± of 15_shared_noise.py: treats the two speculators of a pair as independent."""
    return math.hypot(alone["restraint_se"][0], math.hypot(*pair["restraint_se"]) / 2)


def conv_learn(rec) -> dict:
    t = np.asarray(rec["per_session"]["t_learn"], dtype=float)
    return {"converged": float(np.mean(rec["per_session"]["converged"])), "t_learn_mean": float(t.mean())}


# ------------------------------------------------------------------ this experiment

def plan() -> list[dict]:
    pts = [{**p, "cap": CAP} for p in SN.plan([SIGMA_U])]
    return pts + [dict(sigma_u=SIGMA_U, w=CAP_W, who=who, cap=CAP_LONG) for who in ("pair", "only0")]


def key(p):
    return SN.key(p) + (int(p["cap"]),)


def params_for(p, records, max_periods=None):
    main = [r for r in records if int(r["cap"]) == CAP]          # solo_level0 reads this sample's only0
    prm = SN.params_for(p, main)
    scale = (max_periods / CAP) if max_periods else 1.0        # smoke: both caps scaled down alike
    return dataclasses.replace(prm, max_periods=int(p["cap"] * scale))


def r1(slope, se) -> str:
    return "rises" if slope > 2 * se else "no clear rise"


def r2(d, se) -> str:
    return "higher at w = 1" if d > 2 * se else "not clearly higher"


def c1(dc, se) -> str:
    return "cap-robust" if abs(dc) < 2 * se else "cap-sensitive"


def ols_slope(x, Y) -> np.ndarray:
    """OLS slope of Y on x; Y has x along axis 0 (extra axes are separate regressions)."""
    x = np.asarray(x, dtype=float)
    xc = (x - x.mean()).reshape((-1,) + (1,) * (np.ndim(Y) - 1))
    return (xc * (Y - Y.mean(0))).sum(0) / (xc ** 2).sum()


def boot_slope_se(x, cmat, B=BOOT_B, seed=BOOT_SEED) -> float:
    """cmat: (points, seeds). Resample seed indices with replacement, the same draw for all points."""
    n = cmat.shape[1]
    draw = np.random.default_rng(seed).integers(0, n, size=(B, n))
    return float(ols_slope(x, cmat[:, draw].mean(-1)).std(ddof=1))


def analyze(records) -> dict:
    main = [r for r in records if int(r["cap"]) == CAP]
    idx = {SN.key(r): r for r in main}
    get = lambda w, who, cap=CAP: next((r for r in records if key(r) == (SIGMA_U, float(w), who, cap)), None)
    v = {"rules": {"boot_B": BOOT_B, "boot_seed": BOOT_SEED}, "curve": {},
         "R1": "not run", "R2_c": "not run", "R2_direct": "not run", "C1": "not run"}
    cs = {}
    for w in SN.SHARES:
        pr, al = get(w, "pair"), get(w, "only0")
        if pr is None or al is None:
            continue
        cs[w] = c_sessions(pr, al)
        c, se = mean_se(cs[w])
        old = SN.collusive_part(idx, SIGMA_U, w)
        d = {"c": c, "c_se": se, "c_se_old": old["c_se"],
             "pair_term": mean_se(restr(pr).mean(1)), "pair_term_se_old": math.hypot(*pr["restraint_se"]) / 2,
             "lone_term": mean_se(restr(al)[:, 0]), "lone_term_se_old": al["restraint_se"][0],
             "pair": conv_learn(pr), "only0": conv_learn(al)}
        so = get(w, "solo_level0")
        if so is not None:
            d["c_direct"], d["c_direct_se"] = mean_se(c_direct_sessions(pr, so))
            d["c_direct_se_old"] = old["c_direct_se"]
        v["curve"][f"{w:g}"] = d
    if len(cs) == len(SN.SHARES):
        x = np.array(SN.SHARES)
        cmat = np.array([cs[w] for w in SN.SHARES])
        slope = float(ols_slope(x, cmat.mean(1)))
        se = boot_slope_se(x, cmat)
        v["slope"] = {"slope": slope, "se_boot": se}
        v["R1"] = r1(slope, se)
    if 0.0 in cs and 1.0 in cs:
        d, se = paired(cs[1.0], cs[0.0])
        c0, c1_ = v["curve"]["0"], v["curve"]["1"]
        v["diff_c"] = {"d": d, "se": se, "se_old": math.hypot(c0["c_se_old"], c1_["c_se_old"])}
        v["R2_c"] = r2(d, se)
        if "c_direct" in c0 and "c_direct" in c1_:
            dd = c_direct_sessions(get(1, "pair"), get(1, "solo_level0")) - \
                c_direct_sessions(get(0, "pair"), get(0, "solo_level0"))
            d, se = mean_se(dd)
            v["diff_direct"] = {"d": d, "se": se, "se_old": math.hypot(c0["c_direct_se_old"], c1_["c_direct_se_old"])}
            v["R2_direct"] = r2(d, se)
    lp, la = get(CAP_W, "pair", CAP_LONG), get(CAP_W, "only0", CAP_LONG)
    if lp is not None and la is not None and CAP_W in cs:
        cl = c_sessions(lp, la)
        if not same_seeds(lp, get(CAP_W, "pair")):
            raise ValueError("cap test run on different seeds")
        d, se = paired(cl, cs[CAP_W])
        v["cap"] = {"c_long": mean_se(cl)[0], "c_main": mean_se(cs[CAP_W])[0], "dc": d, "se": se,
                    "se_old": math.hypot(old_se(lp, la), v["curve"][f"{CAP_W:g}"]["c_se_old"]),
                    "converged": {"pair_2e8": v["curve"][f"{CAP_W:g}"]["pair"]["converged"],
                                  "only0_2e8": v["curve"][f"{CAP_W:g}"]["only0"]["converged"],
                                  "pair_8e8": conv_learn(lp)["converged"], "only0_8e8": conv_learn(la)["converged"]}}
        v["C1"] = c1(d, se)
    return v


def report_md(v) -> str:
    pc = lambda x: f"{100 * x:+.2f}"
    L = ["# Measure checks", "",
         f"Rules: PREREGISTRATION.md, pre-registration 10. σᵤ = {SIGMA_U}, noise 0.5 each, seeds 21–40. "
         "c = collusive part (% of best response); SE = session level (paired by seed); "
         "old ± = formula of 15_shared_noise.py (speculators treated as independent).", ""]
    if "slope" in v:
        L.append(f"- **R1 (slope of c on w):** {v['R1']} — slope {pc(v['slope']['slope'])} per unit w, "
                 f"bootstrap SE {100 * v['slope']['se_boot']:.2f}")
    else:
        L.append(f"- **R1 (slope of c on w):** {v['R1']}")
    for name, k in (("R2 (c(1) − c(0))", "c"), ("R2 direct (c_direct(1) − c_direct(0))", "direct")):
        x = v.get(f"diff_{k}")
        L.append(f"- **{name}:** {v[f'R2_{k}']}" + (f" — {pc(x['d'])}, SE {100 * x['se']:.2f} "
                                                     f"(old ± {100 * x['se_old']:.2f})" if x else ""))
    x = v.get("cap")
    L.append(f"- **C1 (cap 8×10⁸ vs 2×10⁸, w = {CAP_W:g}):** {v['C1']}" + (
        f" — Δc {pc(x['dc'])}, SE {100 * x['se']:.2f} (old ± {100 * x['se_old']:.2f}); converged pair "
        f"{x['converged']['pair_2e8']:.0%} → {x['converged']['pair_8e8']:.0%}, lone learner "
        f"{x['converged']['only0_2e8']:.0%} → {x['converged']['only0_8e8']:.0%}" if x else ""))
    L += ["", "| w | c | SE | old ± | direct | SE | old ± | pair term | SE | old ± | lone term | SE | old ± | conv pair | conv alone |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for w, d in v["curve"].items():
        dr = (f"{pc(d['c_direct'])} | {100 * d['c_direct_se']:.2f} | {100 * d['c_direct_se_old']:.2f}"
              if "c_direct" in d else "– | – | –")
        L.append(f"| {w} | {pc(d['c'])} | {100 * d['c_se']:.2f} | {100 * d['c_se_old']:.2f} | {dr} | "
                 f"{d['pair_term'][0]:.4f} | {d['pair_term'][1]:.4f} | {d['pair_term_se_old']:.4f} | "
                 f"{d['lone_term'][0]:.4f} | {d['lone_term'][1]:.4f} | {d['lone_term_se_old']:.4f} | "
                 f"{d['pair']['converged']:.0%} | {d['only0']['converged']:.0%} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=len(SEEDS))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="measure_checks")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--max-periods", type=float, default=None, help="smoke tests only")
    args = ap.parse_args()

    out = C.out_dir(smoke_out(args))
    path = out / "points.json"
    records = json.loads(path.read_text())["points"] if path.exists() else []
    if not args.analyze_only:
        threads = C.init_threads(args.threads)
        seeds = SEEDS[:args.sessions]
        todo = pending(plan(), key, records, len(seeds))
        print(f"{len(todo)} points to run ({len(records)} already saved), {len(seeds)} sessions each", flush=True)
        records = run(todo, key, lambda p, rs: params_for(p, rs, args.max_periods), records, path, seeds, threads,
                      lambda p: f"σu={p['sigma_u']:g} w={p['w']:g} {p['who']} cap={p['cap']:.0e}")
    v = analyze(records)
    C.save_json(out / "verdicts.json", v)
    md = report_md(v)
    (out / "verdicts.md").write_text(md)
    print()
    print(md)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
