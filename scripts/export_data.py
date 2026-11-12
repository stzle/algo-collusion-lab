"""
Export every saved run into two flat tables (CSV) for checking the results by hand.

  data/all_runs.csv   one row per run: parameters, convergence, restraint, Δ, market
                      quality (where saved) and shock responses (shock test only)
  data/derived.csv    the numbers the verdicts are built from: collusive part c and
                      learning bias b per speculator, the part of c explained by two
                      independent learning biases (exploratory), and the punishment
                      response P of the shock test

    python scripts/export_data.py

Reads reference_results/{robustness,precision,shock,final,hygiene}/points.json and, when present,
results/final/points.json (or reference_results/final/points.json).
Column definitions: data/COLUMNS.md
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from lab import dgj_benchmarks as bmk  # noqa: E402

OUT = ROOT / "data"
SHOCKS = ["0.25", "2.5", "11.5", "15"]

# who learns -> (label, which speculators learn)
CELL = {"pair": ("pair", (0, 1)), "alone": ("spec1_alone", (0,)), "informed_only": ("spec1_alone", (0,)),
        "noisy_only": ("spec2_alone", (1,)), "only0": ("spec1_alone", (0,)), "only1": ("spec2_alone", (1,)),
        "solo_level0": ("spec1_vs_solo_level", (0,)), "solo_level1": ("spec2_vs_solo_level", (1,))}


def _load(*names):
    for n in names:
        p = ROOT / n
        if p.exists():
            return json.loads(p.read_text())
    return None


def rows_all():
    out = []
    sources = [("robustness", _load("reference_results/robustness/points.json")),
               ("precision", _load("reference_results/precision/points.json")),
               ("shock", _load("reference_results/shock/points.json")),
               ("final", _load("results/final/points.json", "reference_results/final/points.json")),
               ("checks", _load("results/hygiene/points.json", "reference_results/hygiene/points.json")),
               ("audit", _load("results/price_grid/points.json", "reference_results/price_grid/points.json")),
               ("precision_fine", _load("results/precision_fine/points.json", "reference_results/precision_fine/points.json")),
               ("shock_fine", _load("results/shock_fine/points.json", "reference_results/shock_fine/points.json"))]
    for exp, data in sources:
        if data is None:
            continue
        date = data["machine"].get("date", "")
        for p in data["points"]:
            q1, q2 = (0.0, p["noise"]) if exp == "robustness" else (p["q1"], p["q2"])
            cell, learners = CELL[p["who"]]
            paper = (exp == "checks" and p["part"] == "B") or (exp == "audit" and p["part"] == "P")
            fine = exp in ("audit", "precision_fine", "shock_fine")
            monitor = "own_signal" if (exp == "audit" and p["part"] == "M") else p.get("monitor", "realized")
            r = {"experiment": exp if exp not in ("final", "checks", "audit") else f"{exp}_{p['part']}",
                 "sigma_u": p["sigma_u"], "noise_spec1": q1, "noise_spec2": q2,
                 "grid_width": "paper" if paper else p.get("width", 0.3), "monitoring": monitor,
                 "price_grid": "per_value" if fine else "global",
                 "period_cap": (10_000_000_000 if exp == "audit" else 2_000_000_000) if paper else p.get("cap", 200_000_000),
                 "value_levels": 10 if paper else p.get("n_v", 6),
                 "learning_setting": p.get("setting", "base"),
                 "who_learns": cell, "runs": p["sessions"], "converged_share": p["converged_share"]}
            for i in (0, 1):
                learn = i in learners
                r[f"restraint_spec{i + 1}"] = p["restraint"][i] if learn else ""
                se = p.get("restraint_se")
                r[f"restraint_se_spec{i + 1}"] = se[i] if (learn and se is not None) else ""
            d = p.get("delta")
            r["delta_mean"] = d["mean"] if d else ""
            r["delta_se"] = d["se"] if d else ""
            for k in ("informativeness", "mispricing", "price_impact"):
                r[k] = p[k]["mean"] if k in p else ""
            for i in (0, 1):
                r[f"profit_spec{i + 1}"] = p["profit_mean"][i] if "profit_mean" in p else ""
            if "benchmarks" in p:
                b = p["benchmarks"]
                r["lambda_nash"], r["lambda_cartel"] = b["lambda_nash"], b["lambda_cartel"]
                for i in (0, 1):
                    r[f"profit_nash_spec{i + 1}"] = b["profit_nash"][i]
                    r[f"profit_cartel_spec{i + 1}"] = b["profit_cartel"][i]
            if "R_t4" in p:
                R = np.array(p["R_t4"])                          # sessions x shocks x speculators
                for k, lab in enumerate(SHOCKS):
                    for i in (0, 1):
                        if i in learners:
                            r[f"shock{lab}_response_spec{i + 1}"] = R[:, k, i].mean()
                            r[f"shock{lab}_response_se_spec{i + 1}"] = R[:, k, i].std(ddof=1) / math.sqrt(R.shape[0])
            r["date"] = date
            r = {"id": "|".join(str(r[k]) for k in ("experiment", "sigma_u", "noise_spec1", "noise_spec2",
                                                     "grid_width", "monitoring", "price_grid", "period_cap", "value_levels",
                                                     "learning_setting", "who_learns")), **r}
            out.append(r)
    return out


def _key(r):
    return (r["experiment"], r["sigma_u"], r["noise_spec1"], r["noise_spec2"], r["grid_width"],
            r["monitoring"], r["price_grid"], r["period_cap"], r["value_levels"], r["learning_setting"])


def _n(x):
    return float("nan") if x == "" else float(x)


def rows_derived(allr):
    groups = {}
    for r in allr:
        groups.setdefault(_key(r), {})[r["who_learns"]] = r
    v_grids = {}
    out = []
    hv = _load("results/hygiene/verdicts.json", "reference_results/hygiene/verdicts.json")
    clean = {(float(r["sigma_u"]), float(r["profile"][0]), float(r["profile"][1]), r["speculator"]):
             (r["clean"], r["clean_se"]) for r in (hv["S"]["rows"] if hv else [])}
    for k, g in groups.items():
        if "pair" not in g or "spec1_alone" not in g:
            continue
        exp, su, q1, q2, w, mon, pg, cap, n_v, setting = k
        sym = q1 == q2
        if not sym and "spec2_alone" not in g:
            continue
        pr = g["pair"]
        # mechanical part: both speculators keep their lone-learner behaviour, no coordination
        if n_v not in v_grids:
            v_grids[n_v] = bmk.value_grid(n_v, 1.0, 1.0)
        bm = bmk.benchmarks(v_grids[n_v], (q1, q2), 500.0, 0.1, su, 1.0)
        C, S, lam, chiN = bm["C"], bm["S"], bm["nash"]["lam"], np.asarray(bm["nash"]["chi"])
        alone = {}
        for i in (0, 1):
            src = g["spec1_alone"] if (sym or i == 0) else g["spec2_alone"]
            j = 0 if sym else i
            alone[i] = (_n(src[f"restraint_spec{j + 1}"]), _n(src[f"restraint_se_spec{j + 1}"]))
        chi_alone = np.array([alone[i][0] * bmk.best_response(i, chiN.copy(), C, S, lam) for i in (0, 1)])
        r_mech = bmk.restraint(chi_alone, C, S, lam)["restraint"]
        specs = [("both", None)] if sym else [("spec1", 0), ("spec2", 1)]
        for name, i in specs:
            if sym:
                rp = (_n(pr["restraint_spec1"]) + _n(pr["restraint_spec2"])) / 2
                rp_se = math.hypot(_n(pr["restraint_se_spec1"]), _n(pr["restraint_se_spec2"])) / 2
                ra, ra_se = alone[0]
                mech = ra - r_mech[0]
            else:
                rp, rp_se = _n(pr[f"restraint_spec{i + 1}"]), _n(pr[f"restraint_se_spec{i + 1}"])
                ra, ra_se = alone[i]
                mech = ra - r_mech[i]
            c = ra - rp
            row = {"id": f"{exp}|{su}|{q1}|{q2}|{w}|{mon}|{pg}|{cap}|{n_v}|{setting}|{name}", "experiment": exp, "sigma_u": su, "noise_spec1": q1, "noise_spec2": q2, "grid_width": w,
                   "monitoring": mon, "price_grid": pg, "period_cap": cap, "value_levels": n_v, "learning_setting": setting,
                   "speculator": name,
                   "restraint_pair": rp, "restraint_alone": ra,
                   "collusive_part": c, "collusive_part_se": math.hypot(ra_se, rp_se),
                   "learning_bias": 1 - ra, "learning_bias_se": ra_se,
                   "phantom_part": mech, "collusive_part_corrected": c - mech,
                   "collusive_part_clean": float("nan"), "collusive_part_clean_se": float("nan"),
                   "converged_share_pair": pr["converged_share"]}
            cl = clean.get((float(su), float(q1), float(q2), name)) if exp == "precision" else None
            if cl:
                row["collusive_part_clean"], row["collusive_part_clean_se"] = cl
            if "shock15_response_spec1" in pr:
                for lab in SHOCKS:
                    for_spec = [0, 1] if sym else [i]
                    for j in for_spec:
                        src = g["spec1_alone"] if (sym or j == 0) else g["spec2_alone"]
                        jj = 0 if sym else j
                        P = pr[f"shock{lab}_response_spec{j + 1}"] - src[f"shock{lab}_response_spec{jj + 1}"]
                        se = math.hypot(pr[f"shock{lab}_response_se_spec{j + 1}"],
                                        src[f"shock{lab}_response_se_spec{jj + 1}"])
                        suffix = "" if not sym else f"_spec{j + 1}"
                        row[f"punishment_shock{lab}{suffix}"] = P
                        row[f"punishment_se_shock{lab}{suffix}"] = se
            out.append(row)
    return out


def write(path, rows):
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if isinstance(r.get(c), float) and math.isnan(r[c]) else
                            f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    return cols


def main():
    OUT.mkdir(exist_ok=True)
    allr = rows_all()
    der = rows_derived(allr)
    c1 = write(OUT / "all_runs.csv", allr)
    c2 = write(OUT / "derived.csv", der)
    print(f"all_runs.csv: {len(allr)} rows, {len(c1)} columns")
    print(f"derived.csv: {len(der)} rows, {len(c2)} columns")


if __name__ == "__main__":
    main()
