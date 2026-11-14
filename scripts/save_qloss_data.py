"""
Saved sessions for the Q-loss analysis (PREREGISTRATION.md, pre-registration 13).
Repeats existing runs with the same seeds and writes the learned Q-tables, the final
pricing rule and the final chain state to disk, so the analysis needs no new simulation
runs. Nothing new is revealed: each point is checked against the restraint means
already saved by 08_precision_sweep.py (fine grid) and 15_shared_noise.py.

Profiles at sigma_u = 0.1, price grid per lagged value, variant A, seeds 1-20:
  exact   exact information                        (reference: results/precision_fine)
  w0      noise 0.5 each, independent  (w = 0)     (reference: results/shared_noise)
  w1      noise 0.5 each, fully shared (w = 1)     (reference: results/shared_noise)
each for the pair and the lone learner (speculator 2 fixed at Nash).

    python scripts/save_qloss_data.py                       # 6 points, about 10 min on an M4
    python scripts/save_qloss_data.py --max-periods 1e6     # smoke test, no reference check

Outputs in results/qloss_data/: <profile>_<who>.npz (about 5.5 MB each), check.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _common as C
import _dgj_common as G

SIGMA_U = 0.1
PROFILES = {"exact": dict(sigma_eps=(0.0, 0.0), noise_shared=0.0),
            "w0": dict(sigma_eps=(0.5, 0.5), noise_shared=0.0),
            "w1": dict(sigma_eps=(0.5, 0.5), noise_shared=1.0)}
FIXED = {"pair": (None, None), "only0": (None, 1.0)}


def params_for(profile: str, who: str):
    return G.make_params(sigma_u=SIGMA_U, fixed=FIXED[who], grid="nash", grid_width=0.3,
                         monitor="realized", price_grid="per_value", **PROFILES[profile])


def reference(profile: str, who: str) -> list[float] | None:
    """Restraint means saved by the original run of this point."""
    if profile == "exact":
        path, match = C.RESULTS / "precision_fine" / "points.json", \
            lambda r: (r["sigma_u"], r["q1"], r["q2"], r["who"]) == (SIGMA_U, 0.0, 0.0, who)
    else:
        w = PROFILES[profile]["noise_shared"]
        path, match = C.RESULTS / "shared_noise" / "points.json", \
            lambda r: (r["sigma_u"], r["w"], r["who"]) == (SIGMA_U, w, who)
    if not path.exists():
        return None
    hits = [r for r in json.loads(path.read_text())["points"] if match(r)]
    return hits[0]["restraint"] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="qloss_data")
    ap.add_argument("--max-periods", type=float, default=None, help="smoke tests only")
    args = ap.parse_args()
    smoke = args.max_periods is not None
    if smoke:
        G.LAPTOP["max_periods"] = int(args.max_periods)
    out = C.out_dir(args.out + ("_smoke" if smoke else ""))
    threads = C.init_threads(args.threads)
    seeds = np.arange(1, args.sessions + 1)
    check = {"machine": C.machine_info(threads), "points": {}}
    for profile in PROFILES:
        for who in FIXED:
            name = f"{profile}_{who}"
            prm = params_for(profile, who)
            s, res = G.run_point(prm, seeds)
            G.save_sessions(out / f"{name}.npz", prm, seeds, res)
            ref = None if smoke else reference(profile, who)
            got = s["restraint_vs_best_response"]
            ok = None if ref is None else bool(np.allclose(got, ref, rtol=0, atol=1e-9))
            check["points"][name] = {"restraint": got, "reference": ref, "matches": ok,
                                     "converged_share": s["converged_share"]}
            G.print_point(f"{name}", s)
            print(f"    reference {ref}  ->  {'matches' if ok else 'MISMATCH' if ok is False else 'not checked'}",
                  flush=True)
            C.save_json(out / "check.json", check)
    bad = [k for k, v in check["points"].items() if v["matches"] is False]
    print(f"\n{'All points match their reference.' if not bad and not smoke else 'Mismatch: ' + ', '.join(bad) if bad else 'Smoke test done.'}"
          f" Results in {out}")


if __name__ == "__main__":
    main()
