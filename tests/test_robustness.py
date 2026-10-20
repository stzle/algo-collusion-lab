"""Decision rules of scripts/07_robustness.py, checked on the saved first results
(reference_results/control, equal grid +-30 %) and on synthetic cases."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("rob", ROOT / "scripts" / "07_robustness.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def control_records():
    res = json.loads((ROOT / "reference_results" / "control" / "summary.json").read_text())
    out = []
    for p in res["points"]:
        if p["grid"] != "nash":
            continue
        out.append(dict(sigma_u=p["sigma_u"], width=0.3, monitor="realized", cap=R.DEFAULT_CAP,
                        noise=p["noise"], who=p["who"], sessions=p["sessions"],
                        converged_share=p["converged_share"],
                        restraint=p["restraint_vs_best_response"],
                        restraint_se=p["restraint_vs_best_response_se"]))
    return out


def test_first_results_reproduce_published_decomposition():
    v = R.analyze(control_records(), [0.1, 100.0], [0.5, 1.0], [0.3], [R.DEFAULT_CAP])
    d = {e["sigma_u"]: e["cases"] for e in v["decomposition"]}
    # numbers shown in the Vorarbeiten chart
    assert d[0.1]["sym"]["c"] == pytest.approx(0.10, abs=0.005)
    assert d[100.0]["sym"]["c"] == pytest.approx(0.072, abs=0.005)
    assert d[0.1]["informed_q0.5"]["c"] == pytest.approx(0.02, abs=0.002)
    assert d[0.1]["noisy_q1"]["b"] == pytest.approx(0.133, abs=0.002)
    assert v["R1"]["per_width"] == {"0.3": ["holds", "holds"]}
    assert v["R1"]["verdict"] == "insufficient"          # one width is not enough
    assert v["S1"] == "holds"
    assert v["H3_proxy"] == "supported"


def _cases(c_sym, c_asym, b_noisy=0.1, se=0.002):
    d = {"sym": {"c": c_sym, "c_se": se, "b": 0.0, "b_se": se}}
    for q in (0.5, 1.0):
        d[f"informed_q{q:g}"] = {"c": c_asym, "c_se": se, "b": 0.01, "b_se": se}
        d[f"noisy_q{q:g}"] = {"c": c_asym, "c_se": se, "b": b_noisy, "b_se": se}
    return d


def test_r1_cell_rules():
    assert R.r1_cell(_cases(0.08, 0.04)) == "holds"          # exactly half
    assert R.r1_cell(_cases(0.08, 0.041)) == "fails"
    assert R.r1_cell(_cases(0.02, 0.0)) == "uninformative"   # no collusion to break
    assert R.r1_cell(_cases(0.04, 0.0, se=0.03)) == "uninformative"  # not significant


def test_r1_overall():
    assert R.r1_overall({0.2: ["holds", "holds"], 0.3: ["holds", "holds"], 0.4: ["holds", "uninformative"]}) == "robust"
    assert R.r1_overall({0.2: ["fails", "holds"], 0.3: ["holds", "holds"], 0.4: ["holds", "holds"]}) == "grid-dependent"
    assert R.r1_overall({0.2: ["holds"], 0.3: ["fails"], 0.4: ["holds"]}) == "fails"
    assert R.r1_overall({0.2: ["uninformative"], 0.3: ["holds"], 0.4: ["uninformative"]}) == "insufficient"


def test_s1_and_h3():
    assert R.s1_ok(_cases(0.08, 0.0, b_noisy=0.05))
    assert not R.s1_ok(_cases(0.08, 0.0, b_noisy=0.02))
    assert R.h3_ok(_cases(0.08, 0.0, b_noisy=0.05))
    assert not R.h3_ok(_cases(0.08, 0.06, b_noisy=0.05))     # collusion did not fall


def test_r2_paired_caps():
    recs = []
    for cap, shift in ((int(2e8), 0.0), (int(4e8), 0.005)):
        for q, who in R.cells([0.5]):
            recs.append(dict(sigma_u=100.0, width=0.3, monitor="realized", cap=cap, noise=q, who=who,
                             sessions=20, converged_share=0.0,
                             restraint=[0.95 + shift, 0.9 + shift], restraint_se=[0.002, 0.002]))
    idx = R._index(recs)
    assert R.r2_check(idx, 100.0, [0.5], 2e8, 4e8)["ok"]
    for r in recs:
        if r["cap"] == int(4e8) and r["who"] == "pair":
            r["restraint"] = [0.97, 0.9]
    assert not R.r2_check(R._index(recs), 100.0, [0.5], 2e8, 4e8)["ok"]


def test_plan_has_no_duplicates():
    pts = R.plan_points(["grid", "monitor", "caps"], [0.1, 100.0], [0.5, 1.0], [0.2, 0.3, 0.4], [1e8, 2e8, 4e8])
    keys = [R.key(p) for p in pts]
    assert len(keys) == len(set(keys))
    # 8 cells x 2 sigma_u x (3 widths + 2 extra caps) + 6 unequal-precision cells x 2 sigma_u (variant B)
    assert len(pts) == 8 * 2 * 5 + 6 * 2
