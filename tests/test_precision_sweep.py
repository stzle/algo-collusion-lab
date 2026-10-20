"""Rules of scripts/08_precision_sweep.py (pre-registration 2)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("sweep", ROOT / "scripts" / "08_precision_sweep.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

WHO = {"pair": "pair", "alone": "only0", "informed_only": "only0", "noisy_only": "only1"}


def robustness_records():
    """Width 0.3 / variant A / cap 2e8 points of the robustness run, in sweep format."""
    pts = json.loads((ROOT / "reference_results" / "robustness" / "points.json").read_text())["points"]
    return [dict(sigma_u=p["sigma_u"], q1=0.0, q2=p["noise"], who=WHO[p["who"]], sessions=p["sessions"],
                 converged_share=p["converged_share"], restraint=p["restraint"], restraint_se=p["restraint_se"])
            for p in pts if p["width"] == 0.3 and p["monitor"] == "realized" and p["cap"] == int(2e8)]


def test_decomposition_matches_robustness_run():
    idx = {S.key(r): r for r in robustness_records()}
    ref = S.decompose(idx, 0.1, (0.0, 0.0))
    assert ref["both"]["c"] == pytest.approx(0.102, abs=0.001)
    d = S.decompose(idx, 0.1, (0.0, 0.5))
    assert d["spec1"]["c"] == pytest.approx(0.020, abs=0.001)
    assert d["spec2"]["b"] == pytest.approx(0.060, abs=0.001)
    assert S.rho(d, ref) == pytest.approx(0.020 / 0.102, abs=0.01)


def test_p1_rules():
    assert S.p1_cell({0.1: 0.5, 0.25: 0.4, 0.5: 0.2, 1.0: 0.1, 2.0: 0.1}) == "collapse"
    assert S.p1_cell({0.1: 0.9, 0.25: 0.7, 0.5: 0.4, 1.0: 0.3, 2.0: 0.2}) == "gradual"
    assert S.p1_cell({0.1: 0.9, 0.25: 0.9, 0.5: 0.8, 1.0: 0.7, 2.0: 0.2}) == "mixed"
    assert S.combine(["collapse", "collapse"], "x") == "collapse"
    assert S.combine(["collapse", "gradual"], "x") == "x"
    assert S.combine(["uninformative", "gradual"], "x") == "gradual"


def _sym(c, se=0.002):
    return {"both": {"c": c, "c_se": se}}


def test_informative_and_rho():
    assert S.informative(_sym(0.05))
    assert not S.informative(_sym(0.02))
    assert not S.informative(_sym(0.04, se=0.03))
    d = {"spec1": {"c": 0.01}, "spec2": {"c": 0.03}}
    assert S.rho(d, _sym(0.06)) == pytest.approx(0.5)


def test_plan_size():
    # per sigma_u: 2 + 5 x 3 + 2 + 3 = 22
    assert len(S.plan([0.1, 100.0])) == 44
    assert len({S.key(p) for p in S.plan([0.1, 100.0])}) == 44
