"""Plan and rules of scripts/12_hygiene_checks.py (pre-registration 6)."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("hyg", ROOT / "scripts" / "12_hygiene_checks.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def test_plan():
    p = H.plan(["S", "H", "B"], [0.1, 100.0])
    assert len(p) == 2 * 8 + 4 * 7 + 2
    assert len({H.key(x) for x in p}) == len(p)


def test_solo_level_comes_from_reference_runs():
    ref = H.reference_index()
    # lone learner at exact equal information trades at its best response (no bias)
    assert H.solo_level(ref, 0.1, (0.0, 0.0), 1) == pytest.approx(1.0, abs=0.01)
    # the noisy speculator at (0, 1) holds back about 13 % on its own
    assert H.solo_level(ref, 0.1, (0.0, 1.0), 1) == pytest.approx(0.867, abs=0.005)
    prm = H.params_for(dict(part="S", sigma_u=0.1, q1=0.0, q2=1.0, who="solo_level0", setting="base"), ref)
    assert prm.fixed[0] is None and prm.fixed[1] == pytest.approx(0.867, abs=0.005)


def _row(su, prof, spec, measured, corrected, clean, se=0.002):
    return dict(sigma_u=su, profile=list(prof), speculator=spec, measured=measured,
                corrected=corrected, clean=clean, clean_se=se)


def test_s_rules():
    rows = []
    for su in (0.1, 100.0):
        rows += [_row(su, (0.0, 0.0), "both", 0.10, 0.10, 0.10),
                 _row(su, (0.0, 1.0), "spec1", 0.025, 0.008, 0.01),
                 _row(su, (0.0, 1.0), "spec2", 0.004, -0.002, 0.0),
                 _row(su, (1.0, 1.0), "both", 0.047, 0.027, 0.03)]
    v = H.s_verdicts(rows, [0.1, 100.0])
    assert v["S1"] == "correction supported"       # clean is nearer to corrected where they differ
    assert v["S2"] == "holds"
    assert v["S3"] == "real"
    for r in rows:
        if r["profile"] == [1.0, 1.0]:
            r["clean"] = 0.045
        if r["speculator"] == "spec1":
            r["clean"] = 0.06
    v = H.s_verdicts(rows, [0.1, 100.0])
    assert v["S1"] == "correction rejected"
    assert v["S2"] == "fails"


def test_b_rule():
    def rec(who, delta, r):
        return dict(part="B", who=who, delta={"mean": delta, "se": 0.01}, converged_share=1.0, sessions=50,
                    restraint=r, periods_learning={"median": 1e8})
    assert H.b_verdict([rec("pair", 0.70, [0.85, 0.85])])["B1"] == "reproduced"
    assert H.b_verdict([rec("pair", 0.55, [0.9, 0.9])])["B1"] == "partly"
    out = H.b_verdict([rec("pair", 0.40, [0.9, 0.92]), rec("only0", 0.0, [0.99, 1.0])])
    assert out["B1"] == "not reproduced"
    assert out["collusive_part"] == pytest.approx(0.08)
