"""Discount test (pre-registration 11): rule, gate, reference check."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("discount", ROOT / "scripts" / "17_discount_test.py")
DT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DT)

from test_measure_checks import rec  # noqa: E402


def test_discount_rule_labels():
    assert DT.discount_rule(0.095, 0.10, 0.002) == "unchanged"         # |Δc| < 0.01
    assert DT.discount_rule(0.07, 0.10, 0.02) == "unchanged"           # |Δc| < 2 SE
    assert DT.discount_rule(0.04, 0.10, 0.002) == "falls"
    assert DT.discount_rule(0.13, 0.10, 0.002) == "rises"
    assert DT.discount_rule(0.07, 0.10, 0.002) == "in between"


def test_channel_gate():
    assert DT.channel("unchanged", "unchanged") == "test not informative"
    assert DT.channel("in between", "falls") == "test not informative"
    assert DT.channel("falls", "unchanged") == "evidence against a punishment channel"
    assert DT.channel("falls", "falls").startswith("consistent with punishment")
    assert DT.channel("falls", "in between") == "inconclusive"
    assert DT.channel("falls", "rises") == "inconclusive"
    assert DT.channel(None, "falls") == "not run"


def test_plan_and_params():
    p = DT.plan()
    assert len(p) == 24 and len({DT.key(x) for x in p}) == 24
    assert sum(x["rho"] != DT.RHO_REF for x in p) == 16
    prm = DT.params_for(dict(profile="w1", rho=0.0, who="only0"))
    assert prm.rho == 0.0 and prm.noise_shared == 1.0 and prm.sigma_eps == (0.5, 0.5) and prm.fixed == (None, 1.0)
    assert prm.price_grid == "per_value" and prm.max_periods == DT.CAP
    assert DT.params_for(dict(profile="exact100", rho=0.5, who="pair"), max_periods=1e6).max_periods == 1_000_000
    assert list(DT.SEEDS[:2]) == [1, 2]


def _records(c, n=20, seed=0, max_periods=DT.CAP):
    """c: {(profile, rho): collusive part}."""
    rng = np.random.default_rng(seed)
    out = []
    for (pr, rho), cv in c.items():
        pair = 0.9 + 0.003 * rng.standard_normal((n, 2))
        alone = pair.mean(1, keepdims=True) + cv + 0.003 * rng.standard_normal((n, 2))
        out += [{**rec(pair, seeds=range(1, n + 1), profile=pr, rho=rho, who="pair"), "max_periods": max_periods},
                {**rec(alone, seeds=range(1, n + 1), profile=pr, rho=rho, who="only0"), "max_periods": max_periods}]
    return out


def test_analyze_gate_passes(tmp_path):
    c = {}
    for pr, (ref, r0, r5) in {"exact": (0.12, 0.02, 0.07), "w0": (0.033, 0.03, 0.033),
                              "w1": (0.097, 0.095, 0.03), "exact100": (0.11, 0.15, 0.11)}.items():
        c.update({(pr, 0.95): ref, (pr, 0.0): r0, (pr, 0.5): r5})
    v = DT.analyze(_records(c), results=tmp_path)
    assert v["profiles"]["exact"]["0"]["label"] == "falls"
    assert v["profiles"]["exact100"]["0"]["label"] == "rises"
    assert v["profiles"]["exact"]["0.5"]["label"] == "in between"
    assert v["channel"]["0"] == "evidence against a punishment channel"
    assert v["channel"]["0.5"] == "test not informative"        # exact does not fall at 0.5
    assert v["profiles"]["w1"]["0"]["dc"] == pytest.approx(-0.002, abs=0.003)
    assert all(x["status"] == "no reference" for x in v["reference"].values())
    assert "evidence against" in DT.report_md(v)


def test_reference_check(tmp_path):
    (tmp_path / "precision_fine").mkdir()
    ref = {"sigma_u": 0.1, "q1": 0.0, "q2": 0.0, "who": "pair", "sessions": 20, "restraint": [0.877, 0.879]}
    (tmp_path / "precision_fine" / "points.json").write_text(json.dumps({"points": [ref]}))
    r = {"profile": "exact", "rho": 0.95, "who": "pair", "sessions": 20, "max_periods": DT.CAP,
         "restraint": [0.877, 0.879]}
    found = DT.reference(r, tmp_path)
    assert found is not None and DT.check_reference(r, found)["status"] == "match"
    assert DT.check_reference({**r, "restraint": [0.877, 0.88]}, found)["status"] == "mismatch"
    assert DT.check_reference({**r, "sessions": 2}, found)["status"].startswith("not comparable")
    assert DT.check_reference(r, None)["status"] == "no reference"
    assert DT.reference({**r, "profile": "w1"}, tmp_path) is None


def test_references_exist_in_saved_results():
    if not (ROOT / "results" / "precision_fine" / "points.json").exists():
        pytest.skip("no saved results")
    for p in DT.plan():
        if p["rho"] == DT.RHO_REF:
            assert DT.reference(p) is not None, p
