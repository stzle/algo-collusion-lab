"""Rules of scripts/09_noise_vs_inequality.py (pre-registration 3)."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("noise", ROOT / "scripts" / "09_noise_vs_inequality.py")
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)


def test_matched_profiles_have_equal_mean_noise():
    for m, f in N.families().items():
        for p in f.values():
            if p is not None:
                assert abs((p[0] + p[1]) / 2 - m) < 1e-12


def test_plan_reuses_sweep_points():
    sweep = {N.S.key(p) for p in N.S.plan([0.1, 100.0])}
    plan = N.plan([0.1, 100.0])
    assert len(plan) == 72
    assert len([p for p in plan if N.S.key(p) not in sweep]) == 28
    recs = json.loads((ROOT / "reference_results" / "precision" / "points.json").read_text())["points"]
    assert {N.S.key(p) for p in plan} <= {N.S.key(r) for r in recs}


def test_i1_and_i2():
    assert N.i1([3, 4], 5) == "yes"
    assert N.i1([1, 0], 5) == "no"
    assert N.i1([3, 1], 5) == "partial"
    assert N.i1([2, 2], 3) == "yes"
    assert N.i1([0, 0], 3) == "no"
    rows = {0.25: {"share": 0.2}, 0.5: {"share": 0.4}, 1.0: {"share": None}}
    assert N.i2_cell(rows) == "mainly noise"
    rows[0.25]["share"] = 0.7
    assert N.i2_cell(rows) == "mixed"


def test_effects_missing_returns_none():
    assert N.effects({}, 0.1, "exact") is None
