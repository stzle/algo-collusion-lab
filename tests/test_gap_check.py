"""Decision rule of pre-registration 14 (scripts/20_gap_check.py) on synthetic records."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "gap_check", Path(__file__).resolve().parents[1] / "scripts" / "20_gap_check.py")
GC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GC)


def test_g1_labels():
    assert GC.g1(0.70, 0.28, 0.02) == "reproduced"
    assert GC.g1(0.55, 0.13, 0.02) == "partly"
    assert GC.g1(0.48, 0.06, 0.01) == "not reproduced"      # moves Δ, but below 0.50
    assert GC.g1(0.70, 0.28, 0.20) == "not reproduced"      # not > 2 SE


def test_cliff_labels():
    assert GC.cliff(-0.30, 0.03) == "cliff"
    assert GC.cliff(-0.05, 0.01) == "flat"                   # below the 0.10 minimum
    assert GC.cliff(-0.15, 0.10) == "flat"                   # not > 2 SE


def _rec(cand, deltas, seeds=(1, 2, 3)):
    n = len(deltas)
    return {"candidate": cand, "sessions": n, "seeds": list(seeds), "max_periods": 10,
            "per_session": {"delta": deltas, "t_learn": [1e9] * n, "converged": [True] * n}}


def test_analyze_pairs_by_seed():
    v = GC.analyze([_rec("baseline", [0.40, 0.42, 0.44]), _rec("learn_longer", [0.70, 0.71, 0.75])])
    x = v["candidates"]["learn_longer"]
    assert x["change"] == pytest.approx(np.mean([0.30, 0.29, 0.31]))
    assert x["verdict"] == "reproduced"
    assert v["baseline"]["reference"].startswith("not comparable")
    with pytest.raises(ValueError):
        GC.analyze([_rec("baseline", [0.4] * 3), _rec("cliff", [0.3] * 3, seeds=(4, 5, 6))])
