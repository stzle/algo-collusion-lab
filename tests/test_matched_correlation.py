"""Matched correlation (pre-registration 12): measured correlation, q-matching, rule A1."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from lab import dgj_benchmarks as bmk  # noqa: E402

spec = importlib.util.spec_from_file_location("matched", ROOT / "scripts" / "18_matched_correlation.py")
MCOR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MCOR)

from test_measure_checks import rec  # noqa: E402

V = MCOR.V_GRID


@pytest.mark.parametrize("q1,q2,w", [(0.5, 0.5, 0.0), (0.0, 0.5, 0.0), (0.25, 0.75, 0.0), (0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.75)])
def test_correlation_agrees_with_moments(q1, q2, w):
    C, S, _ = bmk.moments(V, np.array([q1, q2]), shared=w)     # signals have mean v_bar: S = covariances
    assert MCOR.measured_correlation(V, q1, q2, w) == pytest.approx(S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]), abs=1e-6)


def test_correlation_limits():
    assert MCOR.measured_correlation(V, 0.0, 0.0, 0.0) == pytest.approx(1.0)     # exact information
    assert MCOR.measured_correlation(V, 0.5, 0.5, 1.0) == pytest.approx(1.0)     # identical signals
    cs = [MCOR.measured_correlation(V, 0.5, 0.5, w) for w in (0, 0.25, 0.5, 0.75, 1)]
    assert np.all(np.diff(cs) > 0)


def test_q_matching():
    lv = MCOR.levels()
    assert set(lv) == {"0.5", "0.75"}
    for w, x in lv.items():
        assert x["corr_target"] == pytest.approx(MCOR.measured_correlation(V, 0.5, 0.5, float(w)))
        assert x["corr_achieved"] == pytest.approx(x["corr_target"], abs=1e-8)
    assert 0 < lv["0.75"]["q"] < lv["0.5"]["q"] < 1      # higher correlation needs less noise
    p = MCOR.plan(lv)
    assert len(p) == 6 and len({MCOR.key(x) for x in p}) == 6
    prm = MCOR.params_for(p[2])
    assert prm.sigma_eps == (0.0, lv["0.5"]["q"]) and prm.fixed == (1.0, None) and prm.noise_shared == 0.0


def test_c_unequal_is_the_mean_over_speculators():
    pair = rec([[0.9, 0.8], [0.92, 0.82]])
    o0 = rec([[0.95, 1.0], [0.97, 1.0]])
    o1 = rec([[1.0, 0.81], [1.0, 0.83]])
    assert MCOR.c_unequal_sessions(pair, o0, o1) == pytest.approx([(0.05 + 0.01) / 2, (0.05 + 0.01) / 2])


def test_a1_labels():
    assert MCOR.a1(0.02, 0.005) == "lower" and MCOR.a1(0.005, 0.005) == "not lower"
    assert MCOR.overall(["lower", "lower"]) == "correlation is not enough"
    assert MCOR.overall(["lower", "not lower"]) == "partly"
    assert MCOR.overall(["not lower", "not lower"]) == "not supported"


def _arms(c_eq, c_un, n=20, seed=0):
    rng = np.random.default_rng(seed)
    unequal, equal = [], []
    for w in MCOR.LEVELS:
        pair = 0.9 + 0.003 * rng.standard_normal((n, 2))
        o0, o1 = pair + c_un[w] + 0.003 * rng.standard_normal((n, 2)), pair + c_un[w]
        for who, R in (("pair", pair), ("only0", o0), ("only1", o1)):
            unequal.append(rec(R, level=w, q1=0.0, q2=0.4, corr_target=0.9, corr_achieved=0.9, who=who))
        ep = 0.9 + 0.003 * rng.standard_normal((n, 2))
        equal += [rec(ep, sigma_u=0.1, w=w, who="pair", cap=MCOR.MC.CAP),
                  rec(ep.mean(1, keepdims=True) + c_eq[w] + 0.003 * rng.standard_normal((n, 2)),
                      sigma_u=0.1, w=w, who="only0", cap=MCOR.MC.CAP)]
    return unequal, equal


def test_analyze():
    un, eq = _arms({0.5: 0.044, 0.75: 0.063}, {0.5: 0.025, 0.75: 0.063})
    v = MCOR.analyze(un, eq)
    assert v["levels"]["0.5"]["A1"] == "lower" and v["levels"]["0.75"]["A1"] == "not lower"
    assert v["A1"] == "partly"
    assert v["levels"]["0.5"]["d"] == pytest.approx(0.019, abs=0.003)
    assert MCOR.analyze(un, [])["A1"] == "not run"
    assert "partly" in MCOR.report_md(v)
