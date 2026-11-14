"""Measure checks (pre-registration 10): session-level errors, bootstrap and the rules R1, R2, C1."""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("measure", ROOT / "scripts" / "16_measure_checks.py")
MC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MC)


def rec(R, seeds=None, **p):
    """Synthetic record with per-session restraint R (sessions x 2)."""
    R = np.asarray(R, dtype=float)
    n = len(R)
    return {**p, "sessions": n, "seeds": list(seeds if seeds is not None else range(21, 21 + n)),
            "restraint": R.mean(0).tolist(), "restraint_se": (R.std(0, ddof=1) / math.sqrt(n)).tolist(),
            "converged_share": 0.5, "delta": {"mean": 0.5},
            "per_session": {"restraint": R.tolist(), "converged": [True, False] * (n // 2),
                            "t_learn": [1e6] * n, "delta": [0.5] * n}}


def sample(c_of_w, n=20, seed=0, long_shift=0.0, noise=0.005):
    """A full measure-check sample in which the collusive part at w is c_of_w(w)."""
    rng = np.random.default_rng(seed)
    out = []
    for w in MC.SN.SHARES:
        pair = 0.9 + noise * rng.standard_normal((n, 2))
        alone = pair.mean(1, keepdims=True) + c_of_w(w) + noise * rng.standard_normal((n, 2))
        out += [rec(pair, sigma_u=0.1, w=w, who="pair", cap=MC.CAP),
                rec(alone, sigma_u=0.1, w=w, who="only0", cap=MC.CAP)]
        if w in MC.SN.DIRECT:
            out.append(rec(pair + c_of_w(w) * 0.9, sigma_u=0.1, w=w, who="solo_level0", cap=MC.CAP))
        if w == MC.CAP_W:
            out += [rec(pair, sigma_u=0.1, w=w, who="pair", cap=MC.CAP_LONG),
                    rec(alone + long_shift + noise * rng.standard_normal((n, 2)), sigma_u=0.1, w=w, who="only0", cap=MC.CAP_LONG)]
    return out


def test_paired_se():
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 1.0, 2.5, 3.0])
    d, se = MC.paired(a, b)
    diff = a - b
    assert d == pytest.approx(diff.mean())
    assert se == pytest.approx(diff.std(ddof=1) / 2)
    # perfectly correlated sessions: the paired SE vanishes, the unpaired one does not
    assert MC.paired(a, a - 0.1) == pytest.approx((0.1, 0.0))


def test_session_c_has_the_old_mean_and_reports_the_old_se():
    rng = np.random.default_rng(1)
    pair, alone = rec(0.9 + 0.01 * rng.standard_normal((20, 2))), rec(0.95 + 0.01 * rng.standard_normal((20, 2)))
    for r, who in ((pair, "pair"), (alone, "only0")):
        r.update(sigma_u=0.1, w=0.0, who=who)
    old = MC.SN.collusive_part({MC.SN.key(r): r for r in (pair, alone)}, 0.1, 0.0)
    assert MC.c_sessions(pair, alone).mean() == pytest.approx(old["c"])
    assert MC.old_se(pair, alone) == pytest.approx(old["c_se"])
    with pytest.raises(ValueError):
        MC.c_sessions(pair, rec(np.ones((20, 2)), seeds=range(1, 21)))


def test_ols_slope_and_bootstrap():
    x = np.array(MC.SN.SHARES)
    assert MC.ols_slope(x, 0.03 + 0.06 * x) == pytest.approx(0.06)
    # no session variation -> zero bootstrap error
    cmat = np.tile((0.03 + 0.06 * x)[:, None], (1, 20))
    assert MC.boot_slope_se(x, cmat, B=200) == pytest.approx(0.0, abs=1e-12)
    # independent noise per point: bootstrap SE close to the analytic SE of the OLS slope
    rng = np.random.default_rng(2)
    n, sd = 200, 0.01
    cmat = (0.03 + 0.06 * x)[:, None] + sd * rng.standard_normal((5, n))
    analytic = sd / math.sqrt(n) / math.sqrt(((x - x.mean()) ** 2).sum())
    se = MC.boot_slope_se(x, cmat, B=2000)
    assert se == pytest.approx(analytic, rel=0.15)
    assert MC.boot_slope_se(x, cmat, B=2000) == se                 # fixed bootstrap seed


def test_rule_labels():
    assert MC.r1(0.05, 0.01) == "rises" and MC.r1(0.05, 0.03) == "no clear rise"
    assert MC.r2(0.06, 0.01) == "higher at w = 1" and MC.r2(0.01, 0.01) == "not clearly higher"
    assert MC.c1(0.005, 0.01) == "cap-robust" and MC.c1(-0.03, 0.01) == "cap-sensitive"


def test_analyze_rising_curve_cap_robust():
    v = MC.analyze(sample(lambda w: 0.03 + 0.06 * w))
    assert v["R1"] == "rises" and v["slope"]["slope"] == pytest.approx(0.06, abs=0.005)
    assert v["R2_c"] == "higher at w = 1" and v["R2_direct"] == "higher at w = 1"
    assert v["C1"] == "cap-robust"
    assert v["curve"]["0.5"]["pair_term"][0] == pytest.approx(0.9, abs=0.005)


def test_analyze_flat_curve_cap_sensitive():
    v = MC.analyze(sample(lambda w: 0.03, long_shift=0.03))
    assert v["R1"] == "no clear rise" and v["R2_c"] == "not clearly higher"
    assert v["R2_direct"] == "not clearly higher" and v["C1"] == "cap-sensitive"
    assert v["cap"]["dc"] == pytest.approx(0.03, abs=0.005)
    assert "## " not in MC.report_md(v) and "cap-sensitive" in MC.report_md(v)


def test_partial_sample_is_not_run():
    v = MC.analyze(sample(lambda w: 0.03)[:4])
    assert v["R1"] == "not run" and v["C1"] == "not run"


def test_plan_and_params():
    p = MC.plan()
    assert len(p) == 14 and len({MC.key(x) for x in p}) == 14
    order = [MC.key(x) for x in p]
    assert order.index((0.1, 1.0, "solo_level0", MC.CAP)) > order.index((0.1, 1.0, "only0", MC.CAP))
    records = sample(lambda w: 0.04)
    solo = MC.params_for(dict(sigma_u=0.1, w=1.0, who="solo_level0", cap=MC.CAP), records)
    alone = next(r for r in records if MC.key(r) == (0.1, 1.0, "only0", MC.CAP))
    assert solo.fixed[1] == pytest.approx(alone["restraint"][0]) and solo.max_periods == MC.CAP
    long = MC.params_for(dict(sigma_u=0.1, w=0.75, who="pair", cap=MC.CAP_LONG), records, max_periods=1e6)
    assert long.max_periods == 4_000_000 and long.price_grid == "per_value" and long.noise_shared == 0.75
    assert list(MC.SEEDS[:2]) == [21, 22]


def test_smoke_out():
    assert MC.smoke_out(SimpleNamespace(out="measure_checks", max_periods=1e6)) == "measure_checks_smoke"
    assert MC.smoke_out(SimpleNamespace(out="measure_checks", max_periods=None)) == "measure_checks"
