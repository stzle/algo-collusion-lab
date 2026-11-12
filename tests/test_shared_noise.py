"""Shared noise (pre-registration 7): engine, benchmarks and the decision rules."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import _dgj_common as G  # noqa: E402
from lab import dgj, dgj_benchmarks as bmk  # noqa: E402

spec = importlib.util.spec_from_file_location("shared", ROOT / "scripts" / "15_shared_noise.py")
SN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SN)


def test_moments_independent_and_identical_limits():
    v = bmk.value_grid(6)
    J = bmk.joint_signal_matrix(v, 0.5, 0.5, 0.0)
    T = bmk.signal_matrix(v, 0.5)
    assert np.allclose(J, np.einsum("ja,jb->jab", T, T))
    C, S, _ = bmk.moments(v, np.array([0.5, 0.5]), shared=1.0)
    assert S[0, 1] == pytest.approx(S[0, 0], abs=1e-3)        # identical signals
    C0, S0, _ = bmk.moments(v, np.array([0.5, 0.5]))
    assert np.allclose(C, C0) and S[0, 0] == pytest.approx(S0[0, 0])


def test_full_sharing_restores_the_cartel_gain_of_exact_information():
    v = bmk.value_grid(6)
    b = bmk.benchmarks(v, (0.5, 0.5), 500.0, 0.1, 0.1, shared=1.0)
    gain = b["cartel"]["profit"].sum() / b["nash"]["profit"].sum() - 1
    assert gain == pytest.approx(0.125, abs=0.002)


def test_engine_draws_match_the_analytical_profits():
    prm = G.make_params(sigma_u=0.1, sigma_eps=(0.5, 0.5), noise_shared=0.5)
    env = dgj.setup(prm)
    chi = env["bm"]["nash"]["chi"]
    prof, _ = dgj.run_fixed_strategies(7, chi, env["v_grid"], env["edges"], env["sig_eps"], prm.v_bar,
                                       prm.xi, prm.theta, prm.sigma_u, prm.Tm, env["lam0"], 1_000_000)
    assert np.allclose(prof, env["bm"]["nash"]["profit"], rtol=0.01)


def test_no_sharing_leaves_the_engine_unchanged():
    a = dgj.setup(G.make_params(sigma_eps=(0.5, 0.5)))
    b = dgj.setup(G.make_params(sigma_eps=(0.5, 0.5), noise_shared=0.0))
    assert np.array_equal(a["sig_eps"], b["sig_eps"]) and len(a["sig_eps"]) == 2
    with pytest.raises(ValueError):
        dgj.setup(G.make_params(sigma_eps=(0.5, 0.5), noise_shared=1.5))


def test_v1_thresholds_are_renats():
    assert SN.v1({"c": 0.07, "c_se": 0.003}) == "confirmed"
    assert SN.v1({"c": 0.029, "c_se": 0.003}) == "wrong"
    assert SN.v1({"c": 0.045, "c_se": 0.003}) == "partly right / inconclusive"
    assert SN.v1({"c": 0.061, "c_se": 0.04}) == "partly right / inconclusive"   # not significant


def test_v2_needs_a_rise_over_independent_noise():
    d0 = {"c": 0.047, "c_se": 0.002}
    assert SN.v2({"c": 0.048, "c_se": 0.002}, d0, 100.0) == "wrong"
    assert SN.v2({"c": 0.07, "c_se": 0.002}, d0, 100.0) == "confirmed"
    assert SN.v2({"c": 0.06, "c_se": 0.002}, d0, 100.0) == "partly right / inconclusive"


def test_plan():
    p = SN.plan([0.1, 100.0])
    assert len(p) == 2 * (5 * 2 + 2) and len({SN.key(x) for x in p}) == len(p)
    # direct measurements come after the lone learner they depend on
    order = [SN.key(x) for x in p]
    assert order.index((0.1, 1.0, "solo_level0")) > order.index((0.1, 1.0, "only0"))
