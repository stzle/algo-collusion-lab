"""Checks for the analytical benchmarks (dgj_benchmarks.py).  Run: pytest -q"""
import numpy as np
import pytest

from lab import dgj_benchmarks as bmk

VG = bmk.value_grid(6)


@pytest.mark.parametrize("I", [2, 3])
def test_symmetric_perfect_information_closed_forms(I):
    """DGJ Section 4.2: chi_N = 1/((I+1) lam), chi_M = 1/(2 I lam)."""
    b = bmk.benchmarks(VG, [0.0] * I, xi=500.0, theta=0.1, sigma_u=0.1)
    lamN, lamM = b["nash"]["lam"], b["cartel"]["lam"]
    assert np.allclose(b["nash"]["chi"], 1 / ((I + 1) * lamN), rtol=1e-9)
    assert np.allclose(b["cartel"]["chi"], 1 / (2 * I * lamM), rtol=1e-9)


def test_kyle_limit():
    """xi = 0 and one speculator is Kyle (1985): lam = sigma_v / (2 sigma_u),
    order = (v - v_bar) * sigma_u / sigma_v."""
    for su in (0.5, 2.0):
        b = bmk.benchmarks(VG, [0.0], xi=0.0, theta=0.1, sigma_u=su)
        sv = np.sqrt(b["var_v"])
        assert b["nash"]["lam"] == pytest.approx(sv / (2 * su), rel=1e-8)
        assert b["nash"]["chi"][0] == pytest.approx(su / sv, rel=1e-8)


def test_large_xi_price_impact_is_one_over_xi():
    b = bmk.benchmarks(VG, [0.0, 0.0], xi=500.0, theta=0.1, sigma_u=100.0)
    assert b["nash"]["lam"] == pytest.approx(1 / 500.0, rel=1e-6)


@pytest.mark.parametrize("eps", [(0.0, 0.0), (0.5, 0.5), (0.0, 0.7), (0.3, 1.0)])
def test_nash_has_no_profitable_unilateral_deviation(eps):
    b = bmk.benchmarks(VG, eps, xi=500.0, theta=0.1, sigma_u=1.0)
    chi, lam = b["nash"]["chi"], b["nash"]["lam"]
    base = bmk.profit_given_chi(chi, b["C"], b["S"], lam)
    for i in range(2):
        for step in (-0.05, -0.01, 0.01, 0.05):
            dev = chi.copy()
            dev[i] *= 1 + step
            assert bmk.profit_given_chi(dev, b["C"], b["S"], lam)[i] < base[i]


@pytest.mark.parametrize("eps", [(0.0, 0.0), (0.5, 0.5), (0.0, 0.7)])
def test_cartel_beats_nash_jointly(eps):
    b = bmk.benchmarks(VG, eps, xi=500.0, theta=0.1, sigma_u=1.0)
    assert b["cartel"]["profit"].sum() > b["nash"]["profit"].sum()


def test_cartel_corner_when_one_agent_is_perfectly_informed():
    """If speculator 1 sees v exactly, speculator 2's signal adds only noise:
    the joint-profit optimum lets speculator 1 trade alone."""
    b = bmk.benchmarks(VG, (0.0, 0.5), xi=500.0, theta=0.1, sigma_u=0.1)
    assert b["cartel"]["chi"][1] == pytest.approx(0.0, abs=1e-8)
    assert b["cartel"]["chi"][0] == pytest.approx(1 / (2 * b["cartel"]["lam"]), rel=1e-8)


def test_signal_matrix_and_moments_against_monte_carlo():
    rng = np.random.default_rng(0)
    T = bmk.signal_matrix(VG, 0.6)
    assert np.allclose(T.sum(axis=1), 1.0)
    assert np.allclose(bmk.signal_matrix(VG, 0.0), np.eye(6))
    C, S, _ = bmk.moments(VG, np.array([0.0, 0.6]))
    n = 400_000
    j = rng.integers(0, 6, n)
    v = VG[j]
    s2 = VG[np.searchsorted(bmk.bin_edges(VG), v + 0.6 * rng.standard_normal(n))]
    d1, d2, w = v - 1, s2 - 1, v - 1
    assert C[1] == pytest.approx(np.mean(d2 * w), abs=0.01)
    assert S[1, 1] == pytest.approx(np.mean(d2 * d2), abs=0.01)
    assert S[0, 1] == pytest.approx(np.mean(d1 * d2), abs=0.01)


@pytest.mark.parametrize("eps", [(0.0, 0.0), (0.5, 0.5), (0.0, 0.7)])
def test_restraint_is_one_at_nash(eps):
    """At the Nash benchmark every speculator is at its best response."""
    b = bmk.benchmarks(VG, eps, xi=500.0, theta=0.1, sigma_u=0.1)
    r = bmk.restraint(b["nash"]["chi"], b["C"], b["S"], b["nash"]["lam"])
    assert np.allclose(r["restraint"], 1.0, atol=1e-9)
    assert np.allclose(r["deviation_gain"], 0.0, atol=1e-9)


def test_restraint_at_symmetric_cartel_is_two_thirds():
    """Perfect information: chi_M = 1/(4 lam), best response 1/(2 lam) - chi_M/2 = 3/(8 lam)."""
    b = bmk.benchmarks(VG, (0.0, 0.0), xi=500.0, theta=0.1, sigma_u=0.1)
    r = bmk.restraint(b["cartel"]["chi"], b["C"], b["S"], b["nash"]["lam"])
    assert np.allclose(r["restraint"], 2 / 3, rtol=1e-6)
    assert np.all(r["deviation_gain"] > 0)
