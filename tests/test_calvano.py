"""Checks for the Calvano et al. (2020) replication.  Run: pytest -q"""
import numpy as np
import pytest

from lab import calvano as cv

PRM = cv.CalvanoParams()


def test_benchmark_prices_match_literature():
    bm = cv.benchmarks(PRM)
    # values reported throughout the literature for this parameterisation
    assert bm["p_nash"] == pytest.approx(1.4729, abs=1e-3)
    assert bm["p_monopoly"] == pytest.approx(1.9250, abs=1e-3)
    assert bm["pi_monopoly"] > bm["pi_nash"]


def test_nash_is_best_response():
    pN = cv.nash_price(PRM)
    grid = np.linspace(pN - 0.2, pN + 0.2, 401)
    best = grid[np.argmax([cv.profits(p, pN, PRM)[0] for p in grid])]
    assert best == pytest.approx(pN, abs=2e-3)


def test_grid_matches_paper():
    prices = cv.price_grid(PRM)
    assert len(prices) == 15
    # Calvano et al. Fig. 6 deviates "to the 9th price (1.74)"
    assert prices[8] == pytest.approx(1.74, abs=5e-3)


def test_initial_q_is_discounted_average_profit():
    prices = cv.price_grid(PRM)
    PI = cv.profit_matrix(prices, PRM)
    Q0 = cv.initial_Q(PI, PRM.delta)
    assert Q0.shape == (2, 225, 15)
    assert Q0[0, 17, 4] == pytest.approx(PI[4, :, 0].mean() / (1 - PRM.delta))
    assert Q0[1, 3, 9] == pytest.approx(PI[:, 9, 1].mean() / (1 - PRM.delta))


def test_sessions_are_reproducible_and_parallel_equals_serial():
    prices = cv.price_grid(PRM)
    PI = cv.profit_matrix(prices, PRM)
    Q0 = cv.initial_Q(PI, PRM.delta)
    args = (PI, Q0, PRM.alpha, PRM.beta, PRM.delta, 1000, 300_000)
    a = cv.run_session(5, *args)
    b = cv.run_session(5, *args)
    assert np.array_equal(a[0], b[0]) and a[2] == b[2]
    Qs, conv, per, sf = cv.run_sessions(np.array([5, 6], dtype=np.int64), *args)
    assert np.array_equal(Qs[0], a[0]) and per[0] == a[2]


def test_short_replication_hits_published_delta():
    """30 sessions: mean Delta should be near the published 0.85."""
    prices = cv.price_grid(PRM)
    PI = cv.profit_matrix(prices, PRM)
    Q0 = cv.initial_Q(PI, PRM.delta)
    bm = cv.benchmarks(PRM)
    seeds = np.arange(1, 31, dtype=np.int64)
    Qs, conv, per, sf = cv.run_sessions(seeds, PI, Q0, PRM.alpha, PRM.beta, PRM.delta,
                                        PRM.conv_window, PRM.max_periods)
    assert conv.all()
    d = np.mean([cv.session_outcome(Qs[k], sf[k], PI, bm)["delta"] for k in range(30)])
    assert 0.78 < d < 0.92
