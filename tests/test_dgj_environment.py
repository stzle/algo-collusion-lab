"""Checks that the simulated market behaves like the model.  Run: pytest -q"""
import numpy as np
import pytest

from lab import dgj


@pytest.mark.parametrize("sigma_u,eps", [(0.1, (0.0, 0.0)), (100.0, (0.0, 0.0)), (0.1, (0.0, 0.5))])
@pytest.mark.parametrize("kind", ["nash", "cartel"])
def test_fixed_strategies_reproduce_benchmark_profits(sigma_u, eps, kind):
    """Same market, same adaptive market maker, but fixed benchmark strategies:
    simulated average profits must equal the analytical ones."""
    prm = dgj.DGJParams(sigma_u=sigma_u, sigma_eps=eps)
    env = dgj.setup(prm)
    b = env["bm"][kind]
    prof, lam = dgj.run_fixed_strategies(3, b["chi"], env["v_grid"], env["edges"], env["sig_eps"],
                                         prm.v_bar, prm.xi, prm.theta, sigma_u, prm.Tm,
                                         env["lam0"], 1_000_000)
    assert lam == pytest.approx(b["lam"], rel=1e-4)
    tol = 0.01 * np.abs(b["profit"]).max()
    assert np.allclose(prof, b["profit"], atol=tol)


def test_benchmarks_are_on_the_action_grid():
    """iota = 0.1 and n_x = 13: cartel at index 1, Nash at index n_x - 2."""
    env = dgj.setup(dgj.DGJParams(n_x=13))
    g = env["chi_grid"][0]
    assert g[1] == pytest.approx(env["bm"]["cartel"]["chi"][0], rel=1e-9)
    assert g[-2] == pytest.approx(env["bm"]["nash"]["chi"][0], rel=1e-9)


def short(prm_kw, seeds=(1, 2)):
    prm = dgj.DGJParams(max_periods=400_000, conv_window=10**9, eval_periods=20_000, **prm_kw)
    return dgj.simulate(prm, list(seeds))


def test_simulation_is_reproducible():
    a = short({})
    b = short({})
    assert np.array_equal(a["Q"], b["Q"]) and np.array_equal(a["profit"], b["profit"])


def test_variants_coincide_without_signal_noise():
    a = short({"monitor": "realized"})
    b = short({"monitor": "own_signal"})
    assert np.array_equal(a["Q"], b["Q"])


def test_market_maker_learns_price_impact():
    r = short({"sigma_u": 0.1})
    assert np.allclose(r["mm"][:, 1], 1 / 500.0, rtol=1e-3)


def test_both_fixed_at_nash_reproduces_nash_profits():
    """The fixed-strategy path inside run_session: both speculators fixed at Nash
    must earn the analytical Nash profits (same check as run_fixed_strategies)."""
    prm = dgj.DGJParams(sigma_u=0.1, sigma_eps=(0.0, 0.5), fixed=(1.0, 1.0),
                        conv_window=1000, eval_periods=1_000_000)
    r = dgj.simulate(prm, [4])
    b = r["env"]["bm"]["nash"]
    assert np.allclose(r["chi"][0], b["chi"], rtol=1e-12)
    assert np.allclose(r["profit"][0], b["profit"], atol=0.01 * b["profit"].max())


def test_fixed_agent_never_learns():
    prm = dgj.DGJParams(fixed=(None, 1.05), max_periods=300_000, conv_window=10**9,
                        eval_periods=10_000)
    r = dgj.simulate(prm, [1])
    env = r["env"]
    assert np.array_equal(r["Q"][0][1], env["Q0"][1])
    assert r["chi"][0][1] == pytest.approx(1.05 * env["bm"]["nash"]["chi"][1], rel=1e-12)
    assert not np.array_equal(r["Q"][0][0], env["Q0"][0])


def test_nash_centred_grid():
    env = dgj.setup(dgj.DGJParams(sigma_eps=(0.0, 0.5), grid="nash", grid_width=0.3))
    g = env["chi_grid"] / env["bm"]["nash"]["chi"][:, None]
    assert np.allclose(g[:, 0], 0.7) and np.allclose(g[:, -1], 1.3) and np.allclose(g[:, 6], 1.0)
    base = dgj.setup(dgj.DGJParams(sigma_eps=(0.0, 0.5)))
    assert np.array_equal(env["p_grid"], base["p_grid"])


def test_default_path_unchanged_by_control_options():
    """Adding the control options must not change default results (reproducibility
    with runs made before they existed)."""
    a = dgj.simulate(dgj.DGJParams(max_periods=300_000, conv_window=10**9, eval_periods=10_000), [3])
    b = dgj.simulate(dgj.DGJParams(max_periods=300_000, conv_window=10**9, eval_periods=10_000,
                                   fixed=(None, None), grid="benchmarks"), [3])
    assert np.array_equal(a["Q"], b["Q"])


def test_no_convergence_before_exploration_has_died_down():
    """A session may only count as converged once every signal value has been seen
    n_req times (exploration below conv_max_explore)."""
    prm = dgj.DGJParams(beta=1e-4, conv_window=2_000, conv_max_explore=0.01,
                        max_periods=5_000_000, eval_periods=1_000)
    n_req = dgj._n_req(prm)
    assert n_req == int(np.ceil(np.log(100) / 1e-4))
    r = dgj.simulate(prm, [1, 2])
    for k in range(2):
        if r["converged"][k]:
            assert r["t_learn"][k] >= prm.n_v * n_req + prm.conv_window
