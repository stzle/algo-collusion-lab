"""Q-loss (pre-registration 13): generic MDP core, the deviator's model, and the script."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import _dgj_common as G  # noqa: E402
from lab import dgj, qloss as QL  # noqa: E402

spec = importlib.util.spec_from_file_location("qloss_script", ROOT / "scripts" / "19_qloss.py")
SC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SC)


# ------------------------------------------------------------------ generic core

def two_state():
    """rho = 0.9. State 0: a0 stays (reward 1), a1 moves to state 1 (reward 0).
    State 1: a0 pays 2 and moves to either state with 1/2, a1 pays 1 and stays.
    By hand: V(a0, a0) = (10, 130/11), V(a1, a0) = (360/29, 400/29) (optimal; the myopic
    policy is (a0, a0)), V(a1, a1) = (9, 10) with stationary distribution (0, 1).
    Stationary distribution of (a1, a0): mu0 = mu1 / 2, so (1/3, 2/3)."""
    P = np.zeros((2, 2, 2))
    P[0, 0] = [1, 0]
    P[0, 1] = [0, 1]
    P[1, 0] = [0.5, 0.5]
    P[1, 1] = [0, 1]
    R = np.array([[1.0, 0.0], [2.0, 1.0]])
    return P, R, 0.9


def test_two_state_mdp_matches_the_closed_form():
    P, R, rho = two_state()
    V, pol, _ = QL.best_response(P, R, rho)
    assert list(pol) == [1, 0]
    assert np.allclose(V, [360 / 29, 400 / 29], rtol=0, atol=1e-12)
    assert np.allclose(QL.policy_value(P, R, [0, 0], rho), [10, 130 / 11], atol=1e-12)
    assert np.allclose(QL.policy_value(P, R, [1, 1], rho), [9, 10], atol=1e-12)
    assert np.allclose(QL.stationary(QL._pi(P, [1, 0])), [1 / 3, 2 / 3], atol=1e-12)
    mu = QL.stationary(QL._pi(P, [1, 1]))
    assert np.allclose(mu, [0, 1], atol=1e-12)
    # Q-loss of (a1, a1) weighted by its own visits: (400/29 - 10) / 10 = 11/29
    assert mu @ (V - [9, 10]) / (mu @ [9, 10]) == pytest.approx(11 / 29, abs=1e-12)


def test_factored_transitions_equal_the_dense_ones():
    rng = np.random.default_rng(0)
    T = rng.random((3, 8, 4))
    T /= T.sum(axis=2, keepdims=True)
    marg = np.array([0.25, 0.75])
    P = QL.FactoredP(T, marg)
    dense = np.einsum("asc,e->sace", T, marg).reshape(8, 3, 8)
    R = rng.standard_normal((8, 3))
    V1, p1, _ = QL.best_response(P, R, 0.9)
    V2, p2, _ = QL.best_response(dense, R, 0.9)
    assert np.array_equal(p1, p2) and np.allclose(V1, V2)
    assert np.allclose(P.pi(p1), QL._pi(dense, p1))


def test_stationary_refuses_two_recurrent_classes():
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        QL.stationary(np.eye(2))


@pytest.mark.skip(reason="placeholder: Renat's hand-solved 2-state, 2-action example goes here")
def test_renats_hand_example():
    P = np.zeros((2, 2, 2))          # fill in: P[s, a] = next-state probabilities
    R = np.zeros((2, 2))             # fill in: R[s, a]
    rho = 0.9                        # fill in
    V_star_by_hand, policy_by_hand = None, None
    V, pol, _ = QL.best_response(P, R, rho)
    assert list(pol) == policy_by_hand and np.allclose(V, V_star_by_hand)


# ------------------------------------------------------------------ the deviator's MDP

def market(**kw):
    prm = G.make_params(sigma_u=0.1, grid="nash", grid_width=0.3, monitor="realized",
                        price_grid="per_value", **kw)
    return prm, dgj.setup(prm)


def test_best_response_to_a_fixed_nash_rival_is_static_nash():
    prm, env = market(fixed=(None, 1.0))
    mm = (0.0, env["bm"]["nash"]["lam"])
    mid = prm.n_x // 2
    assert env["chi_grid"][0, mid] == pytest.approx(env["bm"]["nash"]["chi"][0], rel=1e-12)
    Q = env["Q0"]
    r = QL.analyze(env, prm, mm, Q, 0)
    assert np.all(r["policy_br"] == mid)                  # Nash chi in every state
    assert np.array_equal(r["policy_br"], r["policy_myopic"])
    assert r["q_loss"] == pytest.approx(r["myopic_gain_dynamic"], abs=1e-12)
    assert r["bellman_residual"] < 1e-8


def random_tables(prm, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((prm.I, prm.n_p * prm.n_v * prm.n_v, prm.n_x))


@pytest.mark.parametrize("kw", [dict(),                                              # exact information
                                dict(sigma_eps=(0.5, 0.5), noise_shared=0.5),       # w != 0
                                dict(sigma_eps=(0.5, 0.25), sigma_u=100.0)])        # wide price noise
def test_model_matches_the_engine_path(kw):
    """Random (state-dependent) tables for both; the model's long-run averages must match
    10^6 periods of dgj._greedy_path for the learned policy and the best response."""
    kw = dict(kw)
    su = kw.pop("sigma_u", 0.1)
    prm = G.make_params(sigma_u=su, grid="nash", grid_width=0.3, monitor="realized",
                        price_grid="per_value", **kw)
    env = dgj.setup(prm)
    Q = random_tables(prm, 3)
    mm = (0.01 * env["lam0"], 1.03 * env["lam0"])
    c = QL.model_vs_sim(env, prm, mm, Q, 1, periods=1_000_000, seed=5)
    for name in ("learned", "br"):
        x = c[name]
        assert abs(x["sim"] - x["model"]) < 4 * x["sim_se"], (name, x)
        assert x["tv"] < 0.03, (name, x)
    d = c["diff"]
    assert d["model"] > 0 and abs(d["sim"] - d["model"]) < 4 * d["sim_se"], d


def test_own_signal_monitoring_is_refused():
    prm, env = market()
    prm.monitor = "own_signal"
    with pytest.raises(ValueError):
        QL.build(env, prm, (0.0, env["lam0"]), env["Q0"], 0)


# ------------------------------------------------------------------ script

def test_parts_and_verdict_stub():
    def rec(learners, q):
        return {"spec": {f"spec{i + 1}": {"per_session": {"q_loss": q[i]},
                                          "summary": {k: {"mean": 0.0, "se": 0.0} for k in
                                                      ("q_loss", "myopic_gain_static", "myopic_gain_dynamic",
                                                       "static_deviation_gain")}}
                         for i in learners}}
    pts = {"exact_pair": rec([0, 1], {0: [0.10, 0.12], 1: [0.14, 0.12]}),
           "exact_only0": rec([0], {0: [0.02, 0.04]})}
    p = SC.parts(pts)["exact"]
    assert p["pair"]["mean"] == pytest.approx(0.12) and p["lone"]["mean"] == pytest.approx(0.03)
    assert p["pair_part"]["mean"] == pytest.approx(0.09)
    v = SC.verdicts(pts)
    assert "not yet" in v["rules"] and set(v["points"]) == set(pts)
    assert "Q-loss" in SC.report_md(v)
