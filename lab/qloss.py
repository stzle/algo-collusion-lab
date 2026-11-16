"""
Q-loss of the learned strategies: how much one speculator would gain by switching to its
exact dynamic best response while the others keep their learned (or fixed) strategies.
The method is the "Q-loss" check of Calvano, Calzolari, Denicolò & Pastorello (2020, AER)
(section number unverified): compare the value of the learned greedy policy with the
optimal value, both computed exactly from the model, weighted by how often the learned
play visits each state.

Deviator's model (monitor = "realized"; the others and the pricing rule are frozen)
  state     s = (pb * n_v + m) * n_v + sig: price bin of last period's price on the grid of
            the lagged value m, m = last period's realised value, sig = own current signal.
            With monitor "realized" both speculators share (pb, m), so the rival's state is
            the deviator's state with its own signal, and the deviator's problem is an exact
            MDP on its own state.
  period    value k and rival signal sig_j ~ P(k, sig_j | sig) (values iid uniform, so this
            does not depend on pb or m); orders x = chi_grid[i, a] (v_sig - v_bar) and
            x_j = chi_j(pb, m, sig_j) (v_sig_j - v_bar); price p = v_bar + p_int +
            p_slope (x + x_j + u), u ~ N(0, sigma_u^2)
  reward    E[(v_k - p) x] = (v_k - v_bar - p_int - p_slope (x + x_j)) x
  next      pb' = bin of p on the grid of k (as dgj._price_bin rounds, end bins take the
            tails), m' = k, sig' ~ marginal of the own signal (fresh value, independent)
  discount  rho

Generic core (any small MDP, P dense (S, A, S) or FactoredP): policy_value, best_response,
stationary.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.special import ndtr

from . import dgj
from . import dgj_benchmarks as bmk
from .dgj import _draw_signals, _greedy_path


# --------------------------------------------------------------------------
# Generic MDP core
# --------------------------------------------------------------------------
class FactoredP:
    """Transitions P[a][s, c * n_e + e] = T[a, s, c] * marg[e]: the next state is an outcome c
    (here price bin x value) and an independent draw e (here the next own signal).
    Stores (A, S, C) instead of (S, A, S)."""

    def __init__(self, T, marg):
        self.T = np.asarray(T, dtype=float)
        self.marg = np.asarray(marg, dtype=float)
        self.n_actions, self.n_states = self.T.shape[:2]

    def pi(self, policy) -> np.ndarray:
        """Dense (S, S) transition matrix of a deterministic policy."""
        Tp = self.T[np.asarray(policy), np.arange(self.n_states)]
        return (Tp[:, :, None] * self.marg[None, None, :]).reshape(self.n_states, -1)

    def expect(self, V) -> np.ndarray:
        """E[V(s') | s, a], shape (S, A)."""
        w = np.asarray(V).reshape(-1, len(self.marg)) @ self.marg
        return (self.T @ w).T


def _pi(P, policy):
    if isinstance(P, FactoredP):
        return P.pi(policy)
    return P[np.arange(P.shape[0]), np.asarray(policy)]


def _expect(P, V):
    if isinstance(P, FactoredP):
        return P.expect(V)
    return P @ V


def policy_value(P, R, policy, rho) -> np.ndarray:
    """Value of a deterministic stationary policy: solves (I - rho P_pi) V = r_pi."""
    policy = np.asarray(policy)
    S = R.shape[0]
    r = R[np.arange(S), policy]
    return np.linalg.solve(np.eye(S) - rho * _pi(P, policy), r)


def best_response(P, R, rho, policy0=None, tol=1e-12, max_iter=1000):
    """Optimal value and policy by policy iteration (exact up to the linear solves).
    A state only switches action when that gains more than tol * max(1, |V|), so ties keep
    the current action and the loop ends. Returns (V*, policy*, iterations)."""
    S = R.shape[0]
    pol = np.argmax(R, axis=1) if policy0 is None else np.asarray(policy0).copy()
    for it in range(1, max_iter + 1):
        V = policy_value(P, R, pol, rho)
        Qv = R + rho * _expect(P, V)
        best = np.argmax(Qv, axis=1)
        gain = Qv[np.arange(S), best] - Qv[np.arange(S), pol]
        switch = gain > tol * np.maximum(1.0, np.abs(V))
        if not switch.any():
            return V, pol, it
        pol[switch] = best[switch]
    raise RuntimeError("policy iteration did not converge")


def stationary(P_pi) -> np.ndarray:
    """Stationary distribution mu = mu P_pi. Needs a single recurrent class (checked)."""
    S = P_pi.shape[0]
    A = np.asarray(P_pi, dtype=float).T - np.eye(S)
    A[-1] = 1.0                       # one balance equation is redundant: normalise instead
    b = np.zeros(S)
    b[-1] = 1.0
    mu = np.linalg.solve(A, b)
    if mu.min() < -1e-9 or np.abs(mu @ P_pi - mu).max() > 1e-9:
        raise ValueError("no unique stationary distribution (more than one recurrent class?)")
    mu = np.clip(mu, 0.0, None)
    return mu / mu.sum()


# --------------------------------------------------------------------------
# The deviator's MDP in the DGJ market
# --------------------------------------------------------------------------
def _check(prm):
    if prm.I != 2:
        raise ValueError("the Q-loss model is written for two speculators")
    if prm.monitor != "realized":
        # with own_signal monitoring the rival's state holds its own past signal, which the
        # deviator does not see: its problem is then not an MDP on its own state
        raise ValueError("Q-loss needs monitor = 'realized'")


def signal_law(env, prm, i):
    """W[sig_i, k, sig_j] = P(value k, rival signal sig_j | own signal sig_i) and the marginal
    of the own signal. Exact for independent noise; with shared noise integrated on a fine
    grid (bmk.joint_signal_matrix)."""
    q1, q2 = (float(q) for q in prm.sigma_eps)
    n_eta = 20_001 if prm.noise_shared > 0 else 3
    J = bmk.joint_signal_matrix(env["v_grid"], q1, q2, float(prm.noise_shared), n_eta=n_eta)
    if i == 1:
        J = J.transpose(0, 2, 1)                        # J[k, own, rival]
    joint = J / J.shape[0]                              # uniform values
    joint /= joint.sum()
    marg = joint.sum(axis=(0, 2))
    W = joint.transpose(1, 0, 2) / marg[:, None, None]
    return W, marg


def rival_chi(env, prm, Q, j) -> np.ndarray:
    """chi of speculator j in every (pb * n_v + m, sig_j): greedy on Q_j or its fixed chi."""
    PM = prm.n_p * prm.n_v
    if env["fixed_mask"][j]:
        return np.full((PM, prm.n_v), env["fixed_chi"][j])
    return env["chi_grid"][j, np.argmax(Q[j], axis=1)].reshape(PM, prm.n_v)


def _bin_probs(c, k, env, prm, sd):
    """P(price bin b) on the grid of value k for centred mean prices c; last axis = b."""
    n_p = prm.n_p
    p0, dp = env["p0"][k] - prm.v_bar, env["dp"][k]
    if sd <= 0.0:
        b = np.clip(np.floor((c - p0) / dp + 0.5), 0, n_p - 1).astype(np.int64)
        return (b[..., None] == np.arange(n_p)).astype(float)
    # bin b covers [p0 + (b - 1/2) dp, p0 + (b + 1/2) dp); the end bins take the tails
    bnd = p0 + (np.arange(1, n_p) - 0.5) * dp
    F = ndtr((bnd - c[..., None]) / sd)
    shape = c.shape + (1,)
    F = np.concatenate([np.zeros(shape), F, np.ones(shape)], axis=-1)
    return np.clip(np.diff(F, axis=-1), 0.0, None)


def build(env, prm, mm, Q, i) -> dict:
    """Deviator i's MDP against the frozen rival and pricing rule mm = (p_int, p_slope).
    Returns P (FactoredP), R (S, A) and the learned greedy policy."""
    _check(prm)
    if env["fixed_mask"][i]:
        raise ValueError("the deviator must be a learner")
    n_p, n_v, n_x = prm.n_p, prm.n_v, prm.n_x
    PM = n_p * n_v
    d = env["v_grid"] - prm.v_bar
    p_int, p_slope = float(mm[0]), float(mm[1])
    W, marg = signal_law(env, prm, i)
    x_own = d[:, None] * env["chi_grid"][i][None, :]                 # (sig, a)
    x_riv = rival_chi(env, prm, Q, 1 - i) * d[None, :]               # (pm, sig_j)
    c = p_int + p_slope * (x_own[None, :, :, None] + x_riv[:, None, None, :])   # (pm, sig, a, sig_j)

    # expected reward
    Ed = np.einsum("skj,k->s", W, d)                                 # E[v - v_bar | sig]
    Wj = W.sum(axis=1)                                               # P(sig_j | sig)
    R = x_own[None] * (Ed[None, :, None] - np.einsum("psaj,sj->psa", c, Wj))
    R = R.reshape(PM * n_v, n_x)

    # transitions to (pb', m' = k); the next own signal is factored out
    sd = abs(p_slope) * prm.sigma_u
    T = np.empty((n_x, PM, n_v, n_p, n_v))
    for k in range(n_v):
        pr = _bin_probs(c, k, env, prm, sd)                          # (pm, sig, a, sig_j, b)
        T[..., k] = np.einsum("psajb,sj->apsb", pr, W[:, k, :])
    T = T.reshape(n_x, PM * n_v, n_p * n_v)
    return {"P": FactoredP(T, marg), "R": R, "policy": np.argmax(Q[i], axis=1),
            "marg": marg, "W": W}


def analyze(env, prm, mm, Q, i, model: dict | None = None) -> dict:
    """Q-loss and the myopic comparison for deviator i. Ratios are fractions (x100 for %);
    values in per-period units ((1 - rho) V), weighted by the stationary distribution mu of
    the learned play."""
    m = model or build(env, prm, mm, Q, i)
    P, R, pol, rho = m["P"], m["R"], m["policy"], prm.rho
    S = R.shape[0]
    ar = np.arange(S)
    V_pi = policy_value(P, R, pol, rho)
    V_br, pol_br, iters = best_response(P, R, rho, policy0=pol)
    pol_my = np.argmax(R, axis=1)
    V_my = policy_value(P, R, pol_my, rho)
    mu = stationary(P.pi(pol))
    mu_br = stationary(P.pi(pol_br))
    mu_my = stationary(P.pi(pol_my))
    r_pi = R[ar, pol]
    chiN = env["bm"]["nash"]["chi"][i]
    chi = env["chi_grid"][i]
    # Bellman residual of V* (should be at rounding level)
    resid = np.abs((R + rho * P.expect(V_br)).max(axis=1) - V_br).max()
    return {
        "q_loss": float(mu @ (V_br - V_pi) / (mu @ V_pi)),
        "value_learned": float((1 - rho) * mu @ V_pi),
        "value_br": float((1 - rho) * mu @ V_br),
        "value_myopic": float((1 - rho) * mu @ V_my),
        "myopic_gain_static": float(mu @ (R.max(axis=1) - r_pi) / (mu @ r_pi)),
        "myopic_gain_dynamic": float(mu @ (V_my - V_pi) / (mu @ V_pi)),
        "avg_profit_learned": float(mu @ r_pi),
        "avg_profit_br": float(mu_br @ R[ar, pol_br]),
        "avg_profit_myopic": float(mu_my @ R[ar, pol_my]),
        "br_differs": float(mu @ (pol_br != pol)),
        "br_is_myopic": float(mu @ (pol_br == pol_my)),
        "chi_learned_vs_nash": float(mu @ chi[pol] / chiN),
        "chi_br_vs_nash": float(mu_br @ chi[pol_br] / chiN),
        "bellman_residual": float(resid),
        "pi_iterations": int(iters),
        "policy": pol, "policy_br": pol_br, "policy_myopic": pol_my, "mu": mu, "mu_br": mu_br,
    }


# --------------------------------------------------------------------------
# Validation: play the policies through the engine's greedy path
# --------------------------------------------------------------------------
@njit(cache=True)
def _draw_paths(seed, H, v_grid, edges, sig_eps, I, sigma_u):
    """Value, signal and noise-trade sequences, drawn as the engine draws them."""
    np.random.seed(seed)
    n_v = v_grid.shape[0]
    vseq = np.empty(H + 1, dtype=np.int64)
    sigseq = np.empty((H + 1, I), dtype=np.int64)
    useq = np.empty(H)
    tmp = np.empty(I, dtype=np.int64)
    for h in range(H + 1):
        vseq[h] = np.random.randint(n_v)
        _draw_signals(vseq[h], v_grid, edges, sig_eps, I, tmp)
        for i in range(I):
            sigseq[h, i] = tmp[i]
    for h in range(H):
        useq[h] = sigma_u * np.random.standard_normal()
    return vseq, sigseq, useq


def one_hot(policy, n_x) -> np.ndarray:
    """Q-table whose greedy policy is `policy`."""
    Qd = np.zeros((len(policy), n_x))
    Qd[np.arange(len(policy)), policy] = 1.0
    return Qd


def simulate_check(env, prm, mm, Q, i, policies: dict, periods=1_000_000, seed=12345) -> dict:
    """Plays each policy of deviator i (the others frozen as learned or fixed) through
    dgj._greedy_path for `periods` periods with common random numbers. Returns realised
    average profits (both speculators), their s.e., and the deviator's state frequencies."""
    _check(prm)
    n_p, n_v = prm.n_p, prm.n_v
    v_grid, p0, dp = env["v_grid"], env["p0"], env["dp"]
    vseq, sigseq, useq = _draw_paths(seed, periods, v_grid, env["edges"], env["sig_eps"], prm.I,
                                     prm.sigma_u)
    s_start = ((n_p // 2) * n_v + vseq[0]) * n_v + sigseq[0]        # any start; burn-in negligible
    v = v_grid[vseq[:-1]]
    sgn = np.where(v >= prm.v_bar, 1.0, -1.0)
    out = {}
    for name, pol in policies.items():
        Qp = Q.copy()
        Qp[i] = one_hot(pol, prm.n_x)
        chi_out = np.empty((periods, prm.I))
        ptil_out = np.empty(periods)
        _greedy_path(Qp, v_grid, env["chi_grid"], p0, dp, n_p, prm.v_bar, float(mm[0]), float(mm[1]),
                     dgj.MONITOR[prm.monitor], s_start.astype(np.int64), vseq, sigseq, useq, 0.0, -1,
                     chi_out, ptil_out, env["fixed_mask"], env["fixed_chi"])
        p = prm.v_bar + ptil_out * sgn
        x = chi_out * (v_grid[sigseq[:-1]] - prm.v_bar)
        prof = (v - p)[:, None] * x
        # deviator's state in periods 1..H-1, rebuilt from last period's price and value
        pb = np.clip(np.floor((p[:-1] - p0[vseq[:-2]]) / dp[vseq[:-2]] + 0.5), 0, n_p - 1).astype(np.int64)
        s = (pb * n_v + vseq[:-2]) * n_v + sigseq[1:-1, i]
        out[name] = {"profit": prof.mean(axis=0), "profit_se": prof.std(axis=0, ddof=1) / np.sqrt(periods),
                     "freq": np.bincount(s, minlength=n_p * n_v * n_v) / len(s), "_prof": prof[:, i]}
    names = list(policies)
    for a in names[1:]:
        dlt = out[a]["_prof"] - out[names[0]]["_prof"]
        out[a]["diff_vs_" + names[0]] = float(dlt.mean())
        out[a]["diff_se"] = float(dlt.std(ddof=1) / np.sqrt(periods))
    for a in names:
        del out[a]["_prof"]
    return out


def model_vs_sim(env, prm, mm, Q, i, res: dict | None = None, periods=1_000_000, seed=12345) -> dict:
    """Model long-run averages (stationary distribution x reward) of the learned policy and
    the best response next to the simulated ones; tv = total-variation distance between the
    simulated state frequencies and the model's stationary distribution."""
    m = build(env, prm, mm, Q, i)
    res = res or analyze(env, prm, mm, Q, i, model=m)
    sim = simulate_check(env, prm, mm, Q, i, {"learned": res["policy"], "br": res["policy_br"]},
                         periods, seed)
    out = {}
    for name, mu, model in (("learned", res["mu"], res["avg_profit_learned"]),
                            ("br", res["mu_br"], res["avg_profit_br"])):
        out[name] = {"model": model, "sim": float(sim[name]["profit"][i]),
                     "sim_se": float(sim[name]["profit_se"][i]),
                     "sim_rival": float(sim[name]["profit"][1 - i]),
                     "tv": float(0.5 * np.abs(sim[name]["freq"] - mu).sum())}
    out["diff"] = {"model": res["avg_profit_br"] - res["avg_profit_learned"],
                   "sim": sim["br"]["diff_vs_learned"], "sim_se": sim["br"]["diff_se"]}
    return out


# --------------------------------------------------------------------------
# Robustness: the same deviation against a market maker that keeps learning
# --------------------------------------------------------------------------
BETA_FIRST = 1e3      # exploration exp(-beta n): only the first visit of each signal value explores


def _engine_profit(env, prm, Q0, seed, conv_window, eval_periods) -> np.ndarray:
    """Mean profit per speculator over the evaluation window of dgj.run_sessions with frozen
    tables (alpha = 0) and a learning market maker. With n_req = 1 the evaluation starts
    `conv_window` periods after every signal value has been seen once."""
    out = dgj.run_sessions(np.array([seed], dtype=np.int64), env["v_grid"], env["edges"], env["sig_eps"],
                           env["chi_grid"], env["p0"], env["dp"], prm.n_p, Q0, prm.v_bar, prm.xi,
                           prm.theta, prm.sigma_u, prm.rho, 0.0, BETA_FIRST, prm.Tm, env["lam0"],
                           dgj.MONITOR[prm.monitor], int(conv_window), 2**62, int(eval_periods),
                           env["fixed_mask"], env["fixed_chi"], 1)
    return out[3][0]


def learning_mm_check(env, prm, Q, i, policy_br, seed=777, eval_periods=1_000_000,
                      chunk=None, horizon=None) -> dict:
    """Deviation gain when the market maker keeps learning (dgj.run_sessions, alpha = 0, so
    no table changes; exploration only on first visits). Deviator i plays its one-hot best
    response, the rival its learned table; the baseline run uses the learned tables, with
    the same seed (common random numbers).

    settled: evaluation starts Tm periods in, so the market maker's window holds only
             deviation-period data; gain over `eval_periods`.
    path:    evaluation starts at once; profit per `chunk` periods up to `horizon`, from
             differences of runs with growing eval_periods (same seed, so the paths
             coincide). Caveat: run_session always restarts the market maker from the Nash
             rule (p_int = 0, p_slope = lam0) with an empty window, so the first Tm periods
             are priced by that rule, not by the session's saved rule."""
    chunk = chunk or prm.Tm // 5
    horizon = horizon or 5 * prm.Tm
    Q_dev = Q.copy()
    Q_dev[i] = one_hot(policy_br, prm.n_x)
    out = {}
    base = _engine_profit(env, prm, Q, seed, prm.Tm, eval_periods)
    dev = _engine_profit(env, prm, Q_dev, seed, prm.Tm, eval_periods)
    out["settled"] = {"profit_learned": base, "profit_br": dev, "gain": float(dev[i] / base[i] - 1.0),
                      "eval_periods": int(eval_periods)}
    ends = np.arange(chunk, horizon + 1, chunk)
    cum = {}
    for name, Q0 in (("learned", Q), ("br", Q_dev)):
        tot = np.array([_engine_profit(env, prm, Q0, seed, 1, e)[i] * e for e in ends])
        cum[name] = np.diff(np.concatenate([[0.0], tot])) / chunk
    out["path"] = {"chunk_end": ends, "profit_learned": cum["learned"], "profit_br": cum["br"],
                   "gain": cum["br"] / cum["learned"] - 1.0, "Tm": int(prm.Tm)}
    return out
