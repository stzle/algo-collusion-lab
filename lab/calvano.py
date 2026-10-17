"""
Calvano, Calzolari, Denicolò & Pastorello (2020, AER 110(10)):
"Artificial Intelligence, Algorithmic Pricing, and Collusion".

Two firms sell differentiated products with logit demand and set prices every
period. Each firm is an independent Q-learner whose state is last period's
price pair (one-period memory). Nobody tells the algorithms to collude.

This module is the validation step (Rung 1): the published baseline result is
an average profit gain of about Delta = 0.85, so a correct implementation can be
checked against a number in the literature.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from numba import njit, prange
from scipy.optimize import brentq, minimize_scalar


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass
class CalvanoParams:
    # demand and cost (paper baseline: a_i = 2, c_i = 1, a_0 = 0, mu = 1/4)
    a: float = 2.0
    a0: float = 0.0
    c: float = 1.0
    mu: float = 0.25
    delta: float = 0.95          # discount factor
    # price grid: m points; "aer" = [pN - xi(pM-pN), pM + xi(pM-pN)],
    #             "wp"  = [(1-xi) pN, (1+xi) pM] (2018 working-paper wording)
    m: int = 15
    xi: float = 0.1
    grid: str = "aer"
    # learning
    alpha: float = 0.15          # learning rate
    beta: float = 4e-6           # exploration decay: eps_t = exp(-beta t)
    conv_window: int = 100_000   # strategy unchanged this many periods = converged
    max_periods: int = 100_000_000

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Economics: demand, profits, Nash and monopoly benchmarks
# --------------------------------------------------------------------------
def demand(p1: float, p2: float, prm: CalvanoParams) -> tuple[float, float]:
    e1 = np.exp((prm.a - p1) / prm.mu)
    e2 = np.exp((prm.a - p2) / prm.mu)
    e0 = np.exp(prm.a0 / prm.mu)
    tot = e1 + e2 + e0
    return e1 / tot, e2 / tot


def profits(p1: float, p2: float, prm: CalvanoParams) -> tuple[float, float]:
    q1, q2 = demand(p1, p2, prm)
    return (p1 - prm.c) * q1, (p2 - prm.c) * q2


def nash_price(prm: CalvanoParams) -> float:
    """Symmetric Bertrand-Nash price: FOC 1 - (p - c)(1 - q)/mu = 0."""
    def foc(p):
        q, _ = demand(p, p, prm)
        return 1.0 - (p - prm.c) * (1.0 - q) / prm.mu
    return brentq(foc, prm.c + 1e-9, prm.c + 10.0)


def monopoly_price(prm: CalvanoParams) -> float:
    """Price that maximizes joint profit (symmetric by symmetry of the firms)."""
    res = minimize_scalar(lambda p: -sum(profits(p, p, prm)),
                          bounds=(prm.c, prm.c + 10.0), method="bounded",
                          options={"xatol": 1e-12})
    return float(res.x)


def price_grid(prm: CalvanoParams) -> np.ndarray:
    pN, pM = nash_price(prm), monopoly_price(prm)
    if prm.grid == "aer":
        lo, hi = pN - prm.xi * (pM - pN), pM + prm.xi * (pM - pN)
    elif prm.grid == "wp":
        lo, hi = (1 - prm.xi) * pN, (1 + prm.xi) * pM
    else:
        raise ValueError("grid must be 'aer' or 'wp'")
    return np.linspace(lo, hi, prm.m)


def benchmarks(prm: CalvanoParams) -> dict:
    pN, pM = nash_price(prm), monopoly_price(prm)
    return {"p_nash": pN, "p_monopoly": pM,
            "pi_nash": profits(pN, pN, prm)[0],
            "pi_monopoly": profits(pM, pM, prm)[0]}


def profit_matrix(prices: np.ndarray, prm: CalvanoParams) -> np.ndarray:
    """PI[a1, a2, firm] = profit of `firm` when firm 1 plays price a1, firm 2 plays a2."""
    m = len(prices)
    PI = np.empty((m, m, 2))
    for i in range(m):
        for j in range(m):
            PI[i, j] = profits(prices[i], prices[j], prm)
    return PI


def initial_Q(PI: np.ndarray, delta: float) -> np.ndarray:
    """Q0[firm, state, action] = expected profit against a uniformly random rival,
    discounted forever (the paper's initialisation). Identical across states."""
    m = PI.shape[0]
    Q0 = np.empty((2, m * m, m))
    Q0[0, :, :] = PI[:, :, 0].mean(axis=1)[None, :] / (1 - delta)
    Q0[1, :, :] = PI[:, :, 1].mean(axis=0)[None, :] / (1 - delta)
    return Q0


# --------------------------------------------------------------------------
# Q-learning (compiled with Numba)
# --------------------------------------------------------------------------
@njit(cache=True)
def _argmax(row):
    best = 0
    bv = row[0]
    for k in range(1, row.shape[0]):
        if row[k] > bv:
            bv = row[k]
            best = k
    return best


@njit(cache=True)
def _max(row):
    bv = row[0]
    for k in range(1, row.shape[0]):
        if row[k] > bv:
            bv = row[k]
    return bv


@njit(cache=True)
def run_session(seed, PI, Q0, alpha, beta, delta, conv_window, max_periods):
    """One learning session. State s = a1 * m + a2 (last period's price indices).

    Returns (Q, converged, periods, final_state).
    """
    np.random.seed(seed)
    m = PI.shape[0]
    Q = Q0.copy()
    s = np.random.randint(m) * m + np.random.randint(m)
    stable = 0
    t = 0
    converged = False
    while t < max_periods:
        eps = np.exp(-beta * t)
        # each firm explores independently with probability eps
        if np.random.random() < eps:
            b1 = np.random.randint(m)
        else:
            b1 = _argmax(Q[0, s])
        if np.random.random() < eps:
            b2 = np.random.randint(m)
        else:
            b2 = _argmax(Q[1, s])
        s_next = b1 * m + b2
        changed = False
        # firm 1 update
        old = _argmax(Q[0, s])
        Q[0, s, b1] = (1 - alpha) * Q[0, s, b1] + alpha * (PI[b1, b2, 0] + delta * _max(Q[0, s_next]))
        if _argmax(Q[0, s]) != old:
            changed = True
        # firm 2 update
        old = _argmax(Q[1, s])
        Q[1, s, b2] = (1 - alpha) * Q[1, s, b2] + alpha * (PI[b1, b2, 1] + delta * _max(Q[1, s_next]))
        if _argmax(Q[1, s]) != old:
            changed = True
        stable = 0 if changed else stable + 1
        s = s_next
        t += 1
        if stable >= conv_window:
            converged = True
            break
    return Q, converged, t, s


@njit(parallel=True, cache=True)
def run_sessions(seeds, PI, Q0, alpha, beta, delta, conv_window, max_periods):
    n = seeds.shape[0]
    Qs = np.empty((n, Q0.shape[0], Q0.shape[1], Q0.shape[2]))
    conv = np.zeros(n, dtype=np.bool_)
    periods = np.zeros(n, dtype=np.int64)
    final_state = np.zeros(n, dtype=np.int64)
    for k in prange(n):
        Q, c, t, s = run_session(seeds[k], PI, Q0, alpha, beta, delta, conv_window, max_periods)
        Qs[k] = Q
        conv[k] = c
        periods[k] = t
        final_state[k] = s
    return Qs, conv, periods, final_state


# --------------------------------------------------------------------------
# Outcomes after convergence
# --------------------------------------------------------------------------
def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """P[firm, state] = argmax_a Q[firm, state, a]."""
    return Q.argmax(axis=2)


def limit_cycle(P: np.ndarray, s0: int, m: int) -> list[int]:
    """Greedy play is deterministic, so it ends in a cycle of states. Return it."""
    seen = {}
    path = []
    s = s0
    while s not in seen:
        seen[s] = len(path)
        path.append(s)
        s = int(P[0, s]) * m + int(P[1, s])
    return path[seen[s]:]


def session_outcome(Q, s_final, PI, bm) -> dict:
    m = PI.shape[0]
    P = greedy_policy(Q)
    cyc = limit_cycle(P, int(s_final), m)
    pis = np.array([[PI[s // m, s % m, 0], PI[s // m, s % m, 1]] for s in cyc])
    pi_bar = pis.mean(axis=0)
    gain = (pi_bar - bm["pi_nash"]) / (bm["pi_monopoly"] - bm["pi_nash"])
    return {"cycle_len": len(cyc), "cycle": cyc, "pi": pi_bar,
            "delta_firm": gain, "delta": float(gain.mean())}


def impulse_response(Q, cycle_state, prices, PI, horizon=25):
    """Calvano's deviation experiment: in the first period, override firm 1 and
    make it play the static best response to firm 2's current price; afterwards
    both firms follow their learned strategies. Returns price paths of shape
    (horizon + 1, 2); row 0 is the pre-deviation period."""
    m = PI.shape[0]
    P = greedy_policy(Q)
    s = int(cycle_state)
    path = np.empty((horizon + 1, 2))
    path[0] = prices[s // m], prices[s % m]
    a2 = int(P[1, s])
    a1 = int(np.argmax(PI[:, a2, 0]))          # static best response of firm 1
    path[1] = prices[a1], prices[a2]
    s = a1 * m + a2
    for tau in range(2, horizon + 1):
        a1, a2 = int(P[0, s]), int(P[1, s])
        path[tau] = prices[a1], prices[a2]
        s = a1 * m + a2
    return path
