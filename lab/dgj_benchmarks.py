"""
Theoretical benchmarks for the Dou, Goldstein & Ji (2025) trading model,
generalised to heterogeneous signal precision.

Setup (one period; all shocks i.i.d. across periods)
  v       fundamental value, discrete: uniform over the grid v_1..v_nv
  s_i     speculator i's observed signal: nearest grid point to v + eps_i,
          eps_i ~ N(0, sigma_eps_i^2)   (sigma_eps_i = 0 gives the DGJ baseline)
  x_i     order = chi_i * d_i,  d_i = s_i - v_bar      (linear strategy)
  u       noise trade ~ N(0, sigma_u^2)
  y       = sum_i x_i + u                              (total order flow)
  z       = -xi (p - v_bar)                            (information-insensitive investors)
  p       market maker minimises E[(y+z)^2 + theta (p - v)^2 | y]
          => p = v_bar + lam * y,  lam = (xi + theta*gamma1) / (xi^2 + theta),
             gamma1 = Cov(v, y) / Var(y)               (DGJ eq. 3.4 with E[v|y] linear)

With w = v - v_bar, C_i = E[d_i w], S_ij = E[d_i d_j]:
  E[profit_i] = chi_i C_i - lam * chi_i * sum_j chi_j S_ij

Non-collusive Nash (each i takes lam and the others' chi as given):
  dE/dchi_i = 0  =>  lam * (S + diag(S)) chi = C
Perfect cartel (maximise joint profit, no pooling of signals):
  d/dchi sum_i E[profit_i] = 0  =>  2 lam S chi = C
In both cases lam is then solved as a fixed point together with gamma1(chi).

Check: perfect information, I symmetric agents => S_ij = C_i = sigma_v^2 and
  chi_N = 1 / ((I+1) lam),  chi_M = 1 / (2 I lam)   — DGJ Section 4.2.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def value_grid(n_v: int, v_bar: float = 1.0, sigma_v: float = 1.0) -> np.ndarray:
    """DGJ discretisation: v_k = v_bar + sigma_v * Phi^{-1}((2k-1)/(2 n_v)), equal probabilities."""
    k = np.arange(1, n_v + 1)
    return v_bar + sigma_v * norm.ppf((2 * k - 1) / (2 * n_v))


def bin_edges(v_grid: np.ndarray) -> np.ndarray:
    """Midpoints between grid points; a continuous signal is mapped to the nearest grid point."""
    return 0.5 * (v_grid[1:] + v_grid[:-1])


def signal_matrix(v_grid: np.ndarray, sigma_eps: float) -> np.ndarray:
    """T[j, k] = P(observed signal = grid point k | true value = grid point j)."""
    n = len(v_grid)
    if sigma_eps <= 0:
        return np.eye(n)
    edges = np.concatenate([[-np.inf], bin_edges(v_grid), [np.inf]])
    T = np.empty((n, n))
    for j in range(n):
        cdf = norm.cdf((edges - v_grid[j]) / sigma_eps)
        T[j] = np.diff(cdf)
    return T


def moments(v_grid: np.ndarray, sigma_eps: np.ndarray, v_bar: float = 1.0):
    """Exact second moments of the discretised information structure.

    Returns C (I,), S (I, I), and var_v (variance of the discrete v)."""
    I = len(sigma_eps)
    n = len(v_grid)
    w = v_grid - v_bar
    d = v_grid - v_bar
    pv = np.full(n, 1.0 / n)
    T = [signal_matrix(v_grid, s) for s in sigma_eps]
    Ed = np.array([T[i] @ d for i in range(I)])          # E[d_i | v_j], shape (I, n)
    Ed2 = np.array([T[i] @ d ** 2 for i in range(I)])    # E[d_i^2 | v_j]
    C = np.array([np.sum(pv * w * Ed[i]) for i in range(I)])
    S = np.empty((I, I))
    for i in range(I):
        for j in range(I):
            # signals are independent conditional on v
            S[i, j] = np.sum(pv * Ed2[i]) if i == j else np.sum(pv * Ed[i] * Ed[j])
    return C, S, float(np.sum(pv * w ** 2))


def _chi_given_lam(lam: float, C: np.ndarray, S: np.ndarray, kind: str) -> np.ndarray:
    if kind == "nash":
        return np.linalg.solve(lam * (S + np.diag(np.diag(S))), C)
    if kind == "cartel":
        # S is singular under perfect information (all signals identical); the
        # minimum-norm solution then splits the cartel order equally.
        return np.linalg.lstsq(2.0 * lam * S, C, rcond=None)[0]
    raise ValueError(kind)


def _lam_map(lam, C, S, kind, xi, theta, sigma_u):
    chi = _chi_given_lam(lam, C, S, kind)
    var_y = chi @ S @ chi + sigma_u ** 2
    gamma1 = (chi @ C) / var_y
    return (xi + theta * gamma1) / (xi ** 2 + theta)


def solve(kind: str, C, S, xi: float, theta: float, sigma_u: float) -> dict:
    """Linear-strategy benchmark ('nash' or 'cartel') with lam as a fixed point."""
    f = lambda lam: lam - _lam_map(lam, C, S, kind, xi, theta, sigma_u)
    lo, hi = 1e-12, 1e6
    lam = brentq(f, lo, hi, xtol=1e-16, rtol=1e-14, maxiter=500)
    chi = _chi_given_lam(lam, C, S, kind)
    profit = chi * C - lam * chi * (S @ chi)
    var_y = chi @ S @ chi + sigma_u ** 2
    return {"kind": kind, "lam": lam, "chi": chi, "profit": profit,
            "gamma1": (chi @ C) / var_y}


def benchmarks(v_grid, sigma_eps, xi, theta, sigma_u, v_bar=1.0) -> dict:
    sigma_eps = np.asarray(sigma_eps, dtype=float)
    C, S, var_v = moments(v_grid, sigma_eps, v_bar)
    nash = solve("nash", C, S, xi, theta, sigma_u)
    cartel = solve("cartel", C, S, xi, theta, sigma_u)
    return {"C": C, "S": S, "var_v": var_v, "nash": nash, "cartel": cartel}


def profit_given_chi(chi, C, S, lam):
    """Expected per-agent profit for arbitrary linear strategies and price impact lam."""
    chi = np.asarray(chi, dtype=float)
    return chi * C - lam * chi * (S @ chi)


def best_response(i: int, chi, C, S, lam) -> float:
    """Static best response of speculator i to the others' linear strategies,
    taking lam as given (the same convention as the Nash benchmark):
    chi_i = (C_i - lam * sum_{j != i} S_ij chi_j) / (2 lam S_ii)."""
    chi = np.asarray(chi, dtype=float)
    others = S[i] @ chi - S[i, i] * chi[i]
    return (C[i] - lam * others) / (2.0 * lam * S[i, i])


def restraint(chi, C, S, lam) -> dict:
    """Per-agent collusion diagnostic that stays meaningful under heterogeneity.

    restraint_i = chi_i / BR_i(chi_-i): 1 means speculator i trades exactly at its
    best response to what the other actually does (no restraint); below 1 means
    it holds back. deviation_gain_i = profit gain (fraction) i would get by
    switching to that best response. Both are exactly 1 and 0 at the Nash benchmark."""
    chi = np.asarray(chi, dtype=float)
    base = profit_given_chi(chi, C, S, lam)
    r = np.empty(len(chi))
    g = np.empty(len(chi))
    for i in range(len(chi)):
        br = best_response(i, chi, C, S, lam)
        r[i] = chi[i] / br
        dev = chi.copy()
        dev[i] = br
        g[i] = profit_given_chi(dev, C, S, lam)[i] / base[i] - 1.0
    return {"restraint": r, "deviation_gain": g}
