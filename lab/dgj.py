"""
Laptop-scale implementation of the trading laboratory in
Dou, Goldstein & Ji (2025), "AI-Powered Trading, Algorithmic Collusion, and
Price Efficiency", NBER WP 34054, Sections 3-4 — plus the extension of this
project: speculators observe noisy signals of different precision.

Each period t:
  1. every speculator i sees its signal s_i,t and picks an aggressiveness level
     chi from its grid (epsilon-greedy on its Q-table); its order is
     x_i = chi * (s_i,t - v_bar)
  2. noise trade u_t ~ N(0, sigma_u^2) arrives; total flow y_t = sum x_i + u_t
  3. the adaptive market maker prices p_t from y_t using rolling OLS (DGJ eq. 4.1-4.2)
  4. information-insensitive investors absorb z_t = -xi (p_t - v_bar)
  5. profits pi_i = (v_t - p_t) x_i; the asset expires and v_t becomes public
  6. state moves to s_i,t+1 = (p_t, m_i,t, s_i,t+1) and each Q-table updates one cell,
     where the monitoring variable m_i,t is
        Variant A ("realized"):   the realised value v_t
        Variant B ("own_signal"): speculator i's own past signal s_i,t
     With perfect signals both variants are identical to DGJ's state {p_t-1, v_t-1, v_t}.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from numba import njit, prange

from . import dgj_benchmarks as bmk

MONITOR = {"realized": 0, "own_signal": 1}


@dataclass
class DGJParams:
    # environment (DGJ baseline: I = 2, xi = 500, sigma_u in {0.1, 100})
    I: int = 2
    xi: float = 500.0
    theta: float = 0.1
    sigma_u: float = 0.1
    rho: float = 0.95
    v_bar: float = 1.0
    sigma_v: float = 1.0
    # extension: signal noise per speculator (0 = DGJ baseline) and monitoring variant
    sigma_eps: tuple = field(default_factory=lambda: (0.0, 0.0))
    monitor: str = "realized"
    # controls: fixed[i] = None (speculator i learns) or a number m (it plays the fixed
    # linear strategy m * chi_i^Nash and never learns)
    fixed: tuple = field(default_factory=lambda: (None, None))
    # action grid: "benchmarks" (DGJ: between each agent's Nash and cartel chi, extended
    # by iota) or "nash" ((1 - grid_width) .. (1 + grid_width) x own Nash chi, same for all)
    grid: str = "benchmarks"
    grid_width: float = 0.3
    # discretisation (DGJ: n_v = 10, n_x = 15, n_p = 31, iota = 0.1)
    n_v: int = 6
    n_x: int = 13
    n_p: int = 37
    iota: float = 0.1
    # market maker rolling window (DGJ: 10,000)
    Tm: int = 10_000
    # learning (DGJ: alpha = 0.01, beta = 5e-7, convergence window 1e6)
    alpha: float = 0.05
    beta: float = 2e-6
    conv_window: int = 100_000
    # stability only counts once every learner explores less than this in every signal
    # state (compensates for the 10x shorter window than DGJ's; 0 switches it off)
    conv_max_explore: float = 0.01
    max_periods: int = 200_000_000
    eval_periods: int = 100_000

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sigma_eps"] = list(self.sigma_eps)
        d["fixed"] = list(self.fixed)
        return d


def setup(prm: DGJParams) -> dict:
    """Grids, benchmarks and the initial Q-table for a parameter set."""
    if len(prm.sigma_eps) != prm.I:
        raise ValueError("sigma_eps needs one entry per speculator")
    v_grid = bmk.value_grid(prm.n_v, prm.v_bar, prm.sigma_v)
    edges = bmk.bin_edges(v_grid)
    bm = bmk.benchmarks(v_grid, prm.sigma_eps, prm.xi, prm.theta, prm.sigma_u, prm.v_bar)
    chiN, chiM = bm["nash"]["chi"], bm["cartel"]["chi"]
    lamN = bm["nash"]["lam"]

    # action grid per speculator: n_x points on [lo - iota*gap, hi + iota*gap],
    # lo/hi = the two benchmarks. With iota = 0.1 and n_x - 1 a multiple of 12,
    # both benchmarks are exact grid points (index 1 and n_x - 2).
    if len(prm.fixed) != prm.I:
        raise ValueError("fixed needs one entry per speculator")
    chi_grid = np.empty((prm.I, prm.n_x))
    for i in range(prm.I):
        if prm.grid == "benchmarks":
            lo, hi = min(chiN[i], chiM[i]), max(chiN[i], chiM[i])
            gap = hi - lo
            chi_grid[i] = np.linspace(lo - prm.iota * gap, hi + prm.iota * gap, prm.n_x)
        elif prm.grid == "nash":
            chi_grid[i] = np.linspace((1 - prm.grid_width) * chiN[i],
                                      (1 + prm.grid_width) * chiN[i], prm.n_x)
        else:
            raise ValueError("grid must be 'benchmarks' or 'nash'")
    fixed_mask = np.array([f is not None for f in prm.fixed], dtype=np.bool_)
    fixed_chi = np.array([(f if f is not None else 0.0) * chiN[i]
                          for i, f in enumerate(prm.fixed)], dtype=float)

    # price grid (DGJ Section 4.2): benchmark order flows +/- 1.96 sigma_u, extended by iota
    d = v_grid - prm.v_bar
    # the price grid deliberately does not depend on `grid` or `fixed`, so the controls
    # change exactly one thing relative to the main runs
    chi_max = np.maximum(chiN, chiM)
    pH = prm.v_bar + lamN * (np.sum(chi_max) * d.max() + 1.96 * prm.sigma_u)
    pL = prm.v_bar + lamN * (np.sum(chi_max) * d.min() - 1.96 * prm.sigma_u)
    span = pH - pL
    p_grid = np.linspace(pL - prm.iota * span, pH + prm.iota * span, prm.n_p)

    # initial Q (DGJ Section 4.2): discounted profit if the others pick uniformly
    # from the grid and noise trading is zero; identical across p and m states
    n_states = prm.n_p * prm.n_v * prm.n_v
    Q0 = np.empty((prm.I, n_states, prm.n_x))
    for i in range(prm.I):
        val = np.empty((prm.n_v, prm.n_x))
        for k in range(prm.n_v):
            for a in range(prm.n_x):
                x = chi_grid[i, a] * d[k]
                others = (prm.I - 1) * chi_grid[i] * d[k]
                val[k, a] = np.mean((d[k] - lamN * (x + others)) * x) / (1 - prm.rho)
        for s in range(n_states):
            Q0[i, s] = val[s % prm.n_v]

    return {"v_grid": v_grid, "edges": edges, "bm": bm, "chi_grid": chi_grid,
            "p_grid": p_grid, "Q0": Q0, "lam0": lamN,
            "sig_eps": np.asarray(prm.sigma_eps, dtype=float),
            "fixed_mask": fixed_mask, "fixed_chi": fixed_chi}


# --------------------------------------------------------------------------
# Compiled core
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
def _draw_signal(v_idx, v_grid, edges, sd):
    """Index of the grid point nearest to v + N(0, sd^2)."""
    if sd <= 0.0:
        return v_idx
    val = v_grid[v_idx] + sd * np.random.standard_normal()
    k = 0
    while k < edges.shape[0] and val > edges[k]:
        k += 1
    return k


@njit(cache=True)
def _price_bin(p, p0, dp, n_p):
    b = int(np.floor((p - p0) / dp + 0.5))
    if b < 0:
        return 0
    if b > n_p - 1:
        return n_p - 1
    return b


@njit(cache=True)
def run_session(seed, v_grid, edges, sig_eps, chi_grid, p0, dp, n_p, Q0,
                v_bar, xi, theta, sigma_u, rho, alpha, beta, Tm, lam0, monitor,
                conv_window, max_periods, eval_periods, fixed_mask, fixed_chi, n_req):
    """One session: learn until the greedy strategies of all speculators are
    unchanged for `conv_window` periods (or `max_periods` is hit), then play
    greedily for `eval_periods` with learning switched off and record outcomes.
    Stability only counts once every learner has seen every signal value at least
    `n_req` times, i.e. once its exploration rate is below conv_max_explore."""
    np.random.seed(seed)
    I = Q0.shape[0]
    n_x = Q0.shape[2]
    n_v = v_grid.shape[0]
    Q = Q0.copy()
    counts = np.zeros((I, n_v), dtype=np.int64)
    sig = np.empty(I, dtype=np.int64)
    sig_next = np.empty(I, dtype=np.int64)
    s = np.empty(I, dtype=np.int64)
    act = np.empty(I, dtype=np.int64)
    x = np.empty(I)

    v_idx = np.random.randint(n_v)
    for i in range(I):
        sig[i] = _draw_signal(v_idx, v_grid, edges, sig_eps[i])
    pb0 = np.random.randint(n_p)
    for i in range(I):
        s[i] = (pb0 * n_v + np.random.randint(n_v)) * n_v + sig[i]

    # market maker: rolling window of centred data (vc = v - v_bar, pc = p - v_bar)
    bv = np.zeros(Tm)
    bp = np.zeros(Tm)
    by = np.zeros(Tm)
    bz = np.zeros(Tm)
    nb = 0
    head = 0
    Sv = 0.0; Sp = 0.0; Sy = 0.0; Sz = 0.0
    Syy = 0.0; Syv = 0.0; Spp = 0.0; Spz = 0.0
    p_int = 0.0          # centred intercept of the pricing rule
    p_slope = lam0       # price impact, starts at the Nash benchmark

    # evaluation accumulators
    prof = np.zeros(I)
    chis = np.zeros(I)
    sumX = 0.0; sumX2 = 0.0; sumu2 = 0.0
    psum_v = np.zeros(n_v)
    cnt_v = np.zeros(n_v)

    n_learn = 0
    for i in range(I):
        if not fixed_mask[i]:
            n_learn += 1
    n_done = 0
    explore_done = n_req <= 0 or n_learn == 0
    learning = True
    converged = False
    stable = 0
    t = 0
    t_learn = 0
    n_eval = 0
    while True:
        X = 0.0
        for i in range(I):
            if fixed_mask[i]:
                x[i] = fixed_chi[i] * (v_grid[sig[i]] - v_bar)
                X += x[i]
                continue
            if learning:
                eps = np.exp(-beta * counts[i, sig[i]])
                counts[i, sig[i]] += 1
                if counts[i, sig[i]] == n_req:
                    n_done += 1
                    if n_done == n_learn * n_v:
                        explore_done = True
                if np.random.random() < eps:
                    act[i] = np.random.randint(n_x)
                else:
                    act[i] = _argmax(Q[i, s[i]])
            else:
                act[i] = _argmax(Q[i, s[i]])
            x[i] = chi_grid[i, act[i]] * (v_grid[sig[i]] - v_bar)
            X += x[i]
        u = sigma_u * np.random.standard_normal()
        y = X + u
        pc = p_int + p_slope * y
        p = v_bar + pc
        v = v_grid[v_idx]
        zc = -xi * pc

        v_next = np.random.randint(n_v)
        for i in range(I):
            sig_next[i] = _draw_signal(v_next, v_grid, edges, sig_eps[i])
        pb = _price_bin(p, p0, dp, n_p)

        changed = False
        for i in range(I):
            pi = (v - p) * x[i]
            m_i = v_idx if monitor == 0 else sig[i]
            sn = (pb * n_v + m_i) * n_v + sig_next[i]
            if learning:
                if not fixed_mask[i]:
                    old = _argmax(Q[i, s[i]])
                    Q[i, s[i], act[i]] = (1.0 - alpha) * Q[i, s[i], act[i]] + alpha * (pi + rho * _max(Q[i, sn]))
                    if _argmax(Q[i, s[i]]) != old:
                        changed = True
            else:
                prof[i] += pi
                chis[i] += fixed_chi[i] if fixed_mask[i] else chi_grid[i, act[i]]
            s[i] = sn

        # ---- market maker learns from this period's data
        vc = v - v_bar
        if nb == Tm:
            Sv -= bv[head]; Sp -= bp[head]; Sy -= by[head]; Sz -= bz[head]
            Syy -= by[head] * by[head]; Syv -= by[head] * bv[head]
            Spp -= bp[head] * bp[head]; Spz -= bp[head] * bz[head]
        bv[head] = vc; bp[head] = pc; by[head] = y; bz[head] = zc
        Sv += vc; Sp += pc; Sy += y; Sz += zc
        Syy += y * y; Syv += y * vc; Spp += pc * pc; Spz += pc * zc
        head += 1
        if head == Tm:
            head = 0
        if nb < Tm:
            nb += 1
        if nb == Tm:
            if head == 0:
                # recompute the sums exactly once per window to stop rounding drift
                Sv = 0.0; Sp = 0.0; Sy = 0.0; Sz = 0.0
                Syy = 0.0; Syv = 0.0; Spp = 0.0; Spz = 0.0
                for k in range(Tm):
                    Sv += bv[k]; Sp += bp[k]; Sy += by[k]; Sz += bz[k]
                    Syy += by[k] * by[k]; Syv += by[k] * bv[k]
                    Spp += bp[k] * bp[k]; Spz += bp[k] * bz[k]
            n = float(Tm)
            g1 = (n * Syv - Sy * Sv) / (n * Syy - Sy * Sy)      # E[v|y] slope
            g0 = (Sv - g1 * Sy) / n
            b = (n * Spz - Sp * Sz) / (n * Spp - Sp * Sp)       # z on p slope = -xi_1
            xi1 = -b
            xi0 = (Sz - b * Sp) / n
            p_slope = (xi1 + theta * g1) / (xi1 * xi1 + theta)
            p_int = (xi1 * xi0 + theta * g0) / (xi1 * xi1 + theta)

        for i in range(I):
            sig[i] = sig_next[i]
        if not learning:
            sumX += X; sumX2 += X * X; sumu2 += u * u
            psum_v[v_idx] += p
            cnt_v[v_idx] += 1.0
        v_idx = v_next
        t += 1

        if learning:
            stable = 0 if (changed or not explore_done) else stable + 1
            if stable >= conv_window:
                converged = True
                learning = False
                t_learn = t
            elif t >= max_periods:
                learning = False
                t_learn = t
        else:
            n_eval += 1
            if n_eval >= eval_periods:
                break

    mX = sumX / n_eval
    info = (sumX2 / n_eval - mX * mX) / (sumu2 / n_eval)
    mis = 0.0
    for k in range(n_v):
        if cnt_v[k] > 0:
            mis += abs(psum_v[k] / cnt_v[k] - v_grid[k]) / n_v
    return (Q, converged, t_learn, prof / n_eval, chis / n_eval, info, mis,
            p_int, p_slope, s, v_idx, sig)


@njit(parallel=True, cache=True)
def run_sessions(seeds, v_grid, edges, sig_eps, chi_grid, p0, dp, n_p, Q0,
                 v_bar, xi, theta, sigma_u, rho, alpha, beta, Tm, lam0, monitor,
                 conv_window, max_periods, eval_periods, fixed_mask, fixed_chi, n_req):
    n = seeds.shape[0]
    I = Q0.shape[0]
    Qs = np.empty((n, Q0.shape[0], Q0.shape[1], Q0.shape[2]))
    conv = np.zeros(n, dtype=np.bool_)
    t_learn = np.zeros(n, dtype=np.int64)
    prof = np.zeros((n, I))
    chis = np.zeros((n, I))
    info = np.zeros(n)
    mis = np.zeros(n)
    mm = np.zeros((n, 2))
    s_fin = np.zeros((n, I), dtype=np.int64)
    v_fin = np.zeros(n, dtype=np.int64)
    sig_fin = np.zeros((n, I), dtype=np.int64)
    for k in prange(n):
        r = run_session(seeds[k], v_grid, edges, sig_eps, chi_grid, p0, dp, n_p, Q0,
                        v_bar, xi, theta, sigma_u, rho, alpha, beta, Tm, lam0, monitor,
                        conv_window, max_periods, eval_periods, fixed_mask, fixed_chi, n_req)
        Qs[k] = r[0]
        conv[k] = r[1]
        t_learn[k] = r[2]
        prof[k] = r[3]
        chis[k] = r[4]
        info[k] = r[5]
        mis[k] = r[6]
        mm[k, 0] = r[7]
        mm[k, 1] = r[8]
        s_fin[k] = r[9]
        v_fin[k] = r[10]
        sig_fin[k] = r[11]
    return Qs, conv, t_learn, prof, chis, info, mis, mm, s_fin, v_fin, sig_fin


# --------------------------------------------------------------------------
# Impulse responses (DGJ Section 5.3), with common random numbers
# --------------------------------------------------------------------------
@njit(cache=True)
def _greedy_path(Q, v_grid, chi_grid, p0, dp, n_p, v_bar, p_int, p_slope, monitor,
                 s_start, vseq, sigseq, useq, shock, shock_at, chi_out, ptil_out):
    I = Q.shape[0]
    n_v = v_grid.shape[0]
    s = s_start.copy()
    H = useq.shape[0]
    for h in range(H):
        v_idx = vseq[h]
        X = 0.0
        for i in range(I):
            a = _argmax(Q[i, s[i]])
            chi_out[h, i] = chi_grid[i, a]
            X += chi_grid[i, a] * (v_grid[sigseq[h, i]] - v_bar)
        u = useq[h]
        sgn = 1.0 if v_grid[v_idx] >= v_bar else -1.0
        if h == shock_at:
            u += shock * sgn            # adverse: pushes the price towards v
        p = v_bar + p_int + p_slope * (X + u)
        ptil_out[h] = (p - v_bar) * sgn
        pb = _price_bin(p, p0, dp, n_p)
        for i in range(I):
            m_i = v_idx if monitor == 0 else sigseq[h, i]
            s[i] = (pb * n_v + m_i) * n_v + sigseq[h + 1, i]
    return s


@njit(cache=True)
def impulse_response(seed, Q, v_grid, edges, sig_eps, chi_grid, p0, dp, n_p, v_bar,
                     sigma_u, p_int, p_slope, monitor, s_start, v_start, sig_start,
                     shocks, n_paths, burn, horizon, shock_at):
    """Average effect of a one-off adverse noise-trade shock at period `shock_at`
    on aggressiveness and prices, measured against the same path without the
    shock (identical draws of v, signals and u). Learning is off; the market
    maker's pricing rule is frozen at its final estimate.

    The chain state is (s, v_idx, sig): Q-states, current true value, current signals."""
    np.random.seed(seed)
    I = Q.shape[0]
    n_v = v_grid.shape[0]
    K = shocks.shape[0]
    dchi = np.zeros((K, horizon, I))
    dptil = np.zeros((K, horizon))
    base_chi = np.zeros((horizon, I))
    base_ptil = np.zeros(horizon)
    vseq = np.empty(horizon + 1, dtype=np.int64)
    sigseq = np.empty((horizon + 1, I), dtype=np.int64)
    useq = np.empty(horizon)
    chi_b = np.empty((horizon, I)); pt_b = np.empty(horizon)
    chi_s = np.empty((horizon, I)); pt_s = np.empty(horizon)
    s = s_start.copy()
    v_idx = v_start
    sig = sig_start.copy()
    for path in range(n_paths):
        # burn-in: let the chain move on so paths are not all identical
        for b in range(burn):
            vseq[0] = v_idx
            vseq[1] = np.random.randint(n_v)
            for i in range(I):
                sigseq[0, i] = sig[i]
                sigseq[1, i] = _draw_signal(vseq[1], v_grid, edges, sig_eps[i])
            useq[0] = sigma_u * np.random.standard_normal()
            s = _greedy_path(Q, v_grid, chi_grid, p0, dp, n_p, v_bar, p_int, p_slope, monitor,
                             s, vseq[:2], sigseq[:2], useq[:1], 0.0, -1, chi_b, pt_b)
            v_idx = vseq[1]
            for i in range(I):
                sig[i] = sigseq[1, i]
        # common random numbers for this path
        vseq[0] = v_idx
        for i in range(I):
            sigseq[0, i] = sig[i]
        for h in range(1, horizon + 1):
            vseq[h] = np.random.randint(n_v)
            for i in range(I):
                sigseq[h, i] = _draw_signal(vseq[h], v_grid, edges, sig_eps[i])
        for h in range(horizon):
            useq[h] = sigma_u * np.random.standard_normal()
        s_end = _greedy_path(Q, v_grid, chi_grid, p0, dp, n_p, v_bar, p_int, p_slope, monitor,
                             s, vseq, sigseq, useq, 0.0, -1, chi_b, pt_b)
        for h in range(horizon):
            base_ptil[h] += pt_b[h]
            for i in range(I):
                base_chi[h, i] += chi_b[h, i]
        for k in range(K):
            _greedy_path(Q, v_grid, chi_grid, p0, dp, n_p, v_bar, p_int, p_slope, monitor,
                         s, vseq, sigseq, useq, shocks[k], shock_at, chi_s, pt_s)
            for h in range(horizon):
                dptil[k, h] += pt_s[h] - pt_b[h]
                for i in range(I):
                    dchi[k, h, i] += chi_s[h, i] - chi_b[h, i]
        # continue the chain from the end of the unshocked path
        s = s_end
        v_idx = vseq[horizon]
        for i in range(I):
            sig[i] = sigseq[horizon, i]
    return dchi / n_paths, dptil / n_paths, base_chi / n_paths, base_ptil / n_paths


@njit(cache=True)
def run_fixed_strategies(seed, chi, v_grid, edges, sig_eps, v_bar, xi, theta, sigma_u,
                         Tm, lam0, periods):
    """Validation tool: the same market (noise traders, investors, adaptive market
    maker) but with fixed linear strategies x_i = chi_i * (s_i - v_bar) instead of
    learners. Average profits must match the analytical benchmark for those chi.
    Returns (mean profit per agent, final price impact estimate)."""
    np.random.seed(seed)
    I = chi.shape[0]
    n_v = v_grid.shape[0]
    bv = np.zeros(Tm); bp = np.zeros(Tm); by = np.zeros(Tm); bz = np.zeros(Tm)
    nb = 0; head = 0
    Sv = 0.0; Sp = 0.0; Sy = 0.0; Sz = 0.0; Syy = 0.0; Syv = 0.0; Spp = 0.0; Spz = 0.0
    p_int = 0.0; p_slope = lam0
    prof = np.zeros(I)
    n_rec = 0
    x = np.empty(I)
    for t in range(Tm + periods):
        v_idx = np.random.randint(n_v)
        X = 0.0
        for i in range(I):
            k = _draw_signal(v_idx, v_grid, edges, sig_eps[i])
            x[i] = chi[i] * (v_grid[k] - v_bar)
            X += x[i]
        y = X + sigma_u * np.random.standard_normal()
        pc = p_int + p_slope * y
        p = v_bar + pc
        v = v_grid[v_idx]
        if t >= Tm:                      # record only after the window has filled
            for i in range(I):
                prof[i] += (v - p) * x[i]
            n_rec += 1
        vc = v - v_bar
        zc = -xi * pc
        if nb == Tm:
            Sv -= bv[head]; Sp -= bp[head]; Sy -= by[head]; Sz -= bz[head]
            Syy -= by[head] * by[head]; Syv -= by[head] * bv[head]
            Spp -= bp[head] * bp[head]; Spz -= bp[head] * bz[head]
        bv[head] = vc; bp[head] = pc; by[head] = y; bz[head] = zc
        Sv += vc; Sp += pc; Sy += y; Sz += zc
        Syy += y * y; Syv += y * vc; Spp += pc * pc; Spz += pc * zc
        head = (head + 1) % Tm
        if nb < Tm:
            nb += 1
        if nb == Tm:
            n = float(Tm)
            g1 = (n * Syv - Sy * Sv) / (n * Syy - Sy * Sy)
            g0 = (Sv - g1 * Sy) / n
            b = (n * Spz - Sp * Sz) / (n * Spp - Sp * Sp)
            xi1 = -b
            xi0 = (Sz - b * Sp) / n
            p_slope = (xi1 + theta * g1) / (xi1 * xi1 + theta)
            p_int = (xi1 * xi0 + theta * g0) / (xi1 * xi1 + theta)
    return prof / n_rec, p_slope


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------
def simulate(prm: DGJParams, seeds, env: dict | None = None) -> dict:
    env = env or setup(prm)
    p_grid = env["p_grid"]
    out = run_sessions(np.asarray(seeds, dtype=np.int64), env["v_grid"], env["edges"],
                       env["sig_eps"], env["chi_grid"], p_grid[0], p_grid[1] - p_grid[0],
                       len(p_grid), env["Q0"], prm.v_bar, prm.xi, prm.theta, prm.sigma_u,
                       prm.rho, prm.alpha, prm.beta, prm.Tm, env["lam0"],
                       MONITOR[prm.monitor], prm.conv_window, prm.max_periods,
                       prm.eval_periods, env["fixed_mask"], env["fixed_chi"], _n_req(prm))
    keys = ["Q", "converged", "t_learn", "profit", "chi", "informativeness",
            "mispricing", "mm", "s_final", "v_final", "sig_final"]
    res = dict(zip(keys, out))
    res.update(profit_gain(res["profit"], env["bm"]))
    res["env"] = env
    return res


def _n_req(prm: DGJParams) -> int:
    """Visits per signal value after which exploration exp(-beta n) < conv_max_explore."""
    if prm.conv_max_explore <= 0:
        return 0
    return int(np.ceil(np.log(1.0 / prm.conv_max_explore) / prm.beta))


def profit_gain(profit: np.ndarray, bm: dict) -> dict:
    """Delta = (pi - pi_N) / (pi_M - pi_N).

    'delta' uses joint profits (the cartel benchmark is defined on joint profit);
    'delta_agent' is per speculator and only meaningful when agents are symmetric."""
    piN, piM = bm["nash"]["profit"], bm["cartel"]["profit"]
    delta = (profit.sum(axis=1) - piN.sum()) / (piM.sum() - piN.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_agent = (profit - piN) / (piM - piN)
    return {"delta": delta, "delta_agent": delta_agent}


def session_irf(res: dict, k: int, prm: DGJParams, shock_fracs=(0.0025, 0.025, 0.115, 0.15),
                n_paths=2000, burn=20, horizon=9, shock_at=2, seed=None) -> dict:
    """Impulse response for session k. Shock sizes are fractions of the
    speculators' average absolute order, as in DGJ Figure 3."""
    if any(f is not None for f in prm.fixed):
        raise NotImplementedError("impulse responses are only defined when all speculators learn")
    env = res["env"]
    p_grid = env["p_grid"]
    d_abs = np.mean(np.abs(env["v_grid"] - prm.v_bar))
    mean_abs_x = float(np.mean(res["chi"][k]) * d_abs)
    shocks = np.array(shock_fracs) * mean_abs_x
    p_int, p_slope = res["mm"][k]
    dchi, dptil, bchi, bptil = impulse_response(
        int(seed if seed is not None else 10_000 + k), res["Q"][k], env["v_grid"], env["edges"],
        env["sig_eps"], env["chi_grid"], p_grid[0], p_grid[1] - p_grid[0], len(p_grid),
        prm.v_bar, prm.sigma_u, p_int, p_slope, MONITOR[prm.monitor],
        res["s_final"][k].copy(), int(res["v_final"][k]), res["sig_final"][k].copy(),
        shocks, n_paths, burn, horizon, shock_at)
    return {"dchi": dchi, "dptil": dptil, "base_chi": bchi, "base_ptil": bptil,
            "shocks": shocks, "shock_fracs": np.array(shock_fracs)}
