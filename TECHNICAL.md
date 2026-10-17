# Technical notes

Every equation used, where it is in the code, every deviation from the papers, and what
does not reproduce. Section numbers of the papers refer to Calvano et al. (2020, AER) and
Dou, Goldstein & Ji (2025, NBER WP 34054, July 2025 version).

---

## 0. Scope

The Calvano et al. (2020) baseline is replicated at its published scale (Δ = 0.849 against
the published 0.85, 1,000 sessions). The trading environment of Dou, Goldstein & Ji is
implemented from the paper's equations; with fixed benchmark strategies the simulated
market reproduces the analytical Nash and cartel profits to within about 0.1%. At laptop
scale the Q-learning speculators earn supra-competitive profits in every environment
tested, with a U-shape in noise-trading risk, and at low noise their learned strategies
punish large price deviations (a threshold response as in DGJ's Figure 3).

DGJ is **not** replicated quantitatively: Δ ≈ 0.38 instead of ≈ 0.75 at σᵤ = 0.1, the
punishment is shorter (one period) and graded, it depends on the price grid, and
high-noise sessions do not converge within the laptop budget (section 9).

Extension, preliminary and at laptop scale: with an equal action grid for both
speculators and a control against a fixed Nash opponent, the interaction part of the
restraint is 7–10% of the best response under symmetric information and 0–3% under
unequal signal precision; the worse-informed learner additionally under-trades by 6–13%
on its own (section 5.6).

---

## 1. Code map

| File | Content | Paper reference |
|---|---|---|
| `lab/calvano.py` | logit demand, Nash/monopoly prices, price grid, Q-learning loop, limit cycle, Δ, deviation experiment | Calvano et al. §I–III |
| `lab/dgj_benchmarks.py` | value grid, signal discretisation, exact moments, Nash and cartel benchmarks with λ fixed point | DGJ §3.2, §4.2; extension |
| `lab/dgj.py` `setup` | action grid, price grid, initial Q | DGJ §4.2 |
| `lab/dgj.py` `run_session` | the market, adaptive market maker, Q-learning, convergence, evaluation | DGJ §3.1, §4.1 |
| `lab/dgj.py` `impulse_response` | shock experiment with common random numbers | DGJ §5.3, Fig. 3 |
| `lab/dgj.py` `run_fixed_strategies` | market with fixed linear strategies (validation only) | — |
| `lab/dgj.py` `DGJParams.fixed`, `.grid` | controls: opponent fixed at a multiple of its Nash χ; Nash-centred action grid | extension |

Every compiled function (`@njit`) is plain Python syntax; Numba translates it to machine
code. `run_sessions` runs independent sessions on all cores with `prange`. Each session
seeds its own random-number stream at its start, so the result of session k depends
only on its seed, never on which core ran it (tested).

---

## 2. Rung 1 — Calvano et al. (2020)

### 2.1 Economy

Two firms, logit demand, marginal cost c = 1:

  qᵢ = exp((a − pᵢ)/μ) / ( Σⱼ exp((a − pⱼ)/μ) + exp(a₀/μ) ),  πᵢ = (pᵢ − c) qᵢ

with a = 2, a₀ = 0, μ = 0.25, δ = 0.95.

**Nash price.** ∂qᵢ/∂pᵢ = −qᵢ(1 − qᵢ)/μ, so the first-order condition
qᵢ + (pᵢ − c)∂qᵢ/∂pᵢ = 0 becomes 1 − (p − c)(1 − q)/μ = 0 at a symmetric point.
Solved with `brentq`: **pᴺ = 1.4729**. **Monopoly price** maximises joint profit:
**pᴹ = 1.9250**. Profits: πᴺ = 0.2229, πᴹ = 0.3375.

### 2.2 Price grid

15 equally spaced prices on [pᴺ − ξ(pᴹ − pᴺ), pᴹ + ξ(pᴹ − pᴺ)], ξ = 0.1. The
working-paper version (the PDF hosted by the FTC) describes the interval as [(1 − ξ)pᴺ, (1 + ξ)pᴹ]; both are implemented
(`--grid aer|wp`). The default is pinned down by the paper's own Figure 6, which deviates
"to the 9th price (1.74)": the default grid gives 1.7377, the alternative 1.78
(`tests/test_calvano.py::test_grid_matches_paper`).

### 2.3 Learning

- State: last period's price pair, s = a₁·15 + a₂ → 225 states × 15 actions per firm.
- Update (one cell per firm per period):
  Q(s, a) ← (1 − α) Q(s, a) + α [π + δ maxₐ′ Q(s′, a′)], α = 0.15.
- Exploration: each firm independently plays a random price with probability
  ε_t = exp(−βt), β = 4×10⁻⁶.
- Initial Q: discounted profit against a uniformly random rival, Q₀(s, a) = E_b[π(a, b)]/(1 − δ),
  the same for every state.
- Convergence: neither firm's greedy action (argmax) changes for 100,000 consecutive periods.
  Only the updated state's argmax can change, so this check is exact and cheap.

### 2.4 Outcome

After convergence greedy play is deterministic and ends in a cycle of states
(`limit_cycle`). Profit is averaged over the cycle and both firms:

  Δ = (π̄ − πᴺ)/(πᴹ − πᴺ)

| | Published | This code (1,000 sessions) |
|---|---|---|
| Δ (mean) | 0.85 | 0.849 (s.e. 0.004) |
| converged | all | 100% |
| periods to convergence | ~0.4–2.5 M across their grid | median 1.79 M |

### 2.5 Deviation experiment (their Fig. 5)

From a converged constant-price state, firm 1 is forced once to play the static best
response to firm 2's price; then both follow their learned strategies. Prices drop to near
Nash in the next period (firm 2 punishes), then climb back over about eight periods.
This finite punishment with gradual return is the signature Calvano et al. use to call the
outcome collusion rather than a failure to learn (`results/calvano/deviation_irf.png`).

---

## 3. Rung 2 — the DGJ trading laboratory

### 3.1 Economy and timing (DGJ §3.1)

Each period a new short-lived asset with value vₜ ~ N(v̄, σᵥ²), v̄ = σᵥ = 1, is traded
and expires at the end of the period. Participants:

- I = 2 informed speculators, who know vₜ and submit orders xᵢ,ₜ;
- a noise trader, uₜ ~ N(0, σᵤ²);
- information-insensitive investors with demand zₜ = −ξ(pₜ − v̄), ξ = 500 (DGJ eq. 3.2);
- a market maker who sees total flow yₜ = Σᵢ xᵢ,ₜ + uₜ and sets the price.

**Market maker's problem (eq. 3.3).** min_p E[(y + z)² + θ(p − v)² | y], θ = 0.1: an
inventory cost on the position it must absorb, −(y + z), plus a pricing-error cost.
Substituting z = −ξ(p − v̄) and differentiating:
−2ξ(y − ξ(p − v̄)) + 2θ(p − E[v|y]) = 0, so

  p = [ξ y + ξ² v̄ + θ E[v|y]] / (ξ² + θ)       (DGJ eq. 3.4)

With ξ = 500, ξ² = 250,000 ≫ θ: the price is almost the market-clearing price v̄ + y/ξ.
As ξ → 0 it becomes Kyle's efficient price E[v|y].

**Speculator's payoff:** πᵢ,ₜ = (vₜ − pₜ) xᵢ,ₜ.

### 3.2 Adaptive market maker (DGJ §4.1, eqs. 4.1–4.2)

The market maker does not know the model. It keeps the last T_m = 10,000 periods and runs
two OLS regressions every period:

  z = ξ₀ − ξ₁ p,  v = γ₀ + γ₁ y

and plugs the estimates into the first-order condition:

  p(y) = [ξ₁ξ₀ + θγ₀ + (ξ₁ + θγ₁) y] / (ξ₁² + θ),  price impact λ̂ = (ξ₁ + θγ₁)/(ξ₁² + θ)

DGJ print the intercept as γ̂₀; the full plug-in above has the same slope and an intercept
that differs from theirs by O(θ/ξ²) ≈ 4×10⁻⁷. Implementation details (`run_session`):

- regressions use running sums over a circular buffer, so each period costs O(1);
- data are centred (p − v̄, v − v̄) to avoid cancellation;
- the sums are recomputed exactly once per window to stop rounding drift;
- until the buffer is full the market maker prices with λ = λᴺ (the Nash benchmark).

**Why this matters for the extension:** the market maker never needs to know how
precise the speculators' signals are. It learns the order-flow → value mapping from data,
whatever the speculators do. Heterogeneous signals need no change to the market maker.

### 3.3 Speculators as Q-learners (DGJ §4.1–4.2)

- **State** sᵢ,ₜ = (pₜ₋₁, vₜ₋₁, vₜ) on grids: the current private signal, and last period's
  price and value. Comparing pₜ₋₁ with vₜ₋₁ tells a speculator how strongly the price
  reacted to fundamentals, i.e. whether the other side traded aggressively. That is the
  monitoring device behind price-trigger strategies. Index:
  s = (price_bin · n_v + v_{t−1} index) · n_v + vₜ index.
- **Value grid:** v_k = v̄ + σᵥ Φ⁻¹((2k − 1)/(2n_v)), equal probabilities. With n_v = 6
  the discrete variance is σ̂ᵥ² = 0.804; benchmarks use σ̂ᵥ (as DGJ do, footnote 24).
- **Actions** are aggressiveness levels χ, the order is x = χ(v − v̄). DGJ discretise the
  order interval [xᴹ − ι(xᴺ − xᴹ), xᴺ + ι(xᴺ − xᴹ)] for each v; because both benchmark orders
  are χ·(v − v̄), that is exactly a grid in χ. With ι = 0.1 and n_x = 13 both benchmarks are
  grid points (index 1 = cartel, index 11 = Nash), so the learner can play exactly Nash or
  exactly cartel.
- **Price grid:** from v̄ + λᴺ(I·x_min − 1.96σᵤ) to v̄ + λᴺ(I·x_max + 1.96σᵤ), extended by ι on both
  sides, n_p points. A realised price goes to the nearest grid point.
  **The price grid is the monitoring technology.** A deviation or shock is visible to the
  other speculator only if it moves the price across a bin boundary. At σᵤ = 0.1, a
  15%-of-|x| shock moves the price by 0.035; a bin is 0.092 wide at n_p = 25 and 0.062 at
  n_p = 37, so the shock is visible in roughly 38% vs 57% of cases (11.5% shock: 29% vs 43%).
  With n_p = 25 the learned strategies showed no punishment at all; with n_p = 37 they do
  (section 8). Because the research question is about the monitoring channel, the default
  is n_p = 37. It costs no extra time.
- **Exploration:** εₜ(v) = exp(−β t(v)), where t(v) counts how often the speculator has
  seen signal value v (DGJ eq. 4.3), β = 5×10⁻⁷.
- **Initial Q** (DGJ §4.2): Q₀(s, x) = (1/((1 − ρ) n_x)) Σ_{x₋ᵢ} [v − (v̄ + λᴺ(x + (I − 1)x₋ᵢ))] x,
  i.e. the discounted profit if the other speculator randomises over the grid and u = 0.
- **Update:** as in Calvano, with discount ρ = 0.95 and learning rate α = 0.01.
- **Convergence:** no speculator's greedy action changes for `conv_window` periods
  (DGJ: 10⁶; here 10⁵), or the period cap is reached (here 2×10⁸). Because the window is 10×
  shorter than DGJ's, stability only starts counting once every learner explores less than
  1% of the time in every signal state (`conv_max_explore = 0.01`, i.e. after about 55 M
  periods). Without this rule, lone learners in the control experiment were declared
  converged after 2 M periods while still exploring 80% of the time. No run in sections
  8.1–8.6 is affected: every converged session there stopped after 57 M periods.

### 3.4 Outcomes

After learning, 100,000 more periods are played greedily with learning switched off
(the market maker keeps updating). Recorded per session: mean profit per speculator,
mean aggressiveness, price informativeness I = var(Σx)/var(u) and mispricing
E = mean over v of |E[p | v] − v| (DGJ Definition 3.4), and

  Δ = (Σᵢ π̄ᵢ − Σᵢ πᵢᴺ) / (Σᵢ πᵢᴹ − Σᵢ πᵢᴺ)           (DGJ eq. 3.6)

### 3.5 Impulse responses (DGJ §5.3)

After convergence (learning off, market maker frozen at its final estimate), an adverse
noise shock hits at t = 3: u is increased by `shock × sign(v − v̄)`, which pushes the price
towards the value and lowers the speculators' profit. Shock sizes are 0.25%, 2.5%, 11.5%
and 15% of the average |x|, as in DGJ.

DGJ average 10,000 paths per session and compare with the long-run mean. Here each
shocked path is compared with **the same path without the shock** (identical draws of v,
signals and u: common random numbers). The difference isolates the causal effect of the
shock, so far fewer paths are needed for the same precision.

Price-trigger strategies predict a threshold: no response to a small shock, and a
similar punishment (more aggressive trading at t = 4) for all shocks above the threshold.

---

## 4. Benchmarks — derivations (`lab/dgj_benchmarks.py`)

Write w = v − v̄ and dᵢ = sᵢ − v̄, where sᵢ is speculator i's signal (sᵢ = v in the DGJ
baseline). Linear strategies xᵢ = χᵢ dᵢ. Define

  Cᵢ = E[dᵢ w],  S_ij = E[dᵢ dⱼ]

**Linear pricing.** If speculators use linear strategies, E[v | y] = v̄ + γ₁y with
γ₁ = Cov(v, y)/Var(y) = Σⱼ χⱼCⱼ / (χᵀSχ + σᵤ²). Inserting into eq. 3.4: p = v̄ + λy with

  λ = (ξ + θγ₁)/(ξ² + θ)

**Expected profit.**

  E[πᵢ] = E[χᵢ dᵢ (w − λ(Σⱼ χⱼ dⱼ + u))] = χᵢ Cᵢ − λ χᵢ Σⱼ χⱼ S_ij      (u independent, mean 0)

**Nash (each takes λ and the others' χ as given).** ∂E[πᵢ]/∂χᵢ = Cᵢ − λ(2χᵢS_ii + Σⱼ≠ᵢ χⱼS_ij) = 0,
a linear system λ(S + diag S)χ = C.
Perfect information (all S_ij = Cᵢ = σ̂ᵥ²), symmetric: λσ̂ᵥ²(2χ + (I − 1)χ) = σ̂ᵥ² ⇒

  **χᴺ = 1/((I + 1)λ)**

**Cartel (maximise joint profit, no pooling of private signals).**
∂/∂χ [χ·C − λχᵀSχ] = 0 ⇒ 2λSχ = C. Perfect information: S has rank 1 and only the total
matters, Iχ = 1/(2λ) ⇒

  **χᴹ = 1/(2Iλ)**

(the code takes the minimum-norm solution of the singular system, which is the equal split).

**λ as a fixed point.** χ depends on λ and λ depends on χ through γ₁. Solved with
`brentq` on f(λ) = λ − (ξ + θγ₁(χ(λ)))/(ξ² + θ), which is negative near 0 and positive for
large λ. With ξ = 500 the θγ₁ term is ~10⁻⁹, so λ = 1/ξ = 0.002 to six digits.

**Check against Kyle (1985).** With ξ = 0 and one speculator the model is Kyle's: the code
returns λ = σᵥ/(2σᵤ) and χ = σᵤ/σᵥ exactly (`test_kyle_limit`).

**Numbers at the baseline** (ξ = 500, n_v = 6): χᴺ = 166.67, χᴹ = 125.00, πᴺ = 44.66,
πᴹ = 50.25 per speculator. The cartel earns 12.5% more than Nash, so Δ = 0.75 means about
9% more profit, matching DGJ's statement that Δ ≈ 0.75 corresponds to profits "about 10%
higher" (§5.3). This is a useful sanity check on the benchmark code.

**Discrete moments.** Signals are mapped to the value grid, so all moments are computed
exactly from the discrete distribution (`moments`): P(signal = k | v = j) from normal CDFs
over the bin edges; signals are independent given v. Tested against Monte Carlo.

---

## 5. Rung 3 — heterogeneous signal precision

### 5.1 Signal model

Speculator i observes sᵢ,ₜ = the grid point nearest to vₜ + εᵢ,ₜ, εᵢ ~ N(0, qᵢ²). With
qᵢ = 0 this is the DGJ baseline. Using the same grid keeps the state space and Q-table
shape unchanged, so differences in outcomes cannot come from a different state space.
Orders are xᵢ = χ(sᵢ − v̄): the learner scales its own signal; the Bayesian shrinkage
E[v | sᵢ] is absorbed in the learned χ. Exploration counts visits per own-signal value.

### 5.2 Monitoring variants

- **Variant A (`monitor="realized"`):** the state holds the realised past value vₜ₋₁.
  Only the trading channel is affected by noisy signals.
- **Variant B (`monitor="own_signal"`):** the state holds the agent's own past signal sᵢ,ₜ₋₁.
  Trading and monitoring channels are both affected.

**Is vₜ₋₁ observable? (the open check in D1).** Yes, on two grounds. DGJ's asset "reach[es]
expiration at the end of the period with its fundamental value vₜ realized" (§3.1), and
their market maker stores realised values to run its regression (§4.1). Independently, any
speculator who traded can back out vₜ₋₁ from its own realised profit π = (v − p)x, because
it knows p and x. So Variant A is the faithful extension; Variant B is the counterfactual
that switches the monitoring channel on.

### 5.3 Benchmarks under heterogeneity

Nash and cartel are recomputed for every precision profile with the formulas of section 4,
using the exact discrete moments. Δ is measured on **joint** profits against the joint
benchmarks. Per-agent Δ is reported but only meaningful for symmetric profiles.

**The asymmetric cartel is a corner.** If speculator 1 sees v exactly (q₁ = 0) and
speculator 2 does not, then C₂ = S₁₂ (speculator 1's signal is v itself), so C = S·e₁ and the
cartel solution is χᴹ = (1/(2λ), 0). The joint-profit optimum lets the informed speculator
trade alone as a monopolist; the noisy speculator's signal only adds noise. At q₂ = 0.5:
χᴺ = (192.2, 130.2), χᴹ = (250.0, 0.0). Three consequences:

1. Under asymmetric precision, "collusion" does **not** mean "both trade less". It means
   the informed agent trades *more* than its Nash level and the uninformed one withdraws.
2. The uninformed agent earns nothing in the cartel, so the cartel is not individually
   rational without side payments, and Q-learners cannot make side payments. That is a
   concrete theoretical reason to expect fragility (hypothesis H1 of the project).
3. Δ must be joint, and the cartel benchmark must be stated as "joint-profit maximisation
   without pooling signals". The cartel that pooled signals would be a different benchmark.

There is a parallel in Athey & Bagwell (2001), who study collusion when firms have private
information about their *costs*: efficient collusion lets only the low-cost firm serve the
market, and the difficulty is giving the other firm a reason to stay out. Here the "low
cost" is the better signal. The parallel is loose (their private information is about
costs, not about a common value); read the paper before citing it.

### 5.4 Measuring collusion under heterogeneity: restraint relative to best response

Joint Δ says whether the pair earns more than Nash, not *who* holds back. Under unequal
precision that is the question that matters, so every DGJ run also reports, per speculator,

  restraintᵢ = χᵢ / BRᵢ(χ₋ᵢ),  BRᵢ = (Cᵢ − λ Σⱼ≠ᵢ S_ij χⱼ) / (2λ S_ii)

the observed aggressiveness divided by the static best response to what the other
speculator actually does, and the profit gain from switching to that best response
(`lab/dgj_benchmarks.py::restraint`, fields `restraint_vs_best_response` and
`static_deviation_gain`). Reference values: 1 at Nash for every precision profile; 2/3 for
both agents at the symmetric perfect-information cartel. At the asymmetric cartel corner
the informed speculator is at 1 (it already best-responds as a monopolist) and the noisy
one at 0. **So under asymmetry, collusion only requires the noisy speculator to hold back.**

Caveats: it is a static, linear approximation. It uses the session-average χ, while the
learners condition on the state, and it ignores continuation values (ρ = 0.95), so
restraint below 1 can be dynamically rational if deviations are punished. λ is held at its
equilibrium value, which is exact enough here because ξ = 500 pins λ ≈ 1/ξ.

### 5.5 Pilot design (`scripts/05_heterogeneity_pilot.py`)

Two treatments × two variants × noise levels q ∈ {0.25, 0.5, 1.0}, at σᵤ ∈ {0.1, 100}:
symmetric-noisy (q, q) keeps the benchmarks symmetric and isolates degraded information;
asymmetric (0, q) adds inequality. It is a pilot, because of section 9.

### 5.6 Controls: equal action grid and a fixed opponent (`scripts/06_control_experiment.py`)

The pilot's one-sided pattern (informed speculator at its best response, noisy one holding
back) has two explanations that involve no coordination at all.

**1. The action grid.** DGJ build each speculator's grid between its Nash and cartel levels,
extended by ι. Under symmetric information that is neutral. Under asymmetry it is not:

| Profile | informed speculator's grid | noisy speculator's grid |
|---|---|---|
| (0, 0) | 0.72–1.03 × own Nash | 0.72–1.03 × own Nash |
| (0, 0.5) | 0.97–1.33 × own Nash | −0.10–1.10 × own Nash |
| (0, 1.0) | 0.99–1.15 × own Nash | −0.10–1.10 × own Nash |

The informed speculator's cartel level lies *above* its Nash level, so it cannot hold back
below 0.97 × Nash, while the noisy speculator's range lies mostly below Nash. The control
uses an equal, Nash-centred grid: 0.7–1.3 × own Nash for every speculator (`grid="nash"`).
The price grid is left exactly as in the main runs, so only the action grid changes.

**2. A single-agent learning bias.** A Q-learner with noisy rewards may under-trade even when
nobody reacts to it. The control lets one speculator learn against an opponent fixed at its
Nash strategy (`fixed=(None, 1.0)` or `(1.0, None)`). Restraint that survives against a fixed
opponent is a property of the learner; restraint that appears only when both learn needs
the interaction, which is the collusive component.

So total restraint splits into a **single-agent part** (learner alone) and an **interaction
part** (pair relative to alone). The same split also separates DGJ's two mechanisms: their
over-pruning bias is a learning bias and can appear against a fixed opponent, while
price-trigger collusion cannot.

**Results** (MacBook Pro M4, 15 Oct 2026, 20 sessions per point; six cells re-run with the
corrected convergence rule, section 3.3, same seeds; file:
`reference_results/control/`):

| Restraint (1 = best response) | | informed: pair | alone | noisy: pair | alone |
|---|---|---|---|---|---|
| **DGJ grid**, σᵤ = 0.1 | q = 0 | 0.878 | 0.934 | 0.878 | 0.934 |
| | q = 0.5 | 1.032 | 1.074 | 0.841 | 0.845 |
| | q = 1.0 | 1.015 | 1.053 | 0.708 | 0.713 |
| **DGJ grid**, σᵤ = 100 | q = 0 | 0.853 | 0.907 | 0.853 | 0.907 |
| | q = 0.5 | 1.024 | 1.071 | 0.830 | 0.815 |
| | q = 1.0 | 1.010 | 1.049 | 0.691 | 0.686 |
| **Equal grid**, σᵤ = 0.1 | q = 0 | 0.901 | 1.001 | 0.901 | 1.001 |
| | q = 0.5 | 0.966 | 0.986 | 0.928 | 0.940 |
| | q = 1.0 | 0.966 | 0.991 | 0.863 | 0.867 |
| **Equal grid**, σᵤ = 100 | q = 0 | 0.899 | 0.971 | 0.899 | 0.971 |
| | q = 0.5 | 0.959 | 0.957 | 0.925 | 0.934 |
| | q = 1.0 | 0.955 | 0.965 | 0.876 | 0.884 |

"Alone" = the other speculator is fixed at its Nash strategy. At q = 0 both speculators see
v exactly, so "informed" and "noisy" are the same agent. Standard errors are 0.001–0.006.

**Decomposition on the equal grid** (single-agent part = 1 − alone; interaction part = alone − pair):

| | σᵤ = 0.1 single | σᵤ = 0.1 interaction | σᵤ = 100 single | σᵤ = 100 interaction |
|---|---|---|---|---|
| symmetric, both see v | 0.00 | **0.10** | 0.03 | **0.07** |
| asymmetric q = 0.5, informed | 0.01 | 0.02 | 0.04 | 0.00 |
| asymmetric q = 0.5, noisy | 0.06 | 0.01 | 0.07 | 0.01 |
| asymmetric q = 1.0, informed | 0.01 | 0.03 | 0.04 | 0.01 |
| asymmetric q = 1.0, noisy | 0.13 | 0.00 | 0.12 | 0.01 |

What this shows:

1. **The pilot's one-sided pattern is a design artifact.** Under DGJ's grid the noisy
   speculator holds back exactly as much against a fixed opponent as against a learner
   (0.841 vs 0.845, 0.708 vs 0.713, 0.830 vs 0.815, 0.691 vs 0.686): no interaction at all.
   The lone informed learner overshoots its best response (1.05–1.07) because its grid starts
   just below Nash.
2. **DGJ's grid also biases the symmetric baseline at laptop scale.** A lone, fully informed
   learner against a fixed Nash opponent under-trades (0.934, 0.907): its best response sits
   at the top of its action range. With the equal grid it finds the best response exactly
   at low noise (1.001).
3. **With symmetric information, a clear part of the restraint needs a learning opponent:**
   10% of the best response at σᵤ = 0.1, 7% at σᵤ = 100. That is the collusive component.
   At σᵤ = 100 a lone learner also holds back 3% on its own, which is DGJ's over-pruning
   bias showing up without any interaction, as their theory predicts.
4. **With unequal precision, the collusive component almost disappears:** 0–3% for the
   informed speculator and about 1% for the noisy one, against 7–10% in the symmetric pair.
   This holds at both noise-trading levels and both precision gaps.
5. **The worse-informed learner has its own under-trading bias**, 6–13% of its best response
   and growing with its noise. It does not depend on the opponent learning. In aggregate
   profits it looks like collusion (Δ > 0), but nobody is coordinating.

**For the research question:** at laptop scale, unequal signal precision makes algorithmic
collusion more fragile (H1). The coordinated part shrinks by a factor of 3–10. At the same
time a single-agent learning bias of the worse-informed algorithm produces restraint that
profit-based collusion measures (Δ) would misattribute to coordination.

Caveats: 20 sessions per point; most asymmetric pair sessions stop at the 200 M-period cap;
the restraint measure is a static approximation (5.4); one grid width (±30%) and one market
environment (I = 2, ξ = 500); laptop scale throughout. The next robustness steps are grid
widths ±20% and ±40%, more sessions, longer runs, and I = 3.

---|---|---|
| symmetric pair, both see v | 0.876 ¹ | 0.90 |
| one learner alone, opponent fixed at Nash, both see v | 0.925 | 1.00 |
| asymmetric pair (0, 0.5): informed | 1.03 ¹ | 0.955 |
| asymmetric pair (0, 0.5): noisy | 0.84 ¹ | 0.92 |
| informed learner alone, noisy opponent fixed at Nash | 1.07 | 0.99 |
| noisy learner alone, informed opponent fixed at Nash | 0.85 | 0.94 |

¹ from the full M4 run (30 / 20 sessions).

What it suggests, if it holds with 20 sessions:

- With DGJ's grid, the noisy learner holds back as much against a fixed opponent (0.85) as
  against a learner (0.84). Its restraint is a single-agent learning bias, not coordination,
  and the informed speculator's "no restraint" is partly the grid floor. **The pilot's
  one-sided pattern is largely a design artifact.**
- With DGJ's grid even a lone, fully informed learner under-trades (0.925): its best
  response sits at the top of its action range, and the learner undershoots. So at laptop
  scale part of the baseline restraint is a grid effect, not interaction.
- With the equal grid, a lone fully informed learner finds its best response exactly (1.00)
  while a pair holds back (0.90): that 10% exists only because the opponent learns too.
- Under asymmetry with the equal grid, the noisy learner holds back about 6% on its own
  (noisy rewards), and the interaction adds only 2–4% to each speculator, against 10% in the
  symmetric pair. That would mean **unequal precision weakens the collusive component (H1)
  while adding a single-agent under-trading bias for the worse-informed learner**.

---

## 6. What is scaled down and why

| | DGJ | Here | Why |
|---|---|---|---|
| I, ξ, θ, ρ, σᵤ | 2, 500, 0.1, 0.95, {0.1, 100} | same | — |
| value grid n_v | 10 | 6 | state space 2.8× smaller |
| action grid n_x | 15 | 13 | both benchmarks become exact grid points |
| price grid n_p | 31 | 37 | finer than DGJ on purpose: at laptop scale n_p = 25 produced no observable punishment, n_p = 37 did, at the same cost (section 3.3) |
| ι | 0.1 | 0.1 | — |
| market-maker window T_m | 10,000 | 10,000 | costs nothing (O(1) updates) |
| α | 0.01 | 0.01 | — |
| β | 5×10⁻⁷ | 5×10⁻⁷ | — |
| convergence window | 10⁶ periods | 10⁵, counted only once exploration < 1% | 10× cheaper; the exploration rule prevents premature stops |
| period cap | none: 20 M – 50 B periods | 2×10⁸ | laptop budget; capped sessions are reported separately |
| sessions per point | 1,000 | 30 (20 in the pilot) | laptop budget |
| implementation | C++, 400-core cluster | Python + Numba, one laptop | — |

Speed: about 4.5 M periods per second per core on the 2-core cloud machine used for
testing (Calvano: 9 M/s). Low-noise sessions converge after roughly 60–80 M periods;
high-noise sessions (σᵤ ≥ e³) do not converge within 2×10⁸ and are capped.

---

## 7. Validation evidence (`pytest -q`, 38 tests)

| Test | What it establishes |
|---|---|
| Calvano Nash/monopoly prices = 1.4729 / 1.9250 | demand and benchmark code |
| Nash price is a best response | FOC solved correctly |
| 9th grid price = 1.74 | grid definition matches the paper |
| 30-session replication 0.78 < Δ < 0.92 | learning code (full run: 0.849) |
| χᴺ = 1/((I+1)λ), χᴹ = 1/(2Iλ) for I = 2, 3 | benchmark solver vs DGJ closed forms |
| Kyle limit λ = σᵥ/(2σᵤ) | benchmark solver vs Kyle (1985) |
| no profitable unilateral deviation from Nash, 4 precision profiles | Nash solver |
| cartel joint profit > Nash; asymmetric corner χ₂ᴹ = 0 | cartel solver |
| discrete moments = Monte Carlo | signal model |
| restraint = 1 and deviation gain = 0 at Nash (3 profiles); = 2/3 at the symmetric cartel | restraint measure |
| both speculators fixed at Nash inside the learning loop earn the analytical Nash profits; a fixed speculator's Q-table never changes; the Nash-centred grid spans 0.7–1.3 × Nash and leaves the price grid unchanged; default results are bit-identical with the control options present | control options |
| same seeds on Apple M4 (arm64) and the x86 test machine: same learning paths, Δ equal to 14 digits | cross-platform reproducibility (checked by hand, not in pytest) |
| **fixed benchmark strategies in the simulated market earn the analytical profits** (σᵤ = 0.1, 100; symmetric and asymmetric) | the whole environment: noise traders, investors, adaptive market maker, profit accounting |
| market maker learns λ̂ = 1/ξ | rolling regressions |
| benchmarks are grid points | action grid |
| identical seeds → identical results; parallel = serial | reproducibility |
| no session counts as converged before exploration < 1% in every signal state | convergence rule |
| Variant A = Variant B without signal noise | variants are a clean extension |

The fixed-strategy test is the key one for Rung 2: it shows the market itself is right
before any learning happens. If the learners then behave differently from DGJ's, the cause
is in learning and scale, not in the environment.

---

## 8. Verification run (2-core cloud machine)

Files: `reference_results/`. New runs write to `results/`.

These come from a verification run with fewer sessions than the defaults (12 per DGJ
point, 8 in the pilot). Runs with the default sizes differ in the second decimal.

### 8.1 Calvano baseline (`01`, 1,000 sessions)

| Δ mean (s.e.) | median | p1–p99 | converged | constant prices / 2-cycles / longer | wall clock |
|---|---|---|---|---|---|
| 0.849 (0.004) | 0.868 | 0.55–1.00 | 100% | 63% / 25% / 12% | 109 s on 2 cores |

### 8.2 DGJ baseline, symmetric information (`02`, 12 sessions each)

| | σᵤ = 0.1 | σᵤ = 100 |
|---|---|---|
| Δ mean (s.e.) | 0.384 (0.021) | 0.545 (0.016) |
| converged | 100% (median 66 M periods) | 0% (all at the 200 M cap) |
| aggressiveness / Nash | 0.91 | 0.90 |
| DGJ (their scale) | ≈ 0.75 | high (over-pruning regime) |
| seconds per session per core | ≈ 16 | ≈ 43 |

### 8.3 Noise sweep (`03`, 12 sessions per point)

| ln σᵤ | −5 | −3 | −1 | 1 | 3 | 5 |
|---|---|---|---|---|---|---|
| Δ mean | 0.398 | 0.355 | 0.397 | 0.373 | 0.284 | 0.708 |
| s.e. | 0.032 | 0.019 | 0.019 | 0.027 | 0.011 | 0.014 |
| converged | 100% | 100% | 100% | 100% | 8% | 0% |

U-shape with its minimum at ln σᵤ = 3, as in DGJ Figure 2A; the low-noise side is flat
here, where DGJ's falls from about 0.75 towards the minimum.

### 8.4 Impulse responses at σᵤ = 0.1 (`04`, 12 sessions × 5,000 paired paths)

Change in aggressiveness at t = 4, % of its mean, ± 95% CI:

| Shock (% of mean \|x\|) | 0.25% | 2.5% | 11.5% | 15% |
|---|---|---|---|---|
| n_p = 37 (default) | +0.02 ± 0.04 | +0.13 ± 0.14 | **+2.04 ± 0.42** | **+3.24 ± 0.36** |
| n_p = 25 | +0.01 ± 0.10 | +0.00 ± 0.12 | +0.05 ± 0.49 | +0.10 ± 0.50 |
| n_p = 25, α = 0.05 | +0.05 ± 0.06 | +0.03 ± 0.14 | +0.02 ± 0.47 | +0.02 ± 0.61 |

With the default grid: no reaction to small and medium shocks, a clear punishment after
large ones, gone by t = 5. With n_p = 25 there is none. Δ hardly changes with the grid
(0.384 vs 0.380), so in these runs the profit gain does not depend on punishment.

### 8.5 Stability of capped sessions (σᵤ = 100, n_p = 25, 6 sessions)

| Cap | 50 M | 100 M | 200 M |
|---|---|---|---|
| mean Δ | 0.522 | 0.479 | 0.507 |

The mean is stable; individual sessions still move by up to ±0.1 between caps.

### 8.6 Heterogeneity pilot at σᵤ = 0.1 (`05`, 8 sessions per point)

| Treatment | Variant | q | Δ (s.e.) | converged | cartel gain over Nash | aggressiveness / own Nash (spec. 1, 2) | profit / own Nash (1, 2) |
|---|---|---|---|---|---|---|---|
| baseline | — | 0 | 0.403 (0.024) | 100% | 12.5% | 0.91, 0.91 | — |
| symmetric | A | 0.5 | 0.520 (0.027) | 0% | 8.1% | 0.91, 0.91 | 1.04, 1.04 |
| symmetric | A | 1.0 | 0.596 (0.068) | 0% | 3.2% | 0.91, 0.91 | 1.02, 1.02 |
| symmetric | B | 0.5 | 0.556 (0.029) | 0% | 8.1% | 0.90, 0.90 | 1.05, 1.05 |
| symmetric | B | 1.0 | 0.681 (0.079) | 25% | 3.2% | 0.91, 0.91 | 1.02, 1.02 |
| asymmetric | A | 0.5 | 0.134 (0.013) | 0% | 14.2% | 1.09, 0.79 | 1.08, 0.89 |
| asymmetric | A | 1.0 | 0.337 (0.016) | 75% | 10.0% | 1.06, 0.68 | 1.06, 0.87 |
| asymmetric | B | 0.5 | 0.134 (0.040) | 0% | 14.2% | 1.10, 0.79 | 1.08, 0.88 |
| asymmetric | B | 1.0 | 0.317 (0.024) | 0% | 10.0% | 1.06, 0.67 | 1.06, 0.86 |

A quick run on the M4 (15 Oct 2026, 6 sessions per point, seeds 1–6) reproduces
these points: the sessions follow the same learning paths on both machines, and Δ agrees
to 14 significant digits.

Restraint relative to best response (section 5.4), from the M4 quick run:

| Treatment | Variant | q | informed / spec. 1 | noisy / spec. 2 | static deviation gain (1, 2) |
|---|---|---|---|---|---|
| baseline | — | 0 | 0.876 | 0.874 | 1.6%, 1.6% |
| symmetric | A | 0.5 | 0.877 | 0.875 | 1.5%, 1.6% |
| symmetric | A | 1.0 | 0.896 | 0.897 | 1.1%, 1.1% |
| symmetric | B | 0.5 | 0.868 | 0.868 | 1.8%, 1.8% |
| symmetric | B | 1.0 | 0.889 | 0.890 | 1.3%, 1.2% |
| asymmetric | A | 0.5 | 1.029 | 0.841 | 0.1%, 2.6% |
| asymmetric | A | 1.0 | 1.017 | 0.721 | 0.0%, 8.4% |
| asymmetric | B | 0.5 | 1.032 | 0.850 | 0.1%, 2.3% |
| asymmetric | B | 1.0 | 1.012 | 0.698 | 0.0%, 10.0% |

What the pilot suggests, and what it does not show. **Read section 5.6 first: the control
shows that the asymmetric pattern below is largely produced by DGJ's action grid.**

- **Symmetric noise does not make the learners more collusive.** Restraint stays at
  0.87–0.90 and if anything weakens at q = 1.0; profit over Nash falls (5.0% → 4.2% → 1.9%).
  Δ rises only because noisy signals pull the cartel towards Nash (χᴹ/χᴺ goes from 0.75 to
  0.85), so the same restraint covers more of a smaller gap. The rising blue curves in the
  Δ panel must not be read as "noise makes collusion more robust".
- **Asymmetric noise produces one-sided restraint, under DGJ's grid.** The informed speculator trades at its
  best response (restraint 1.01–1.03, no gain from deviating). All the restraint comes from
  the noisy speculator, and it deepens with its own noise (0.84 → 0.72 in A, 0.85 → 0.70
  in B). The noisy one earns 11–14% *less* than its Nash profit; the informed one 6–8% more.
- **It is not Stackelberg leadership.** With speculator 1 as Stackelberg leader the informed
  side would trade at 1.30 χᴺ (q = 0.5), against 1.09 observed, and joint Δ would be negative
  (−0.51), against +0.14 observed. The noisy speculator also trades well below its best
  response to the informed one (0.79 χᴺ against a best response of 0.94 χᴺ), which a
  Stackelberg follower would not do.
- **Variants A and B are indistinguishable**, for restraint as well as Δ. So the monitoring
  channel plays no visible role at this scale, and the noisy agent's restraint deepening
  with its own noise points to a learning bias (its signal errors are reward noise, which is
  DGJ's over-pruning mechanism), not to a punishment threat.

Six sessions per point, most of them capped before convergence: this is a lead, not a
result.

All outputs of this test run are in `reference_results/`; the n_p = 25 versions and the
stability check are in `reference_results/sensitivity/` (not produced by the scripts at their defaults).

---

## 9. What does not reproduce, and what it means for the project

**1. The level of collusion at low noise is lower.** Δ ≈ 0.38 at σᵤ = 0.1 against DGJ's
≈ 0.75. Profits are still supra-competitive in every session, and speculators trade at
about 91% of Nash aggressiveness.

**2. The price-trigger evidence is partial and grid-dependent.** With the default price
grid (n_p = 37) the learned strategies punish large deviations at t = 4 and ignore small
ones, which is the threshold logic of DGJ's Figure 3. But the punishment lasts one period
instead of about two, it grows with the shock instead of being flat above the threshold,
and with a coarser grid (n_p = 25) it disappears entirely while Δ stays the same. So at
laptop scale the profit gain does not rest on punishment. Most of it is probably
over-pruning, with trigger behaviour layered on top where the price state is fine enough
to see deviations. Likely reasons for the gap to DGJ: the 10× shorter convergence window,
far fewer periods than their 20 M – 50 B, and the coarser value grid.

**3. High-noise sessions do not converge** within 2×10⁸ periods. Late in learning a few
hundred greedy choices per million periods still flip, mostly in states where v is close to
v̄ and every action earns almost the same. The cross-session mean Δ is stable between
50 M and 200 M periods (8.5), but by DGJ's criterion these sessions have not converged.
They are reported separately (`delta_capped`). The same holds for most heterogeneous-signal
runs: signal noise adds reward noise, so they hit the cap too.

**Why this matters for the research question.** Variant A vs B is a contrast in the
*monitoring* channel, which is what price-trigger collusion runs on. At this scale the
monitoring channel exists (8.4) but is weak and sensitive to the price grid. So:

- a difference between Variants A and B can be interpreted only together with impulse
  responses for each variant, and only if the baseline punishment is robust across grids;
- a null result in Variant B is weak evidence, because there was little monitoring to degrade;
- the honest next step before any heterogeneity result is a robustness study of the
  baseline mechanism (grid, convergence window, number of periods). At DGJ's scale that
  needs a computing cluster, which is a concrete reason to want a supervisor.

**What the controls change (5.6).** The one-sided restraint in the pilot was a
consequence of DGJ's action grid under asymmetry plus a single-agent learning bias of the
noisy learner, not evidence of coordination. With an equal grid, the collusive part of the
restraint shrinks from 7–10% (symmetric) to 0–3% (asymmetric). The design for the real experiment follows
from that: an equal, Nash-centred action grid for every speculator, and every treatment run
twice, with both speculators learning and with the opponent fixed at Nash, so restraint
can be split into a single-agent part and an interaction part. The research question then
becomes measurable: does unequal precision change the *interaction* part?

---

## 10. References

- Athey, S. & Bagwell, K. (2001). Optimal collusion with private information. *RAND Journal of Economics* 32(3).
- Calvano, E., Calzolari, G., Denicolò, V. & Pastorello, S. (2020). Artificial intelligence, algorithmic pricing, and collusion. *American Economic Review* 110(10), 3267–3297.
- Colliard, J.-E., Foucault, T. & Lovo, S. (2026). Algorithmic pricing and liquidity in securities markets. *Review of Financial Studies*, advance article.
- Dou, W. W., Goldstein, I. & Ji, Y. (2025). AI-powered trading, algorithmic collusion, and price efficiency. NBER Working Paper 34054.
- Fershtman, C. & Pakes, A. (2012). Dynamic games with asymmetric information: a framework for empirical work. *Quarterly Journal of Economics* 127(4).
- Green, E. J. & Porter, R. H. (1984). Noncooperative collusion under imperfect price information. *Econometrica* 52(1).
- Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica* 53(6).
- Kyle, A. S. & Xiong, W. (2001). Contagion as a wealth effect. *Journal of Finance* 56(4).
- Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. *Machine Learning* 8.
