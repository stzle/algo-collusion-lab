# Data tables — what every column means

Two tables, made by `scripts/export_data.py` from the saved results of every experiment.
Every number in the final report can be traced to a row here.

- **all_runs.csv** — one row per *run*: one parameter setting, learned 20 times with
  seeds 1–20. Numbers are averages over those 20 learning runs.
- **derived.csv** — the quantities the verdicts are computed from, one row per
  information profile and speculator.

## The setting in one paragraph

Two speculators learn by Q-learning to trade on a private signal about an asset's value
in a market with noise traders and a market maker who sets prices (Dou, Goldstein & Ji
2025). Speculator 1 and 2 can receive signals of different precision. We measure how far
each trades below its best possible trade ("restraint"), and whether that needs the other
to be learning too ("collusive part").

## all_runs.csv

| Column | Meaning |
|---|---|
| experiment | Which experiment: `robustness` (pre-registration 1), `precision` (2 and 3), `shock` (4), `final_M` / `final_P` / `final_G` (5) |
| sigma_u | Noise-trading volume σᵤ: 0.1 = low (prices informative), 100 = high |
| noise_spec1, noise_spec2 | Standard deviation of the signal noise of each speculator (0 = sees the value exactly). The asset value has standard deviation 1. |
| grid_width | Action grid: each speculator chooses aggressiveness between (1 − w) and (1 + w) × its own Nash level |
| monitoring | What the speculators remember about the last period: `realized` = the true value (variant A), `own_signal` = their own past signal (variant B) |
| period_cap | Maximum learning length in periods; runs that have not converged stop here |
| value_levels | Number of possible asset values (6 by default; 10 in the finer-grid check) |
| who_learns | `pair` = both learn; `spec1_alone` / `spec2_alone` = only this speculator learns, the other plays its fixed Nash strategy (control) |
| runs | Number of independent learning runs (seeds) averaged in this row |
| converged_share | Share of runs whose learned strategies stopped changing for 100,000 periods (after exploration < 1 %). The rest hit the period cap. |
| restraint_spec1/2 | Average aggressiveness ÷ the best response to what the other speculator actually does. 1 = no restraint, 0.9 = trades 10 % below its best response. Empty for a speculator that does not learn. |
| restraint_se_spec1/2 | Standard error of that average across the 20 runs |
| delta_mean, delta_se | Δ: where joint profits lie between competition (0) and the joint-profit maximum (1) |
| informativeness | Price informativeness var(Σx) / var(u): informed order flow relative to noise trading (final run only) |
| price_impact | λ̂: how much the market maker moves the price per unit of order flow (liquidity is its inverse) (final run only) |
| mispricing | Average distance between the expected price and the true value, over value levels (final run only) |
| profit_spec1/2 | Average profit per period of each speculator (final run only) |
| lambda_nash, lambda_cartel | Price impact in the theoretical competitive (Nash) and joint-profit-maximum (cartel) benchmarks |
| profit_nash_spec*, profit_cartel_spec* | Each speculator's profit in those two benchmarks. Under unequal precision the worse-informed speculator earns *less* in the cartel than in competition. |
| shock{s}_response_spec1/2 | Shock test: change in aggressiveness one period after an adverse noise-trade shock of size s % of the average order, relative to the same path without the shock, as a share of the speculator's average aggressiveness. Positive = trades more aggressively. |
| shock{s}_response_se_spec1/2 | Its standard error across runs |
| date | When the experiment finished (MacBook Pro M4) |

## derived.csv

| Column | Meaning |
|---|---|
| experiment … value_levels | As above; identifies the setting |
| speculator | `both` for equal information (the two are interchangeable), otherwise `spec1` / `spec2` |
| restraint_pair | Restraint when both learn (equal information: mean of the two) |
| restraint_alone | Restraint of the same speculator learning alone against a fixed Nash opponent |
| collusive_part | restraint_alone − restraint_pair: restraint that appears only when both learn |
| collusive_part_se | Its standard error (the two sets of runs are treated as independent) |
| learning_bias | 1 − restraint_alone: restraint without anyone to coordinate with |
| phantom_part | *Exploratory, not pre-registered.* The collusive part that would appear if both speculators simply kept their lone-learner behaviour, with no coordination at all. It arises because each one's best response rises when the other trades less. Assumes each speculator's solo bias stays the same in a pair. |
| collusive_part_corrected | collusive_part − phantom_part: coordination beyond independent learning biases |
| converged_share_pair | converged_share of the pair runs |
| punishment_shock{s}[_spec1/2] | Shock test only: shock response in the pair minus that of the lone learner. Punishment rule (pre-registration 4): above 2 SE and ≥ 0.005 at both the 11.5 % and the 15 % shock. |
| punishment_se_shock{s}[…] | Its standard error |

## Things to keep in mind when checking

- Rows with the same parameters in different experiments (e.g. width 0.3, cap 2 × 10⁸ in
  `robustness` and `precision`) are repeated runs with the same seeds and code, so they agree
  exactly. That is a check, not extra evidence.
- Low `converged_share` is normal for unequal or noisy information at this scale; the
  period-cap check (robustness, caps 1–8 × 10⁸) shows how much the averages still move.
- All decision rules and thresholds are in `PREREGISTRATION.md`, written before each run.
