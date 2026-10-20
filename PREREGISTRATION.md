# Pre-registration: robustness of the first results

Written on 19 October 2026, **before** any of the runs below were made. The decision rules
are fixed here so that "robust" cannot be redefined after the numbers are in. Changes to
this file after the runs start are visible in the git history and must be explained in a
dated note at the end. `scripts/07_robustness.py` applies these rules automatically.

## What the first results claim

From `reference_results/control` (equal action grid ±30 %, 20 runs per point):

- Under **equal** precision, each speculator trades 7–10 % below its static best response,
  and only when the other one also learns. This is the *collusive part* of restraint.
- Under **unequal** precision (noise q = 0.5 or 1.0 for speculator 2), the collusive part
  falls to 0–3 %.
- The worse-informed speculator still trades 6–13 % below its best response against a
  *fixed* Nash opponent. This is a single-agent *learning bias*, not coordination.

Known weakness: almost all runs with unequal precision (and all at σᵤ = 100) hit the cap
of 2 × 10⁸ periods without formally converging. Only one grid width has been tried.

## Measures

For speculator *i* in a cell, *r* = average aggressiveness ÷ static best response to the
other's actual behaviour (1 = no restraint). With *r*<sub>pair</sub> from runs where both
learn and *r*<sub>fixed</sub> from runs where the other is fixed at its Nash strategy:

- collusive part *c* = *r*<sub>fixed</sub> − *r*<sub>pair</sub>
- learning bias *b* = 1 − *r*<sub>fixed</sub>

Under equal precision *c*<sub>sym</sub> uses the mean of both speculators in the pair.
Standard errors are across runs; the SE of a difference treats the two runs as
independent (different simulations, same seed list).

## Design

| Stage | Cells | Purpose |
|---|---|---|
| grid | widths ±20 %, ±30 %, ±40 % × σᵤ ∈ {0.1, 100} × q ∈ {0, 0.5, 1.0}, pair + fixed-opponent controls, variant A | R1, S1, H3 |
| monitor | width ±30 %, same cells, variant B (own past signal) | Teilfrage: monitoring channel |
| caps | width ±30 %, variant A, all cells, caps 1, 2, 4 × 10⁸ periods, same seeds | R2 |

20 runs (seeds 1–20) per point. Same seeds across caps, so a longer cap continues the
same learning path; the cap comparison is paired.

## Decision rules

**R1 — Fragility (H1) survives the grid width.** At a given width and σᵤ the test is
*informative* only if the symmetric collusive part exists: *c*<sub>sym</sub> ≥ 0.03 and
*c*<sub>sym</sub> > 2 SE. If informative, H1 *holds* there if for both q and both
speculators *c*<sub>asym</sub> ≤ 0.5 · *c*<sub>sym</sub>.
- *Robust*: holds at every informative width and both σᵤ, with at least two widths informative.
- *Grid-dependent*: holds at some informative widths, fails at others. Reported as such.
- *Fails*: fails at ±30 % (the width the first results came from).

**R2 — Capped runs are not transients.** For every cell where fewer than half of the runs
converged: each speculator's mean *r* moves by at most 0.01 between the 2 × 10⁸ and
4 × 10⁸ caps, and the R1 verdict at ±30 % is the same at both caps. If only the
1 × 10⁸ and 2 × 10⁸ caps are available, the same rule on those is a *provisional* pass.
If R2 fails, the first results are reported as statements about a fixed learning budget,
not about learned long-run behaviour.

**S1 — The learning bias of the worse-informed speculator is not a grid artefact.** Its
*b* ≥ 0.03 and > 2 SE at every width, both σᵤ, both q.

**H3 proxy — the two mechanisms react differently.** The collusive part corresponds to
coordinated restraint, the learning bias to restraint without coordination. Supported if,
at ±30 % and both σᵤ, the collusive part falls by at least half from equal to unequal
precision (R1 holds) **and** the learning bias of the worse-informed speculator does not
fall by half (*b*<sub>asym</sub> ≥ 0.5 · max(*b*<sub>sym</sub>, 0.03)). This is a proxy;
the impulse-response test named in the Exposé remains the main H3 test.

**Monitoring channel (A vs B).** Two-sided, no predicted direction. A difference is
reported if |*c*<sub>B</sub> − *c*<sub>A</sub>| > max(0.02, 2 SE) in a cell. Otherwise
"no detectable difference at this scale", which is not evidence of no difference.

## What is not decided by these runs

Results at the scale of Dou, Goldstein & Ji (larger grids, longer windows), more than two
speculators, and the impulse-response analysis.

## Amendments

**20 October 2026 — added after seeing the results.** The planned runs finished on
19 October (`reference_results/robustness`). R2 failed at σᵤ = 100: between the
2 × 10⁸ and 4 × 10⁸ caps the collusive part moved by at most 0.003, but the restraint of
the worse-informed speculator against a fixed opponent moved by up to 0.030. As a
follow-up, not part of the original design, the σᵤ = 100 cells are rerun at an
8 × 10⁸ cap with the same seeds. The R2 verdict above stays "fail" whatever this shows;
the follow-up only tells whether the learning bias levels off, and is reported as an
exploratory check.

---

# Pre-registration 2: precision sweep

Written on 20 October 2026, **before** any of these runs. Applied automatically by
`scripts/08_precision_sweep.py`.

## Question

The first results use two noise levels for speculator 2 (q = 0.5, 1.0) while
speculator 1 sees the value exactly. The research question is about *continuous*
differences in precision. Two things are open:

1. **Shape.** Does the collusive part fade gradually as the precision gap grows, or is it
   already gone at a tiny gap? At q = 0.1 the Nash strategies of the two speculators differ
   by less than 1 %, but the cartel benchmark jumps to a corner where only the informed
   speculator trades. So a collapse at q = 0.1 would point to the changed incentives of
   joint profit maximisation rather than to the information gap itself.
2. **Corner.** Does fragility need one perfectly informed speculator? With noise
   (0.25, 0.75) both speculators trade in the cartel benchmark.

## Design

Equal grid ±30 %, variant A, cap 2 × 10⁸, 20 runs (seeds 1–20), σᵤ ∈ {0.1, 100}.

| Profile (q₁, q₂) | Cells | Role |
|---|---|---|
| (0, 0) | pair, one learner alone | reference for P1 |
| (0, q), q ∈ {0.1, 0.25, 0.5, 1, 2} | pair, each speculator alone against the other fixed at Nash | P1 |
| (0.5, 0.5) | pair, one learner alone | reference for P2 (same average noise) |
| (0.25, 0.75) | pair, each speculator alone | P2 |

Measures as in pre-registration 1. For an unequal profile, ρ = the larger of the two
speculators' collusive parts divided by the collusive part of its reference profile.

## Decision rules

A reference is *informative* if its collusive part is ≥ 0.03 and > 2 SE; otherwise the rule
reports "uninformative".

**P1 — shape** (per σᵤ, reference (0, 0)):
- *Collapse*: ρ(0.1) ≤ 0.5.
- *Gradual*: ρ(0.1) > 0.5 and ρ(q) ≤ 0.5 for q = 1 and q = 2.
- *Mixed*: anything else (reported with the full curve).
Overall: the common verdict if both σᵤ agree, otherwise "depends on noise trading".

**P2 — no perfectly informed speculator needed** (per σᵤ, reference (0.5, 0.5)):
*Holds* if ρ(0.25, 0.75) ≤ 0.5. Overall: holds if it holds at every informative σᵤ.

**Descriptive, no rule:** the learning bias of the worse-informed speculator across q.

## Amendments

None yet.
