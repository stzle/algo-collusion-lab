# Checks of weak points

Rules: PREREGISTRATION.md, pre-registration 6.

- **S1 (phantom correction):** correction rejected — closer to corrected in 2 of 12 cases
- **S2 (main result with the measured clean part):** holds
- **S3 (rebound at equal noise 1):** real
- **H1 (main result under other α, β):** robust
- **B1 (baseline at the paper's grid sizes):** not reproduced — Δ = 0.308 ± 0.007 (paper ≈ 0.75), 100% of 50 runs converged

At the paper's scale: collusive part 4.4 %, learning bias 7.3 %.

| σᵤ | profile | speculator | measured | corrected | clean | ± |
|---:|---|---|---:|---:|---:|---:|
| 0.1 | 0, 0 | both | +10.2 | +10.2 | +9.6 | 0.5 |
| 0.1 | 0.25, 0.25 | both | +2.5 | +1.3 | +2.1 | 0.3 |
| 0.1 | 0.5, 0.5 | both | +2.4 | +0.2 | +1.9 | 0.3 |
| 0.1 | 1, 1 | both | +4.7 | +2.7 | +3.8 | 0.1 |
| 0.1 | 0, 0.25 | spec1 | +4.9 | +4.0 | +4.1 | 0.4 |
| 0.1 | 0, 0.25 | spec2 | +3.5 | +3.3 | +3.0 | 0.4 |
| 0.1 | 0, 1 | spec1 | +2.5 | +0.8 | +1.7 | 0.3 |
| 0.1 | 0, 1 | spec2 | +0.4 | -0.2 | -0.1 | 0.4 |
| 100 | 0, 0 | both | +7.2 | +5.8 | +6.8 | 0.2 |
| 100 | 0.25, 0.25 | both | +2.7 | +0.7 | +2.3 | 0.2 |
| 100 | 0.5, 0.5 | both | +3.0 | +0.7 | +2.1 | 0.2 |
| 100 | 1, 1 | both | +4.4 | +2.4 | +3.3 | 0.2 |
| 100 | 0, 0.25 | spec1 | +3.1 | +1.4 | +2.7 | 0.3 |
| 100 | 0, 0.25 | spec2 | +3.0 | +1.4 | +2.4 | 0.2 |
| 100 | 0, 1 | spec1 | +1.1 | -0.4 | +1.0 | 0.2 |
| 100 | 0, 1 | spec2 | +0.8 | -1.5 | -0.6 | 0.3 |

| setting | c at (0,0) | ± | converged | ρ (0,1) | ρ (0.5,0.5) | H1 |
|---|---:|---:|---:|---:|---:|---|
| alpha=0.005 | 10.7 | 0.5 | 100% | 0.28 | 0.28 | holds |
| alpha=0.02 | 9.8 | 0.5 | 100% | 0.17 | 0.20 | holds |
| beta=2.5e-07 | 8.3 | 0.7 | 100% | 0.11 | 0.23 | holds |
| beta=1e-06 | 10.1 | 0.7 | 100% | 0.26 | 0.31 | holds |
