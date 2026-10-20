# Robustness verdicts

Rules: PREREGISTRATION.md. c = collusive part, b = learning bias, both in % of the best response; conv = share of runs that converged (the rest hit the cap).

- **R1 (H1 survives grid width):** robust  — per width: {'0.2': ['holds', 'holds'], '0.3': ['holds', 'holds'], '0.4': ['holds', 'holds']}
- **S1 (bias of the worse-informed speculator):** holds
- **H3 proxy:** supported
- **R2 (caps), σᵤ = 0.1:** pass  — 1e+08->2e+08: max move 0.036 over 3 capped cells, R1 same: True; 2e+08->4e+08: max move 0.008 over 1 capped cells, R1 same: True
- **R2 (caps), σᵤ = 100.0:** fail  — 1e+08->2e+08: max move 0.050 over 7 capped cells, R1 same: True; 2e+08->4e+08: max move 0.030 over 7 capped cells, R1 same: True; 4e+08->8e+08: max move 0.018 over 4 capped cells, R1 same: True
- **A vs B:** difference in 0 of 10 cases

| σᵤ | width | monitor | cap | case | c | ± | b | ± | conv | R1 |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| 0.1 | 0.2 | realized | 2e+08 | sym | +8.1 | 0.4 | +0.4 | 0.3 | 100% | holds |
| 0.1 | 0.2 | realized | 2e+08 | informed_q0.5 | +1.2 | 0.2 | +1.2 | 0.2 | 0% |  |
| 0.1 | 0.2 | realized | 2e+08 | noisy_q0.5 | +1.2 | 0.4 | +4.1 | 0.2 | 0% |  |
| 0.1 | 0.2 | realized | 2e+08 | informed_q1 | +1.1 | 0.2 | +0.9 | 0.1 | 5% |  |
| 0.1 | 0.2 | realized | 2e+08 | noisy_q1 | +0.7 | 0.3 | +8.1 | 0.2 | 5% |  |
| 0.1 | 0.3 | realized | 1e+08 | sym | +10.2 | 0.5 | -0.1 | 0.3 | 100% | holds |
| 0.1 | 0.3 | realized | 1e+08 | informed_q0.5 | +1.3 | 0.4 | +1.5 | 0.3 | 0% |  |
| 0.1 | 0.3 | realized | 1e+08 | noisy_q0.5 | +0.9 | 0.4 | +4.5 | 0.3 | 0% |  |
| 0.1 | 0.3 | realized | 1e+08 | informed_q1 | +2.0 | 0.3 | +0.9 | 0.2 | 0% |  |
| 0.1 | 0.3 | realized | 1e+08 | noisy_q1 | -0.8 | 0.4 | +10.9 | 0.2 | 0% |  |
| 0.1 | 0.3 | realized | 2e+08 | sym | +10.2 | 0.5 | -0.1 | 0.3 | 100% | holds |
| 0.1 | 0.3 | realized | 2e+08 | informed_q0.5 | +2.0 | 0.3 | +1.4 | 0.3 | 0% |  |
| 0.1 | 0.3 | realized | 2e+08 | noisy_q0.5 | +1.2 | 0.4 | +6.0 | 0.2 | 0% |  |
| 0.1 | 0.3 | realized | 2e+08 | informed_q1 | +2.5 | 0.3 | +0.9 | 0.2 | 5% |  |
| 0.1 | 0.3 | realized | 2e+08 | noisy_q1 | +0.4 | 0.4 | +13.3 | 0.2 | 5% |  |
| 0.1 | 0.3 | realized | 4e+08 | sym | +10.2 | 0.5 | -0.1 | 0.3 | 100% | holds |
| 0.1 | 0.3 | realized | 4e+08 | informed_q0.5 | +2.4 | 0.3 | +1.4 | 0.3 | 35% |  |
| 0.1 | 0.3 | realized | 4e+08 | noisy_q0.5 | +1.3 | 0.4 | +6.7 | 0.2 | 35% |  |
| 0.1 | 0.3 | realized | 4e+08 | informed_q1 | +2.7 | 0.3 | +0.9 | 0.2 | 95% |  |
| 0.1 | 0.3 | realized | 4e+08 | noisy_q1 | +1.6 | 0.4 | +13.3 | 0.2 | 95% |  |
| 0.1 | 0.4 | realized | 2e+08 | sym | +10.4 | 0.7 | -0.5 | 0.4 | 100% | holds |
| 0.1 | 0.4 | realized | 2e+08 | informed_q0.5 | +2.7 | 0.4 | +2.1 | 0.2 | 0% |  |
| 0.1 | 0.4 | realized | 2e+08 | noisy_q0.5 | +2.7 | 0.6 | +6.1 | 0.3 | 0% |  |
| 0.1 | 0.4 | realized | 2e+08 | informed_q1 | +2.9 | 0.3 | +1.1 | 0.2 | 0% |  |
| 0.1 | 0.4 | realized | 2e+08 | noisy_q1 | +0.6 | 0.4 | +17.0 | 0.3 | 0% |  |
| 0.1 | 0.3 | own_signal | 2e+08 | sym | +10.2 | 0.5 | -0.1 | 0.3 | 100% | holds |
| 0.1 | 0.3 | own_signal | 2e+08 | informed_q0.5 | +2.0 | 0.3 | +1.4 | 0.3 | 0% |  |
| 0.1 | 0.3 | own_signal | 2e+08 | noisy_q0.5 | +1.2 | 0.4 | +6.5 | 0.2 | 0% |  |
| 0.1 | 0.3 | own_signal | 2e+08 | informed_q1 | +2.3 | 0.3 | +0.9 | 0.2 | 0% |  |
| 0.1 | 0.3 | own_signal | 2e+08 | noisy_q1 | -1.4 | 0.3 | +13.2 | 0.2 | 0% |  |
| 100 | 0.2 | realized | 2e+08 | sym | +5.2 | 0.2 | +2.4 | 0.1 | 0% | holds |
| 100 | 0.2 | realized | 2e+08 | informed_q0.5 | +0.3 | 0.2 | +2.7 | 0.1 | 0% |  |
| 100 | 0.2 | realized | 2e+08 | noisy_q0.5 | +0.3 | 0.2 | +4.6 | 0.1 | 0% |  |
| 100 | 0.2 | realized | 2e+08 | informed_q1 | +0.8 | 0.2 | +2.4 | 0.1 | 0% |  |
| 100 | 0.2 | realized | 2e+08 | noisy_q1 | +1.5 | 0.2 | +6.6 | 0.1 | 0% |  |
| 100 | 0.3 | realized | 1e+08 | sym | +7.9 | 0.2 | +1.9 | 0.2 | 0% | holds |
| 100 | 0.3 | realized | 1e+08 | informed_q0.5 | +0.3 | 0.2 | +3.3 | 0.2 | 0% |  |
| 100 | 0.3 | realized | 1e+08 | noisy_q0.5 | +0.9 | 0.3 | +4.5 | 0.2 | 0% |  |
| 100 | 0.3 | realized | 1e+08 | informed_q1 | +0.7 | 0.2 | +3.2 | 0.1 | 0% |  |
| 100 | 0.3 | realized | 1e+08 | noisy_q1 | +1.0 | 0.3 | +6.6 | 0.2 | 0% |  |
| 100 | 0.3 | realized | 2e+08 | sym | +7.2 | 0.2 | +2.9 | 0.2 | 0% | holds |
| 100 | 0.3 | realized | 2e+08 | informed_q0.5 | -0.1 | 0.2 | +4.3 | 0.1 | 0% |  |
| 100 | 0.3 | realized | 2e+08 | noisy_q0.5 | +0.8 | 0.3 | +6.6 | 0.2 | 0% |  |
| 100 | 0.3 | realized | 2e+08 | informed_q1 | +1.1 | 0.2 | +3.5 | 0.1 | 0% |  |
| 100 | 0.3 | realized | 2e+08 | noisy_q1 | +0.8 | 0.3 | +11.6 | 0.2 | 0% |  |
| 100 | 0.3 | realized | 4e+08 | sym | +7.3 | 0.2 | +3.2 | 0.2 | 0% | holds |
| 100 | 0.3 | realized | 4e+08 | informed_q0.5 | +0.2 | 0.3 | +4.2 | 0.2 | 0% |  |
| 100 | 0.3 | realized | 4e+08 | noisy_q0.5 | +0.5 | 0.3 | +8.0 | 0.2 | 0% |  |
| 100 | 0.3 | realized | 4e+08 | informed_q1 | +1.4 | 0.2 | +3.5 | 0.1 | 0% |  |
| 100 | 0.3 | realized | 4e+08 | noisy_q1 | +0.5 | 0.3 | +14.6 | 0.2 | 0% |  |
| 100 | 0.3 | realized | 8e+08 | sym | +7.2 | 0.2 | +3.6 | 0.1 | 60% | holds |
| 100 | 0.3 | realized | 8e+08 | informed_q0.5 | +0.6 | 0.2 | +4.1 | 0.1 | 0% |  |
| 100 | 0.3 | realized | 8e+08 | noisy_q0.5 | +0.6 | 0.3 | +9.1 | 0.2 | 0% |  |
| 100 | 0.3 | realized | 8e+08 | informed_q1 | +1.6 | 0.2 | +3.5 | 0.1 | 0% |  |
| 100 | 0.3 | realized | 8e+08 | noisy_q1 | +0.3 | 0.2 | +16.5 | 0.1 | 0% |  |
| 100 | 0.4 | realized | 2e+08 | sym | +7.6 | 0.3 | +3.5 | 0.2 | 0% | holds |
| 100 | 0.4 | realized | 2e+08 | informed_q0.5 | +1.3 | 0.3 | +4.5 | 0.2 | 0% |  |
| 100 | 0.4 | realized | 2e+08 | noisy_q0.5 | +0.9 | 0.3 | +8.5 | 0.1 | 0% |  |
| 100 | 0.4 | realized | 2e+08 | informed_q1 | +1.8 | 0.3 | +4.2 | 0.2 | 0% |  |
| 100 | 0.4 | realized | 2e+08 | noisy_q1 | +1.4 | 0.4 | +15.4 | 0.3 | 0% |  |
| 100 | 0.3 | own_signal | 2e+08 | sym | +7.2 | 0.2 | +2.9 | 0.2 | 0% | holds |
| 100 | 0.3 | own_signal | 2e+08 | informed_q0.5 | +0.2 | 0.2 | +4.3 | 0.1 | 0% |  |
| 100 | 0.3 | own_signal | 2e+08 | noisy_q0.5 | +1.2 | 0.3 | +6.5 | 0.1 | 0% |  |
| 100 | 0.3 | own_signal | 2e+08 | informed_q1 | +1.1 | 0.2 | +3.5 | 0.1 | 0% |  |
| 100 | 0.3 | own_signal | 2e+08 | noisy_q1 | +1.4 | 0.2 | +10.4 | 0.2 | 0% |  |

| σᵤ | case | c (A) | c (B) | B − A | ± | differs |
|---:|---|---:|---:|---:|---:|---|
| 0.1 | sym | +10.2 | +10.2 | +0.0 | 0.7 | no |
| 0.1 | informed_q0.5 | +2.0 | +2.0 | -0.0 | 0.5 | no |
| 0.1 | noisy_q0.5 | +1.2 | +1.2 | +0.0 | 0.5 | no |
| 0.1 | informed_q1 | +2.5 | +2.3 | -0.2 | 0.4 | no |
| 0.1 | noisy_q1 | +0.4 | -1.4 | -1.9 | 0.5 | no |
| 100 | sym | +7.2 | +7.2 | +0.0 | 0.3 | no |
| 100 | informed_q0.5 | -0.1 | +0.2 | +0.4 | 0.3 | no |
| 100 | noisy_q0.5 | +0.8 | +1.2 | +0.4 | 0.4 | no |
| 100 | informed_q1 | +1.1 | +1.1 | -0.0 | 0.3 | no |
| 100 | noisy_q1 | +0.8 | +1.4 | +0.6 | 0.4 | no |
