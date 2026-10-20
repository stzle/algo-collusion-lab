# Algorithmic collusion lab

Q-learning agents that learn to collude without communicating, built in three steps:

1. **Calvano, Calzolari, Denicolò & Pastorello (2020, AER)** — two pricing algorithms in a
   Bertrand duopoly. Replicated at the published scale (1,000 sessions); the published
   profit gain Δ ≈ 0.85 is the check that the learning code is correct.
2. **Dou, Goldstein & Ji (2025, NBER WP 34054)** — informed AI speculators in a Kyle-type
   market with an adaptive market maker. Implemented from the paper's equations and run
   at laptop scale. The environment is validated against analytical benchmarks; the
   learning results are a *qualitative* comparison, not a replication.
3. **Extension (this project)** — speculators observe the value with *different precision*.
   Benchmarks are derived for any precision profile; the code supports two monitoring
   variants (A: realised past value, B: own past signal). Results so far
   are a pilot.

Author: Renat Shyhin. The code was developed with the help of an AI coding assistant;
model choices, validation and results are documented here and in `TECHNICAL.md`.

`TECHNICAL.md` explains every equation, where it lives in the code, every deviation
from the papers and why, and what does not reproduce.

---

## Setup on a MacBook Pro (Apple M4)

Everything runs natively on Apple Silicon. Numba compiles the simulation loops to
machine code for the M4 and spreads independent sessions over all cores.

**1. Open Terminal and check you are on a native arm64 shell**

```bash
uname -m
```

It must print `arm64`. If it prints `x86_64`, Terminal runs under Rosetta.

Copy only the commands in these boxes. In macOS Terminal (zsh), anything you add after a
command, including a `# note`, is passed to the command as an argument.

**2. Install `uv`** (a fast Python and package manager; it also downloads Python itself)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen Terminal afterwards. If you use Homebrew, `brew install uv` does the same.

**3. Create the environment** (inside the unzipped project folder)

```bash
cd ~/code/algo-collusion-lab
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Use the folder where you put the project; the safest way to get the path right is to type
`cd ` and drag the folder from Finder into the Terminal window. Keep it out of iCloud
Drive, and keep the folder name free of spaces. `uv venv` downloads an arm64 Python 3.12
if needed. Repeat `source .venv/bin/activate` in every new Terminal window.

Python 3.12 is used because every package here ships ready-made Apple Silicon wheels
for it. Without uv: install Python 3.12 from python.org (the "universal2" installer is
native on M4), then `python3.12 -m venv .venv`, `source .venv/bin/activate`,
`pip install -r requirements.txt`.

**4. Check the installation and measure your Mac**

```bash
python scripts/00_check_setup.py
pytest -q
```

`pytest` should report 48 passed.

`00_check_setup.py` warns if Python runs as x86_64 (about 2× slower), compiles the
code once (cached afterwards, so later runs start immediately), prints your
performance/efficiency core counts and estimates how long each script will take on
**your** machine.

## Running the experiments

Run them in this order. Each script prints its headline numbers and writes JSON and
PNG files to `results/`. The outputs of a verification run (2-core cloud machine, fewer
sessions) and of the full M4 control run are in `reference_results/`, so you can compare. The same seed gives the same
learning path on Apple Silicon and on the x86 test machine (checked on an M4:
Δ equal to 14 significant digits), so differences come only from running more sessions.

| Script | What it does | Default size |
|---|---|---|
| `01_calvano_baseline.py` | Rung 1: Calvano baseline, Δ vs published 0.85, deviation-and-punishment figure | 1,000 sessions |
| `02_dgj_baseline.py` | Rung 2: DGJ at σᵤ = 0.1 and σᵤ = 100 | 30 sessions each |
| `03_dgj_sigma_sweep.py` | Δ across ln σᵤ = −5 … 5 (DGJ Fig. 2A analogue) | 6 × 30 sessions |
| `04_dgj_impulse_response.py` | Price-trigger test (DGJ Fig. 3 analogue) | 30 sessions |
| `05_heterogeneity_pilot.py` | Rung 3 pilot: symmetric vs asymmetric noise, Variants A and B | 26 points × 20 sessions |
| `06_control_experiment.py` | Controls for the pilot: equal action grid, and one learner against a fixed Nash opponent | 32 points × 20 sessions |
| `07_robustness.py` | Robustness of the first results: grid widths, variant B, longer learning; applies the rules in `PREREGISTRATION.md` | 92 points × 20 sessions (~2.5 h) |
| `08_precision_sweep.py` | Collusive part across the precision gap, incl. a both-noisy profile; rules in `PREREGISTRATION.md` (part 2) | 44 points × 20 sessions (~1 h) |

A small version of everything, to check that it all works:

```bash
QUICK=1 zsh scripts/run_all.sh
```

Everything at default size (keeps the Mac awake on its own):

```bash
zsh scripts/run_all.sh
```

or one at a time:

```bash
python scripts/01_calvano_baseline.py
caffeinate -i python scripts/03_dgj_sigma_sweep.py
```

`caffeinate -i` keeps the Mac awake while the script runs.

The full default set adds up to about 8 core-hours on the test machine (most of it the
pilot), which should be somewhere between half an hour and an hour and a half on an M4,
depending on the chip. `00_check_setup.py` prints the estimate for yours.

Measured on a 2-core cloud machine (an M4 has 10–16 cores and faster cores, so expect
several times faster; `00_check_setup.py` gives the estimate for your machine):

| Run | Sessions | Wall clock, 2 cores |
|---|---|---|
| 01 Calvano | 1,000 | 1.8 min |
| 02 DGJ baseline | 12 per point | 1.6 min (σᵤ = 0.1), 4.3 min (σᵤ = 100) |
| 04 impulse response | 12 | 2 min |
| 03 sweep | 12 per point | 16 min |
| 05 pilot (σᵤ = 0.1 only, 2 noise levels) | 8 per point | 35 min |

Long runs: plug in the charger, turn off Low Power Mode, keep the lid open, and start
the script with `caffeinate -i` so macOS does not sleep. Sessions at high noise
(σᵤ ≥ e³) hit the 200 M-period cap, so they take the longest.

Useful options (every script has `--help`):

| Command | What it does |
|---|---|
| `python scripts/02_dgj_baseline.py --sessions 10` | quick look with fewer sessions |
| `python scripts/04_dgj_impulse_response.py --sigma-u 100` | impulse responses at high noise |
| `python scripts/04_dgj_impulse_response.py --set n_p=25` | coarser price grid (writes to its own folder) |
| `python scripts/05_heterogeneity_pilot.py --sigma-u 0.1 --noise 0.5 1.0` | smaller pilot |
| `python scripts/01_calvano_baseline.py --threads 4` | limit the number of cores |
| `python scripts/05_heterogeneity_pilot.py --plot-only` | redraw a chart from saved results |
| `python scripts/06_control_experiment.py --sigma-u 0.1` | control at low noise only (about half the time) |
| `python scripts/07_robustness.py --stage caps` | only the longer-learning check (rule R2) |
| `python scripts/07_robustness.py --analyze-only` | verdicts from saved results, without running anything |
| `open results` | show the results folder in Finder |

Every session has its own seed, and parallel runs give bit-identical results to serial
runs, so on the same machine every number reproduces exactly.

## Repository map

```
lab/calvano.py          Calvano model, Nash/monopoly benchmarks, Q-learning, deviation test
lab/dgj_benchmarks.py   Nash and cartel benchmarks for any signal-precision profile
lab/dgj.py              DGJ market: adaptive market maker, Q-learning speculators,
                        evaluation, impulse responses, fixed-strategy validation
scripts/00 … 06         run scripts (see table above)
tests/                  38 checks: published values, closed forms, Kyle limit,
                        environment vs theory, reproducibility
TECHNICAL.md            equations ↔ code, derivations, scaling decisions, open problems
```
