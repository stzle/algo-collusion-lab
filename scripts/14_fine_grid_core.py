"""
Core experiments repeated with the price grid per lagged value
(PREREGISTRATION.md, "Pre-registration 9"). Runs, one after the other:

  08_precision_sweep.py      -> results/precision_fine   (rules P1, P2)
  09_noise_vs_inequality.py  -> results/precision_fine   (rules I1, I2)
  10_shock_test.py           -> results/shock_fine       (rules T1-T4)

    python scripts/14_fine_grid_core.py          # 96 runs, about 2 h on an M4
    python scripts/14_fine_grid_core.py --smoke  # tiny budget, only to see that it runs

Each script saves after every run; starting this again continues where it stopped.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
smoke = "--smoke" in sys.argv
suffix = "_smoke" if smoke else ""
small = ["--sessions", "2", "--max-periods", "1e6"] if smoke else []
jobs = [
    ["08_precision_sweep.py", "--out", "precision_fine" + suffix] + small,
    ["09_noise_vs_inequality.py", "--out", "precision_fine" + suffix] + small,
    ["10_shock_test.py", "--out", "shock_fine" + suffix] + small + (["--paths", "100"] if smoke else []),
]
for job in jobs:
    cmd = [sys.executable, str(HERE / job[0]), "--price-grid", "per_value"] + job[1:]
    print("\n=== " + " ".join(cmd[1:]) + " ===", flush=True)
    subprocess.run(cmd, check=True)
print("\nAll three finished. Results in results/precision_fine and results/shock_fine.")
