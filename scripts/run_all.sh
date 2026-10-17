#!/bin/zsh
# Runs every experiment at its default size and keeps the Mac awake while doing so.
# Usage (from the project folder, with the environment active):
#     zsh scripts/run_all.sh            # everything
#     QUICK=1 zsh scripts/run_all.sh    # small version to check that everything works (~10-15 min on an M4)
set -e
cd "$(dirname "$0")/.."
if [[ "$QUICK" == "1" ]]; then
  S_CAL=200; S=8; S_PILOT=6; PILOT_ARGS=(--sigma-u 0.1 --noise 0.5 1.0); CTRL_ARGS=(--sigma-u 0.1)
else
  S_CAL=1000; S=30; S_PILOT=20; PILOT_ARGS=(); CTRL_ARGS=()
fi
run() { printf "\n=== %s ===\n" "$*"; caffeinate -i python "$@"; }
run scripts/00_check_setup.py
run scripts/01_calvano_baseline.py --sessions $S_CAL
run scripts/02_dgj_baseline.py --sessions $S
run scripts/04_dgj_impulse_response.py --sessions $S
run scripts/03_dgj_sigma_sweep.py --sessions $S
run scripts/05_heterogeneity_pilot.py --sessions $S_PILOT "${PILOT_ARGS[@]}"
run scripts/06_control_experiment.py --sessions $S_PILOT "${CTRL_ARGS[@]}"
printf "\nAll done. Results are in results/.\n"
