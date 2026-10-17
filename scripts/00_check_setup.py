"""
Step 0: check the installation and measure this machine's speed.

    python scripts/00_check_setup.py

Prints versions, warns if Python is running under Rosetta (x86_64 on an Apple
Silicon Mac — about 2x slower), compiles the Numba code once (cached afterwards)
and estimates how long each experiment script will take on this machine.
"""
import argparse
import platform
import subprocess
import time

import _common as C

import numpy as np


def apple_cores():
    """Performance / efficiency core counts on Apple Silicon (None elsewhere)."""
    try:
        p = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip())
        e = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel1.physicalcpu"]).strip())
        return p, e
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    threads = C.init_threads(args.threads)
    info = C.machine_info(threads)
    for k, v in info.items():
        print(f"{k:>10}: {v}")

    if platform.system() == "Darwin":
        cores = apple_cores()
        if cores:
            print(f"     cores: {cores[0]} performance + {cores[1]} efficiency")
        if platform.machine() != "arm64":
            print("\n!! Python is running as x86_64 (Rosetta). Install a native arm64 Python "
                  "(see README) — Numba runs about 2x faster natively.\n")

    from lab import calvano as cv
    from lab import dgj

    print("\nCompiling (first run only, ~30-60 s; cached afterwards) ...")
    prm = cv.CalvanoParams()
    prices = cv.price_grid(prm)
    PI = cv.profit_matrix(prices, prm)
    Q0 = cv.initial_Q(PI, prm.delta)
    t0 = time.perf_counter()
    cv.run_session(1, PI, Q0, prm.alpha, prm.beta, prm.delta, 1000, 2000)
    d = dgj.DGJParams(max_periods=20_000, conv_window=10**9, eval_periods=1_000)
    dgj.simulate(d, [1])
    cv.run_sessions(np.array([1, 2], dtype=np.int64), PI, Q0, prm.alpha, prm.beta, prm.delta, 1000, 2000)
    print(f"done in {time.perf_counter() - t0:.0f} s")

    # speed: periods per second on one core
    Q, conv, t, s = cv.run_session(2, PI, Q0, prm.alpha, prm.beta, prm.delta, 10**9, 5_000_000)
    t0 = time.perf_counter()
    cv.run_session(3, PI, Q0, prm.alpha, prm.beta, prm.delta, 10**9, 5_000_000)
    cal_rate = 5_000_000 / (time.perf_counter() - t0)

    d = dgj.DGJParams(max_periods=5_000_000, conv_window=10**9, eval_periods=1_000)
    env = dgj.setup(d)
    dgj.simulate(d, [4], env)
    t0 = time.perf_counter()
    dgj.simulate(d, [5], env)
    dgj_rate = 5_000_000 / (time.perf_counter() - t0)

    print(f"\nSpeed on one core: Calvano {cal_rate / 1e6:.1f} M periods/s, "
          f"DGJ market {dgj_rate / 1e6:.1f} M periods/s")

    # rough runtime estimates (periods per session measured on the reference runs)
    eff = threads if not (platform.system() == "Darwin" and apple_cores()) else \
        apple_cores()[0] + 0.4 * apple_cores()[1]
    # periods per session measured in the test runs: Calvano ~1.8 M; DGJ ~70 M when it
    # converges, otherwise the 200 M cap; sessions with noisy signals run ~1.7x slower per
    # period (calibrated on an M4 run, Oct 2026)
    conv, cap, slow = 70e6, 200e6, 1.7
    noisy_cap = cap * slow
    est = {
        "01_calvano_baseline (1000 sessions)": 1000 * 1.8e6 / cal_rate / eff,
        "02_dgj_baseline (2 x 30 sessions)": 30 * (conv + cap) / dgj_rate / eff,
        "03_dgj_sigma_sweep (6 x 30 sessions)": 30 * (4 * conv + 2 * cap) / dgj_rate / eff,
        "04_dgj_impulse_response (30 sessions)": 30 * conv / dgj_rate / eff + 60,
        "05_heterogeneity_pilot (26 x 20 sessions)": 20 * (conv + cap + 24 * noisy_cap) / dgj_rate / eff,
        "06_control_experiment (32 x 20 sessions)": 20 * (2 * (2 * conv) + 2 * (2 * cap) + 24 * noisy_cap) / dgj_rate / eff,
    }
    print(f"\nEstimated wall-clock time with {threads} threads (rough, +/- 50%):")
    for k, v in est.items():
        print(f"  {k:<45} {v / 60:6.1f} min")
    print("\nSetup OK.")


if __name__ == "__main__":
    main()
