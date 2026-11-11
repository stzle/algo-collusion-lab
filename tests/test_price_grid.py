"""Price grid per lagged value (pre-registration 8) and plan of scripts/13_price_grid_audit.py."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import _dgj_common as G  # noqa: E402
from lab import dgj  # noqa: E402

spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts" / "13_price_grid_audit.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def test_one_order_step_moves_price_by_about_one_grid_point():
    """DGJ footnote 16: with n_p = 2 n_x + 1 a one-step change in one speculator's order
    changes the price on the grid."""
    prm = G.make_params(sigma_u=0.1, n_v=10, n_x=15, n_p=31, grid="benchmarks", price_grid="per_value")
    env = dgj.setup(prm)
    d = np.abs(env["v_grid"] - prm.v_bar)
    step = (env["chi_grid"][0, 1] - env["chi_grid"][0, 0]) * env["lam0"] * d
    assert np.all(step / env["dp"] > 0.9) and np.all(step / env["dp"] < 1.2)
    # the single grid used so far is far coarser
    env_g = dgj.setup(G.make_params(sigma_u=0.1, n_v=10, n_x=15, n_p=31, grid="benchmarks"))
    assert np.all(env_g["dp"] == env_g["dp"][0])
    assert env_g["dp"][0] > 7 * step.max()


def test_default_is_the_single_grid():
    assert dgj.DGJParams().price_grid == "global"
    with pytest.raises(ValueError):
        dgj.setup(G.make_params(price_grid="something"))


def test_per_value_grid_runs_with_noisy_signals_and_controls():
    prm = G.make_params(sigma_u=0.1, sigma_eps=(0.0, 1.0), fixed=(1.0, None), grid="nash",
                        price_grid="per_value", max_periods=300_000, conv_window=10_000, eval_periods=10_000)
    res = dgj.simulate(prm, np.arange(1, 3))
    assert np.all(np.isfinite(res["profit"]))


def test_plan():
    p = A.plan(["L", "P"], [0.1, 100.0])
    assert len(p) == 2 * 7 + 2 and len({A.key(x) for x in p}) == len(p)
    assert A.params_for(p[0]).price_grid == "per_value"
    assert A.params_for(p[-1]).max_periods == 10_000_000_000
