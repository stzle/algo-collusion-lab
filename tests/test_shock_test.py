"""Rules of scripts/10_shock_test.py (pre-registration 4) and the fixed-speculator
impulse response."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("shock", ROOT / "scripts" / "10_shock_test.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def _rec(su, prof, who, resp, n=20, noise=0.001, seed=0):
    """Synthetic record: every session has response `resp` (shocks x speculators) plus noise."""
    rng = np.random.default_rng(seed)
    base = np.array(resp, dtype=float)
    R = base[None, :, :] + noise * rng.standard_normal((n,) + base.shape)
    return {"sigma_u": su, "q1": prof[0], "q2": prof[1], "who": who, "R_t4": R}


def _records(pair_resp, alone_resp, prof=(0.0, 0.0), su=0.1):
    recs = [_rec(su, prof, "pair", pair_resp, seed=1)]
    if prof[0] == prof[1]:
        recs.append(_rec(su, prof, "only0", alone_resp, seed=2))
    else:
        recs.append(_rec(su, prof, "only0", alone_resp, seed=2))
        recs.append(_rec(su, prof, "only1", alone_resp, seed=3))
    return recs


def test_punishment_and_trigger():
    pair = [[0.0, 0.0], [0.002, 0.002], [0.03, 0.03], [0.035, 0.035]]
    alone = [[0.0, 0.0], [0.0, 0.0], [0.005, 0.005], [0.005, 0.005]]
    d = T.punishment({T.key(r): r for r in _records(pair, alone)}, 0.1, (0.0, 0.0))
    assert d["spec1"]["punishes"] and d["spec1"]["trigger_like"]


def test_no_punishment_when_alone_reacts_the_same():
    resp = [[0.0, 0.0], [0.002, 0.002], [0.03, 0.03], [0.035, 0.035]]
    d = T.punishment({T.key(r): r for r in _records(resp, resp)}, 0.1, (0.0, 0.0))
    assert not d["spec1"]["punishes"]


def test_small_effects_do_not_count():
    pair = [[0.0, 0.0], [0.0, 0.0], [0.004, 0.004], [0.004, 0.004]]     # below 0.5 %
    alone = [[0.0, 0.0]] * 4
    d = T.punishment({T.key(r): r for r in _records(pair, alone, )}, 0.1, (0.0, 0.0))
    assert not d["spec1"]["punishes"]


def test_verdict_chain():
    pun = [[0.0, 0.0], [0.002, 0.002], [0.03, 0.03], [0.035, 0.035]]
    none = [[0.0, 0.0]] * 4
    recs = _records(pun, none) + _records(none, none, prof=(0.0, 1.0)) + _records(pun, none, prof=(1.0, 1.0))
    v = T.analyze(recs, [0.1])
    assert v["T1"] == "yes"
    assert v["T2"] == "punishment-based"
    assert v["T3"] == "inequality removes punishment"
    assert v["T4"] == "supported"
    v = T.analyze(_records(none, none), [0.1])
    assert v["T1"] == "no" and v["T3"] == "inconclusive"


def test_plan_size():
    assert len(T.plan([0.1, 100.0])) == 24


def test_fixed_speculator_does_not_react():
    import _dgj_common as G
    from lab import dgj
    prm = G.make_params(sigma_u=0.1, sigma_eps=(0.0, 0.5), fixed=(None, 1.0), grid="nash",
                        max_periods=1_000_000, conv_window=10_000, eval_periods=10_000)
    res = dgj.simulate(prm, np.arange(1, 2))
    I = dgj.session_irf(res, 0, prm, n_paths=50)
    assert np.all(I["dchi"][:, :, 1] == 0.0)
    assert np.allclose(I["base_chi"][:, 1], res["env"]["fixed_chi"][1])
