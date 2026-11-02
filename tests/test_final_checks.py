"""Plan and rules of scripts/11_final_checks.py (pre-registration 5)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("final", ROOT / "scripts" / "11_final_checks.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


def test_plan():
    p = F.plan(["M", "P", "G"], [0.1, 100.0])
    assert len(p) == 14 * 2 + 2 + 10 * 2
    assert len({F.key(x) for x in p}) == len(p)
    assert all(x["n_v"] == 10 for x in p if x["part"] == "G")


def _g_records(c_sym, c_asym, c_noisy, su=0.1):
    """Synthetic part-G records with given collusive parts (restraint alone = 1)."""
    recs = []
    def add(prof, who, r):
        recs.append(dict(part="G", n_v=10, sigma_u=su, q1=prof[0], q2=prof[1], who=who, sessions=20,
                         converged_share=1.0, restraint=r, restraint_se=[0.002, 0.002]))
    add((0.0, 0.0), "pair", [1 - c_sym, 1 - c_sym]); add((0.0, 0.0), "only0", [1.0, 1.0])
    add((0.5, 0.5), "pair", [1 - c_noisy, 1 - c_noisy]); add((0.5, 0.5), "only0", [1.0, 1.0])
    for q in (0.5, 1.0):
        add((0.0, q), "pair", [1 - c_asym, 1 - c_asym])
        add((0.0, q), "only0", [1.0, 1.0]); add((0.0, q), "only1", [1.0, 1.0])
    return recs


def test_g_rules():
    v = F.analyze(_g_records(0.10, 0.02, 0.03), [0.1])
    assert v["G1"] == "holds" and v["G2"] == "holds"
    v = F.analyze(_g_records(0.10, 0.06, 0.08), [0.1])
    assert v["G1"] == "fails" and v["G2"] == "fails"
    v = F.analyze(_g_records(0.01, 0.0, 0.0), [0.1])
    assert v["G1"] == "uninformative"


def test_p5_mapping():
    rng = np.random.default_rng(0)
    def rec(who, resp):
        R = np.array(resp)[None] + 0.001 * rng.standard_normal((20, 4, 2))
        return dict(part="P", n_v=6, sigma_u=0.1, q1=0.125, q2=0.125, who=who, sessions=20, R_t4=R)
    pun = [[0, 0], [0.002, 0.002], [0.03, 0.03], [0.035, 0.035]]
    none = [[0, 0]] * 4
    assert F.analyze([rec("pair", pun), rec("only0", none)], [0.1])["P5"].startswith("inequality")
    assert F.analyze([rec("pair", none), rec("only0", none)], [0.1])["P5"].startswith("noise")
