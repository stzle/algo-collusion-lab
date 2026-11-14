"""
Figure: collusive part c against the measured correlation of the two discretised signals,
for three families (existing data, no simulation runs):
  equal precision (q, q)        results/precision_fine/points.json
  unequal precision (0, q)      same file; c = mean over the two speculators
  shared noise (0.5, 0.5), w    results/shared_noise/points.json
One panel per sigma_u with data. Points of the matched experiment (results/matched_correlation)
are overlaid when they exist. Error bars: the saved SEs (formula of the source scripts).

    python scripts/plot_correlation.py       # a few seconds

Outputs: results/figures/correlation.png, results/figures/correlation.csv
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import _common as C


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NI = _load("noise_ineq", "09_noise_vs_inequality.py")
SN = _load("shared", "15_shared_noise.py")
MCOR = _load("matched", "18_matched_correlation.py")

FAMILIES = {"equal": "equal precision (q, q)", "unequal": "unequal precision (0, q)",
            "shared": "shared noise, (0.5, 0.5) with share w", "matched": "matched experiment: (0, q)"}


def _points(path):
    return json.loads(path.read_text())["points"] if path.exists() else []


def rows(results=C.RESULTS) -> list[dict]:
    v = MCOR.V_GRID
    corr = lambda q1, q2, w=0.0: MCOR.measured_correlation(v, q1, q2, w)
    out = []
    pf = _points(results / "precision_fine" / "points.json")
    idx = {NI.S.key(r): r for r in pf}
    for su, q1, q2 in sorted({(float(r["sigma_u"]), float(r["q1"]), float(r["q2"])) for r in pf}):
        fam = "equal" if q1 == q2 else ("unequal" if q1 == 0 else None)
        c = NI.c_of(idx, su, (q1, q2)) if fam else None
        if c is not None:
            out.append(dict(sigma_u=su, family=fam, profile=f"({q1:g}, {q2:g})", correlation=corr(q1, q2),
                            c=c[0], c_se=c[1]))
    sn = _points(results / "shared_noise" / "points.json")
    idx = {SN.key(r): r for r in sn}
    for su in sorted({float(r["sigma_u"]) for r in sn}):
        for w in SN.SHARES:
            d = SN.collusive_part(idx, su, w)
            if d is not None:
                out.append(dict(sigma_u=su, family="shared", profile=f"w = {w:g}",
                                correlation=corr(SN.Q, SN.Q, w), c=d["c"], c_se=d["c_se"]))
    mv = results / "matched_correlation" / "verdicts.json"
    if mv.exists():
        for w, e in json.loads(mv.read_text())["levels"].items():
            if "c_unequal" in e:
                out.append(dict(sigma_u=MCOR.SIGMA_U, family="matched", profile=f"(0, {e['q']:.3f}), level w = {w}",
                                correlation=e["corr_achieved"], c=e["c_unequal"], c_se=e["c_unequal_se"]))
    return out


def plot(rs, path):
    sigmas = sorted({r["sigma_u"] for r in rs})
    style = {"equal": (C.SERIES[0], "o"), "unequal": (C.SERIES[1], "s"), "shared": (C.SERIES[2], "D"),
             "matched": (C.INK, "*")}
    plt = C.plot_style()
    fig, axes = plt.subplots(1, len(sigmas), figsize=(6.2 * len(sigmas), 4.6), squeeze=False, sharey=True)
    for ax, su in zip(axes[0], sigmas):
        for fam, (col, mk) in style.items():
            pts = sorted((r for r in rs if r["sigma_u"] == su and r["family"] == fam), key=lambda r: r["correlation"])
            if not pts:
                continue
            x, y, e = ([r[k] for r in pts] for k in ("correlation", "c", "c_se"))
            ax.errorbar(x, [100 * c for c in y], yerr=[200 * s for s in e], color=col, marker=mk, ms=8 if mk != "*" else 13,
                        lw=0 if fam == "matched" else 2, elinewidth=1.2, capsize=0, label=FAMILIES[fam])
        ax.axhline(0, color=C.INK_2, lw=1)
        ax.set_title(f"σᵤ = {su:g}")
        ax.set_xlabel("Measured correlation of the two signals")
    axes[0, 0].set_ylabel("Collusive part c (% of best response), ± 2 SE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=160)


def main():
    rs = rows()
    out = C.out_dir("figures")
    with open(out / "correlation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sigma_u", "family", "profile", "correlation", "c", "c_se"])
        w.writeheader()
        w.writerows(rs)
    plot(rs, out / "correlation.png")
    for r in rs:
        print(f"σu={r['sigma_u']:<5g} {r['family']:<8} {r['profile']:<26} corr {r['correlation']:.4f}  "
              f"c {100 * r['c']:+.1f} ± {100 * r['c_se']:.1f}")
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
