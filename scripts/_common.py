"""Shared helpers for the run scripts: paths, threads, saving, plotting style."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"

# chart style: reference palette slots 1-4 (fixed order), recessive axes
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def init_threads(n: int | None) -> int:
    """Use all cores by default and hand out sessions one at a time, so fast
    performance cores and slower efficiency cores (Apple M-series) stay busy."""
    import numba
    if n:
        numba.set_num_threads(n)
    numba.set_parallel_chunksize(1)
    return numba.get_num_threads()


def machine_info(threads: int) -> dict:
    import numba
    import numpy
    import scipy
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "processor": platform.processor(),
        "numba": numba.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "threads": threads,
    }


def out_dir(name: str) -> Path:
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj) -> None:
    def default(o):
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=default))


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": INK_2,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "lines.linewidth": 2.0, "legend.frameon": False,
    })
    return plt


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def summary_stats(x) -> dict:
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0}
    return {"n": n, "mean": float(x.mean()),
            "se": float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "p01": float(np.percentile(x, 1)), "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)), "p75": float(np.percentile(x, 75)),
            "p99": float(np.percentile(x, 99))}


def env_flag_quick() -> bool:
    return os.environ.get("QUICK", "") == "1"
