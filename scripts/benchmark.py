"""Benchmark steepest descent / Fletcher-Reeves CG / Levenberg-Marquardt.

For every test graph and every random starting layout (same start for all three
methods) record iterations, wall time, evaluations, final energy, edge
crossings and the empirical convergence ratio.

    python scripts/benchmark.py            # 20 starts per graph (~5 min)
    python scripts/benchmark.py --seeds 5  # quick check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from forcelayout.energy import LayoutEnergy  # noqa: E402
from forcelayout.graphs import PLANAR, REGISTRY, random_layout  # noqa: E402
from forcelayout.metrics import count_crossings, edge_length_cv, tail_rate  # noqa: E402
from forcelayout.optimizers import METHODS  # noqa: E402
from forcelayout.viz import COLORS  # noqa: E402

GRAPHS = ["grid", "net", "tree", "clusters"]   # clusters (K6 cliques) is non-planar
LABEL = {"sd": "Steepest descent", "cg": "Conjugate gradient (FR)", "lm": "Levenberg-Marquardt"}


def run(seeds: int):
    rows = []
    for gname in GRAPHS:
        G = REGISTRY[gname]()
        E = LayoutEnergy(G.n, G.edges)
        for seed in range(seeds):
            z0 = random_layout(G.n, seed)
            start_cross = count_crossings(z0, G.edges)
            for key, method in METHODS.items():
                r = method(E, z0)
                rows.append(dict(
                    graph=G.name, family=gname, seed=seed, method=LABEL[key],
                    iters=r.n_iter, seconds=r.seconds[-1],
                    n_f=r.n_f[-1], n_g=r.n_g[-1], n_J=r.n_J[-1],
                    energy=r.energy[-1], gnorm=r.gnorm[-1], converged=int(r.converged),
                    start_crossings=start_cross, final_crossings=count_crossings(r.z, G.edges),
                    edge_cv=edge_length_cv(r.z, G.edges), rate=tail_rate(r.gnorm),
                ))
            print(f"  {G.name} seed {seed + 1}/{seeds} done", flush=True)
    return rows


def _planar_stat(runs, agg, getter):
    """Crossing statistics only make sense for graphs with a crossing-free drawing."""
    vals = [getter(r) for r in runs if r["family"] in PLANAR]
    return float(agg(vals)) if vals else float("nan")


def summarise(rows):
    """Per (graph family, method) medians and success rates."""
    out = []
    methods = list(LABEL.values())
    for fam in GRAPHS + ["ALL"]:
        sub = [r for r in rows if fam == "ALL" or r["family"] == fam]
        best = {}
        for r in sub:
            best[r["family"]] = min(best.get(r["family"], np.inf), r["energy"])
        for m in methods:
            s = [r for r in sub if r["method"] == m]
            near_best = [r["energy"] <= 1.01 * best[r["family"]] for r in s]
            out.append(dict(
                family=fam, method=m, runs=len(s),
                converged_pct=100 * np.mean([r["converged"] for r in s]),
                median_iters=np.median([r["iters"] for r in s]),
                median_seconds=np.median([r["seconds"] for r in s]),
                median_energy=np.median([r["energy"] for r in s]),
                near_best_energy_pct=100 * np.mean(near_best),
                median_final_crossings=_planar_stat(s, np.median, lambda r: r["final_crossings"]),
                zero_crossings_pct=_planar_stat(s, lambda v: 100 * np.mean(np.array(v) == 0),
                                                lambda r: r["final_crossings"]),
                median_edge_cv=np.median([r["edge_cv"] for r in s]),
                median_rate=np.nanmedian([r["rate"] for r in s]),
            ))
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def figure(summary, path):
    methods = list(LABEL.values())
    fams = GRAPHS
    x = np.arange(len(fams))
    w = 0.26
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    panels = [
        ("median_iters", "Median iterations to converge", True),
        ("median_seconds", "Median wall-clock time (s)", True),
        ("zero_crossings_pct", "% of runs ending crossing-free\n(planar graphs only)", False),
    ]
    for a, (key, title, log) in zip(ax, panels):
        for i, m in enumerate(methods):
            vals = [next(s[key] for s in summary if s["family"] == f and s["method"] == m) for f in fams]
            vals = [0 if np.isnan(v) else v for v in vals]          # clusters: no crossing metric
            a.bar(x + (i - 1) * w, vals, w, color=COLORS[m], label=m)
        a.set_xticks(x, fams)
        a.set_title(title, fontsize=11)
        if log:
            a.set_yscale("log")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="y", alpha=0.25)
    ax[2].set_ylim(0, 100)
    ax[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    rows = run(args.seeds)
    summary = summarise(rows)
    write_csv(out / "benchmark_runs.csv", rows)
    write_csv(out / "benchmark_summary.csv", summary)
    figure(summary, out / "benchmark.png")

    print("\nfamily     method                     iters   sec    near-best%  0-cross%  conv%")
    for s in summary:
        print(f"{s['family']:<10} {s['method']:<25} {s['median_iters']:6.0f} {s['median_seconds']:6.2f}"
              f" {s['near_best_energy_pct']:9.0f} {s['zero_crossings_pct']:9.0f} {s['converged_pct']:6.0f}")
