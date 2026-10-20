"""Live demo: untangle one graph with all three optimisers.

    python scripts/run_demo.py --graph net --seed auto
    python scripts/run_demo.py --graph grid --seed 3

Writes snapshots, final-layout comparison, convergence curves and an animated
GIF of conjugate gradient untangling the graph into ``results/``.

``--seed auto`` takes the first starting layout (seed 0, 1, 2, ...) from which
Fletcher-Reeves CG ends crossing-free.  That is a *showcase* choice; the
benchmark (scripts/benchmark.py) reports how often that happens over many starts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forcelayout import viz  # noqa: E402
from forcelayout.energy import LayoutEnergy  # noqa: E402
from forcelayout.graphs import REGISTRY, random_layout  # noqa: E402
from forcelayout.metrics import count_crossings, edge_length_cv, tail_rate  # noqa: E402
from forcelayout.optimizers import conjugate_gradient, levenberg_marquardt, steepest_descent  # noqa: E402


def pick_seed(G, E, max_seed=60):
    skipped = []
    for s in range(max_seed):
        z0 = random_layout(G.n, s)
        r = conjugate_gradient(E, z0)
        if r.converged and count_crossings(r.z, G.edges) == 0 and count_crossings(z0, G.edges) >= 10:
            return s, skipped
        skipped.append(s)
    raise RuntimeError("no showcase seed found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="net", choices=sorted(REGISTRY))
    ap.add_argument("--seed", default="auto")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    G = REGISTRY[args.graph]()
    E = LayoutEnergy(G.n, G.edges)
    if args.seed == "auto":
        seed, skipped = pick_seed(G, E)
        print(f"showcase seed = {seed}  (seeds skipped first: {skipped})")
    else:
        seed = int(args.seed)
    z0 = random_layout(G.n, seed)
    print(f"graph {G.name}: n={G.n} nodes, m={G.m} edges, start crossings = {count_crossings(z0, G.edges)}")

    sd = steepest_descent(E, z0)
    cg = conjugate_gradient(E, z0)
    lm = levenberg_marquardt(E, z0)

    tag = f"demo_{args.graph}"
    viz.snapshots_figure(cg, G, out / f"{tag}_cg_snapshots.png",
                         title=f"Conjugate gradient (Fletcher-Reeves) untangling {G.name}")
    viz.final_layouts_figure([sd, cg, lm], G, out / f"{tag}_final_layouts.png", z0=z0)
    viz.convergence_figure([sd, cg, lm], out / f"{tag}_convergence.png")
    frames = viz.make_gif(cg, G, out / f"{tag}_cg.gif")
    viz.before_after_figure(z0, cg.z, G, out / f"{tag}_before_after.png")
    viz.potentials_figure(out / "energy_terms.png", L=E.L, k=E.k, c=E.c, R=E.R, eps=E.eps)

    print(f"\n{'method':<26}{'iters':>7}{'E_final':>10}{'time s':>9}{'f/g/J evals':>16}"
          f"{'crossings':>11}{'edge CV':>9}{'rate q':>8}  stop")
    for r in (sd, cg, lm):
        print(f"{r.name:<26}{r.n_iter:>7}{r.energy[-1]:>10.3f}{r.seconds[-1]:>9.2f}"
              f"{f'{r.n_f[-1]}/{r.n_g[-1]}/{r.n_J[-1]}':>16}"
              f"{count_crossings(r.z, G.edges):>11}{edge_length_cv(r.z, G.edges):>9.3f}"
              f"{tail_rate(r.gnorm):>8.3f}  {r.stop_reason}")
    import json
    summary = {"graph": G.name, "n": G.n, "m": G.m, "seed": seed,
               "start_crossings": count_crossings(z0, G.edges),
               "methods": {r.name: {"iters": r.n_iter, "seconds": round(r.seconds[-1], 3),
                                    "energy": round(r.energy[-1], 4),
                                    "crossings": count_crossings(r.z, G.edges),
                                    "rate_q": round(tail_rate(r.gnorm), 3)} for r in (sd, cg, lm)}}
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nGIF frames: {frames}.  Files written to {out}/")
