# Force-Directed Graph Layout as Unconstrained Optimization

Mini project for the Optimization course (Module 1 line search, Module 2 gradient-based
multivariable optimization).

Nodes repel like charged particles, edges pull like springs. The "nicest" 2D drawing is the
minimiser of a total energy over all 2n coordinates. We minimise it with three methods written
from scratch in NumPy and compare them:

| Method | Direction | Step |
|---|---|---|
| Steepest descent | `-grad E` | bounding phase + golden-section line search |
| Fletcher-Reeves CG | `-g' + beta d`, `beta = |g'|^2 / |g|^2` | same line search, restart every 2n steps |
| Levenberg-Marquardt | `(J^T J + lambda diag(J^T J)) delta = -J^T r` | none, lambda adapts to gain ratio |

## Energy

```
E(z) = sum_edges (k/2)(d - L)^2  +  sum_pairs (c L^2 / 2) ln(1 + R^2/(d^2 + eps^2))  +  (g/2)|z|^2
```
Defaults: L=1, k=1, c=0.3, R=8, eps=0.15, g=0.02. E is a sum of squares (residual form), so
`grad E = J^T r` and LM applies. Gradient and Jacobian are analytic and tested against finite
differences.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                         # 33 tests
python scripts/run_demo.py --graph net      # figures + animated GIF in results/
python scripts/benchmark.py --seeds 20      # ~6 min, 240 runs, writes results/benchmark_*.csv
npm i pptxgenjs && node scripts/make_deck.js  # docs/force_directed_layout.pptx from results/
```

## Results (20 random starts x 4 graphs x 3 methods = 240 runs, all converged)

* Conjugate gradient needs a median of 9.1x fewer iterations and 9.3x less time than steepest
  descent (paired over the 80 start/graph combinations).
* Levenberg-Marquardt needs about 2.4x CG's iterations (median 250 vs 106) at similar wall time.
* Layout quality is the same for all methods: crossing-free on planar graphs in SD 87%, CG 90%,
  LM 83% of runs (differences are within noise at 20 starts). SD and CG end in the same minimum
  in 91% of paired runs.
* The energy is non-convex: the minimum reached depends on the start. Use several random starts.

Full tables: `results/benchmark_summary.csv`, per-run data: `results/benchmark_runs.csv`.

## Design notes (what went wrong first)

1. The first model used 1/d^2 repulsion on a Delaunay triangulation and gave 0% crossing-free
   layouts. Starting LM from the *true* planar drawing still moved to a folded layout with lower
   energy, so the energy, not the optimiser, was wrong: short-range repulsion makes folds cheap and
   a random triangulation cannot have uniform edge lengths. Fix: long-range log repulsion
   (Fruchterman-Reingold-like) and a sparse planar Gabriel network. `c` and `R` were tuned on
   separate seeds (100-111); benchmark seeds are 0-19.
2. LM was first capped at 300 iterations (vs 3000 for SD/CG) and looked like it failed. It only
   needed more iterations. All methods now share the same budget.
3. A test wrongly assumed a unique minimum; the energy is non-convex, so it was rewritten.

The showcase demo uses seed 0, not a hand-picked seed.

## Layout

```
forcelayout/  graphs.py energy.py linesearch.py optimizers.py metrics.py viz.py
scripts/      run_demo.py benchmark.py make_deck.js
tests/        energy, line search, optimizers, metrics, graphs
results/      figures, GIF, CSVs      docs/  slide deck
```
