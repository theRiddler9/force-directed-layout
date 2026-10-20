"""Plots and animation for the layout project."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .metrics import count_crossings, crossing_edge_mask

# colour-blind-safe palette (Okabe-Ito)
COLORS = {
    "Steepest descent": "#E69F00",
    "Conjugate gradient (FR)": "#0072B2",
    "Levenberg-Marquardt": "#009E73",
}
NODE = "#2B3A55"
EDGE = "#9AA3B2"
CROSS = "#D55E00"


def draw_layout(ax, z, edges, node_size=26, lims=None, show_crossings=True):
    X = np.asarray(z).reshape(-1, 2)
    mask = crossing_edge_mask(z, edges) if show_crossings else np.zeros(len(edges), bool)
    for (a, b), bad in zip(edges, mask):
        ax.plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]],
                color=CROSS if bad else EDGE, lw=1.3 if bad else 1.0,
                alpha=0.95 if bad else 0.8, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], s=node_size, color=NODE, zorder=2, edgecolor="white", linewidth=0.5)
    ax.set_aspect("equal")
    if lims is None:
        lo, hi = X.min(0), X.max(0)
        pad = 0.08 * max(hi - lo) + 0.3
        c, h = (lo + hi) / 2, max(hi - lo) / 2 + pad
        lims = (c[0] - h, c[0] + h, c[1] - h, c[1] + h)
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#C9CED8")


def snapshots_figure(res, graph, path, fractions=(0.0, 0.03, 0.15, 1.0), title=None):
    """Row of layout snapshots taken at fractions of the run's iterations."""
    idx = sorted({int(round(f * res.n_iter)) for f in fractions})
    fig, axes = plt.subplots(1, len(idx), figsize=(3.3 * len(idx), 3.7))
    for ax, k in zip(np.atleast_1d(axes), idx):
        z = res.snapshots[k]
        draw_layout(ax, z, graph.edges)
        ax.set_title(
            f"iteration {k}\nE = {res.energy[k]:.1f}   crossings = {count_crossings(z, graph.edges)}",
            fontsize=10, color="#222")
    if title:
        fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return idx


def final_layouts_figure(results, graph, path, z0=None):
    """Start state followed by each method's final layout."""
    panels = ([("Random start", z0, None)] if z0 is not None else []) + [
        (r.name, r.z, r) for r in results]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.8))
    for ax, (name, z, r) in zip(axes, panels):
        draw_layout(ax, z, graph.edges)
        sub = f"{count_crossings(z, graph.edges)} crossings"
        if r is not None:
            sub = f"E = {r.energy[-1]:.2f}, {r.n_iter} iters, {sub}"
        ax.set_title(f"{name}\n{sub}", fontsize=9.5, color="#222")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def convergence_figure(results, path):
    """Energy vs iteration, gradient norm vs iteration, gradient norm vs time."""
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for r in results:
        c = COLORS.get(r.name, "#444")
        it = np.arange(len(r.energy))
        ax[0].plot(it + 1, r.energy, color=c, lw=2, label=r.name)
        ax[1].plot(it + 1, r.gnorm, color=c, lw=2, label=r.name)
        ax[2].plot(np.maximum(r.seconds, 1e-4), r.gnorm, color=c, lw=2, label=r.name)
    ax[0].set(xscale="log", yscale="log", xlabel="iteration (log)", ylabel="energy E",
              title="Energy")
    ax[1].set(xscale="log", yscale="log", xlabel="iteration (log)", ylabel="||grad E||",
              title="Gradient norm vs iterations")
    ax[2].set(xscale="log", yscale="log", xlabel="wall-clock seconds (log)", ylabel="||grad E||",
              title="Gradient norm vs time")
    for a in ax:
        a.grid(alpha=0.25, which="both")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_gif(res, graph, path, n_frames=70, hold=12, fps=14):
    """Animated untangling: layout (left) + energy curve with moving marker (right)."""
    n = res.n_iter
    t = np.linspace(0, 1, n_frames)
    idx = sorted({int(round(n * s**2.2)) for s in t})          # dense early, sparse late
    snaps = np.array([res.snapshots[k] for k in idx]).reshape(len(idx), -1, 2)
    lo, hi = snaps.min((0, 1)), snaps.max((0, 1))
    c, h = (lo + hi) / 2, max(hi - lo) / 2 * 1.08 + 0.3
    lims = (c[0] - h, c[0] + h, c[1] - h, c[1] + h)

    frames = []
    for k in idx:
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.05, 1]})
        z = res.snapshots[k]
        draw_layout(a0, z, graph.edges, lims=lims, node_size=30)
        a0.set_title(f"{res.name}   iteration {k}", fontsize=11)
        a0.text(0.02, 0.02, f"crossings: {count_crossings(z, graph.edges)}", transform=a0.transAxes,
                fontsize=10, color=CROSS, weight="bold")
        a1.plot(np.arange(len(res.energy)) + 1, res.energy, color=COLORS.get(res.name, "#444"), lw=2)
        a1.scatter([k + 1], [res.energy[k]], s=60, color=CROSS, zorder=3)
        a1.set(xscale="log", yscale="log", xlabel="iteration (log)", ylabel="total energy E(X, Y)")
        a1.grid(alpha=0.25, which="both")
        for s in ("top", "right"):
            a1.spines[s].set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames += [frames[-1]] * hold
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return len(frames)
