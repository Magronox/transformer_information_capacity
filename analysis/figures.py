"""Figures 1 and 2 of the paper from the CSVs written by export_results.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MarkerPath

from analysis.fits import Fit, column, depth_data, depth_law, read_csv, seeds, width_law

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})
WIDTH, HEIGHT = 2.6, 2.15
BLUE, LIGHT_BLUE, GREY = "#3b7dd8", "#8fb8ec", "#9e9e9e"
LEGEND = dict(frameon=False, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.25, handlelength=1.6)
TRIANGLE = MarkerPath([(0, 4 / 3), (-1.15, -2 / 3), (1.15, -2 / 3), (0, 4 / 3)], closed=True)
FAMILIES = {
    "uniform": ("Uniform", BLUE, "o", 4.6, 5),
    "powerlaw": (r"PowerLaw ($\alpha=1$)", "#d4a017", TRIANGLE, 5.2, 6),
    "symmetric": ("Symmetric", "#e91e63", "o", 4.6, 4),
    "two_hop": ("2Hop", "#43a047", "o", 4.6, 4),
}


def family_means(results: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = read_csv("width_min_dim.csv", results)
    out = {}
    for family in dict.fromkeys(r["family"] for r in rows):
        sub = [r for r in rows if r["family"] == family]
        out[family] = column(sub, "facts"), seeds(sub, "min_dim").mean(1)
    return out


def min_dim_figure(means: dict[str, tuple[np.ndarray, np.ndarray]], law: Fit) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    grid = np.logspace(np.log10(38), np.log10(60000), 300)
    ax.plot(grid, law.form.f(grid, *law.params), color=LIGHT_BLUE, lw=1.6, label="fit to Uniform", zorder=1)
    for family, (label, color, marker, size, z) in FAMILIES.items():
        if family in means:
            ax.plot(*means[family], marker=marker, ls="", mew=0, ms=size, color=color, label=label, zorder=z)
    ax.set_xscale("log")
    ax.set_xlabel("Dataset size $N$", labelpad=1)
    ax.set_ylabel(r"Minimum dimension $d_{\min}$", labelpad=2)
    ax.set_xticks([100, 1000, 10000], ["100", "1,000", "10,000"])
    ax.set_yticks([5, 10, 15, 20])
    ax.set_ylim(2, 20.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0, *range(2, len(handles))]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", **LEGEND)
    return fig


def depth_figure(layers: np.ndarray, sizes: np.ndarray, log_law: Fit, power_law: Fit) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    grid = np.linspace(1, 16.4, 300)
    ax.plot(grid, log_law.form.f(grid, *log_law.params), color=LIGHT_BLUE, lw=1.6, label="logarithmic fit", zorder=1)
    ax.plot(grid, power_law.form.f(grid, *power_law.params), color=LIGHT_BLUE, lw=1.4, ls=(0, (4, 2)),
            label="power-law fit", zorder=1)
    ax.plot(np.repeat(layers, sizes.shape[1]), sizes.ravel(), "o", color=GREY, ms=3, mew=0, label="each seed", zorder=3)
    ax.plot(layers, sizes.mean(1), "o", color=BLUE, ms=4.6, mew=0, label="mean of 3 seeds", zorder=5)
    ax.set_xlabel("Number of layers $L$", labelpad=1)
    ax.set_ylabel(r"Max. dataset size $N_{\max}$", labelpad=2)
    ax.set_xticks(layers)
    ax.set_yticks(range(0, 6000, 1000), ["0", "1,000", "2,000", "3,000", "4,000", "5,000"])
    ax.set_ylim(0, 5600)
    ax.set_xlim(0.3, 16.7)
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 2, 0, 1]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", **LEGEND)
    return fig


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="directory with the CSVs written by export_results.py")
    parser.add_argument("--out", type=Path, default=Path("."), help="output directory for the PDFs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    save(min_dim_figure(family_means(args.results), width_law(args.results)), args.out / "fig1_min_dim.pdf")
    _, layers, sizes = depth_data(args.results)
    fits = depth_law(args.results)
    save(depth_figure(layers, sizes, fits["Logarithmic"], fits["Power-law"]), args.out / "fig2_depth.pdf")


if __name__ == "__main__":
    main()
