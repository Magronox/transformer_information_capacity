"""Width experiment with a limited hyperparameter search: learning rate and batch size only.

Training uses random batch sampling (experiments.random_batches): constant learning rate,
weight decay 0.1, a budget of 40,960,000 sampled facts (20,000 batches of 2,048), and a run
succeeds at the first step with 100% accuracy. For every family, dataset size, grid point
(learning rate, batch size) and seed, the smallest successful width is found by halving from
d = 1024 (doubling if 1024 fails) and then bisecting. Per dataset size, the grid point with
the smallest mean width over seeds is kept, and the figure fits d = a N^b + c to those means.

Run from the repository root:
  python -m experiments.limited_search run --family uniform --out limited_search.csv
  python -m experiments.limited_search plot limited_search.csv --out limited_search.pdf
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit

from experiments.random_batches import SampledConfig, train_sampled

FAMILIES = {"uniform": "Uniform", "two_hop": "2Hop", "symmetric": "Symmetric"}
ENTITIES = (10, 100, 500, 1000, 2000, 5000)
RELATIONS = 3
SEEDS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
LEARNING_RATES = (1e-4, 5e-5, 1e-5, 5e-6)
BATCH_SIZES = (2048, 4096, 8192)

SAMPLE_BUDGET = 20_000 * 2048
PATIENCE = 10_000
START_DIM = 1024
DIM_CAP = 4096
COLUMNS = ("family", "num_entities", "num_relations", "lr", "batch_size", "seed", "min_dim")


def smallest_dim(succeeds: Callable[[int], bool]) -> int | None:
    """Halve from START_DIM while training succeeds (or double up to DIM_CAP until it
    does), then bisect between the last failure and the first success."""
    fail, ok = 0, START_DIM
    if succeeds(ok):
        while ok > 1 and succeeds(ok // 2):
            ok //= 2
        fail = ok // 2
    else:
        fail, ok = ok, 2 * ok
        while not succeeds(ok):
            if ok >= DIM_CAP:
                return None
            fail, ok = ok, 2 * ok
    while ok - fail > 1:
        mid = (fail + ok) // 2
        if succeeds(mid):
            ok = mid
        else:
            fail = mid
    return ok


def min_dim(family: str, entities: int, relations: int, lr: float, batch_size: int,
            seed: int) -> int | None:
    def succeeds(dim: int) -> bool:
        return train_sampled(SampledConfig(
            family=family, num_entities=entities, num_relations=relations, dim=dim,
            layers=1, seed=seed, lr=lr, batch_size=batch_size,
            max_batches=SAMPLE_BUDGET // batch_size, patience=PATIENCE)).converged
    return smallest_dim(succeeds)


def run(family: str, entities: list[int], lrs: list[float], batch_sizes: list[int],
        seeds: list[int], out: Path) -> None:
    """Appends one CSV row per (dataset, grid point, seed); rows already in `out` are skipped."""
    done = {_key(r) for r in read_rows(out)} if out.exists() else set()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if f.tell() == 0:
            writer.writeheader()
        for e in entities:
            for lr in lrs:
                for bs in batch_sizes:
                    for seed in seeds:
                        row = {"family": family, "num_entities": e, "num_relations": RELATIONS,
                               "lr": lr, "batch_size": bs, "seed": seed}
                        if _key(row) in done:
                            continue
                        d = min_dim(family, e, RELATIONS, lr, bs, seed)
                        writer.writerow({**row, "min_dim": "" if d is None else d})
                        f.flush()


def _key(row: dict) -> tuple:
    return tuple(str(row[c]) for c in COLUMNS[:-1])


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def best_means(rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per family: dataset sizes N and the mean width of the best grid point at each N.

    A grid point counts only if every seed found a width.
    """
    widths = defaultdict(list)
    for r in rows:
        widths[(r["family"], int(r["num_entities"]) * int(r["num_relations"]),
                r["lr"], r["batch_size"])].append(r["min_dim"])
    best: dict[tuple[str, int], float] = {}
    for (family, n, _, _), ds in widths.items():
        if all(ds):
            mean = float(np.mean([float(d) for d in ds]))
            best[(family, n)] = min(mean, best.get((family, n), np.inf))
    out = {}
    for family in FAMILIES:
        sizes = sorted(n for f, n in best if f == family)
        if sizes:
            out[family] = (np.array(sizes, float), np.array([best[(family, n)] for n in sizes]))
    return out


def power_law(n, a, b, c):
    return a * np.power(n, b) + c


def fit(sizes: np.ndarray, dims: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares fit of d = a N^b + c (offset unconstrained) and its R^2."""
    params, _ = curve_fit(power_law, sizes, dims, p0=[1, 0.5, 0])
    residual = dims - power_law(sizes, *params)
    r2 = 1 - (residual ** 2).sum() / ((dims - dims.mean()) ** 2).sum()
    return params, float(r2)


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for i, (family, (sizes, dims)) in enumerate(best_means(rows).items()):
        color = f"C{i}"
        (a, b, c), r2 = fit(sizes, dims)
        grid = np.linspace(sizes.min(), sizes.max(), 200)
        ax.plot(sizes, dims, "o", color=color)
        ax.plot(grid, power_law(grid, a, b, c), "--", color=color, lw=2,
                label=f"{FAMILIES[family]} fit ($R^2$={r2:.3f})")
        ax.text(0.03, 0.95 - 0.1 * i, f"{FAMILIES[family]}: d = {a:.3g}·N^({b:.3f}) + {c:.3g}",
                transform=ax.transAxes, va="top", fontsize=9, color=color,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor=color))
    ax.set_xlabel("Dataset size", fontsize=12)
    ax.set_ylabel("Minimum Transformer dimension", fontsize=12)
    ax.set_title("Minimum Transformer Dimension vs Dataset Size", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("run", help="run the search and append results to a CSV")
    search.add_argument("--family", choices=FAMILIES, required=True)
    search.add_argument("--entities", nargs="+", type=int, default=list(ENTITIES))
    search.add_argument("--lrs", nargs="+", type=float, default=list(LEARNING_RATES))
    search.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCH_SIZES))
    search.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    search.add_argument("--out", type=Path, required=True)

    draw = sub.add_parser("plot", help="draw the figure from one or more result CSVs")
    draw.add_argument("results", nargs="+", type=Path)
    draw.add_argument("--out", type=Path, default=Path("limited_search.pdf"))

    args = parser.parse_args()
    if args.command == "run":
        run(args.family, args.entities, args.lrs, args.batch_sizes, args.seeds, args.out)
    else:
        plot([r for path in args.results for r in read_rows(path)], args.out)


if __name__ == "__main__":
    main()
