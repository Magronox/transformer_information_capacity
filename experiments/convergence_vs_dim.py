"""Training steps to 100% accuracy vs. width near the minimum width.

For each dataset and schedule, the best trial of the min-d study (search_min_dim.py) gives the
hyperparameters and d_min, the largest per-seed minimum width of that trial. Every width
d_min - 2 ... d_min + 6 is trained for seeds 10, 20, 30 with a 100,000-step budget; the decay
end keeps its step from the 40,000-step search.

Run from the repository root:
  python -m experiments.convergence_vs_dim run --datasets 125,3 250,3 --db-dir studies --out steps.csv
  python -m experiments.convergence_vs_dim plot steps.csv --out convergence_vs_dim.pdf
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna

import search_min_dim
from train import TrainConfig, train

SEEDS = (10, 20, 30)
DIM_OFFSETS = (-2, -1, 0, 1, 2, 3, 4, 5, 6)
MAX_BATCHES = 100_000
COLUMNS = ("num_entities", "num_relations", "decay", "dim", "seed", "converged", "steps_to_100")


def best_trial(setup: search_min_dim.Setup, decay: str, db_dir: str) -> tuple[search_min_dim.Hparams, int]:
    name = search_min_dim.study_name(setup, decay)
    study = optuna.load_study(study_name=name,
                              storage=f"sqlite:///{os.path.join(db_dir, name + '.db')}")
    trial = study.best_trial
    hp = search_min_dim.sample_hparams(optuna.trial.FixedTrial(trial.params), decay)
    dims = [d for d in trial.user_attrs["per_seed_min_dim"].values() if d is not None]
    return hp, max(dims)


def run(datasets: list[tuple[int, int]], decays: list[str], db_dir: str, out: Path) -> None:
    """Appends one CSV row per run; rows already in `out` are skipped."""
    done = {tuple(r[c] for c in COLUMNS[:5]) for r in read_rows(out)} if out.exists() else set()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if f.tell() == 0:
            writer.writeheader()
        for entities, relations in datasets:
            setup = search_min_dim.Setup("uniform", entities, relations, 1, 1.0, 1)
            for decay in decays:
                hp, d_min = best_trial(setup, decay, db_dir)
                for dim in (d_min + k for k in DIM_OFFSETS if d_min + k > 0):
                    for seed in SEEDS:
                        key = (entities, relations, decay, dim, seed)
                        if tuple(map(str, key)) in done:
                            continue
                        result = train(TrainConfig(
                            family="uniform", num_entities=entities, num_relations=relations,
                            dim=dim, layers=1, seed=seed, lr=hp.lr,
                            weight_decay=hp.weight_decay,
                            warmup_steps=search_min_dim.WARMUP_STEPS_PER_LAYER,
                            lr_decay_type=decay, min_lr=hp.min_lr,
                            decay_end_ratio=hp.decay_end_ratio * search_min_dim.MAX_BATCHES
                            / MAX_BATCHES,
                            max_batches=MAX_BATCHES,
                            stability_window=search_min_dim.STABILITY_WINDOW))
                        writer.writerow(dict(zip(COLUMNS, key + (
                            int(result.converged), result.steps_to_100 or ""))))
                        f.flush()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot(rows: list[dict], out: Path) -> None:
    """Mean over converged seeds, with a min-max band, per width and dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["converged"] == "1":
            steps[(r["decay"], int(r["num_entities"]))][int(r["dim"])].append(
                float(r["steps_to_100"]))
    entities = sorted({e for _, e in steps})
    colors = {e: colorsys.hls_to_rgb(i / len(entities), 0.45, 0.75)
              for i, e in enumerate(entities)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, decay in zip(axes, ("cosine", "linear")):
        for e in entities:
            by_dim = steps.get((decay, e))
            if not by_dim:
                continue
            dims = sorted(by_dim)
            values = [np.array(by_dim[d]) for d in dims]
            ax.plot(dims, [v.mean() for v in values], "o-", lw=2, color=colors[e], label=f"E={e}")
            ax.fill_between(dims, [v.min() for v in values], [v.max() for v in values],
                            color=colors[e], alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(f"{decay.capitalize()} Decay", fontsize=14)
        ax.set_xlabel("Transformer Dimension", fontsize=13)
        ax.set_ylabel("Batches until 100% Accuracy", fontsize=13)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=11)
    fig.suptitle(f"Convergence Speed vs Dimension ({len(SEEDS)} seeds, band = min/max)",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _parse_dataset(text: str) -> tuple[int, int]:
    entities, relations = text.split(",")
    return int(entities), int(relations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("run", help="train around d_min and append results to a CSV")
    sweep.add_argument("--datasets", nargs="+", type=_parse_dataset, required=True,
                       metavar="E,R")
    sweep.add_argument("--lr-decay-types", nargs="+", choices=search_min_dim.DECAY_TYPES,
                       default=list(search_min_dim.DECAY_TYPES))
    sweep.add_argument("--db-dir", default="studies")
    sweep.add_argument("--out", type=Path, required=True)

    draw = sub.add_parser("plot", help="draw the figure from a result CSV")
    draw.add_argument("results", type=Path)
    draw.add_argument("--out", type=Path, default=Path("convergence_vs_dim.pdf"))

    args = parser.parse_args()
    if args.command == "run":
        run(args.datasets, args.lr_decay_types, args.db_dir, args.out)
    else:
        plot(read_rows(args.results), args.out)


if __name__ == "__main__":
    main()
