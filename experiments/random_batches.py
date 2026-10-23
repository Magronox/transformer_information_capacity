"""Training with random batch sampling, the protocol used before full-batch training.

Every step draws `batch_size` facts uniformly with replacement. Adam runs at a constant
learning rate with weight decay 0.1, in float32 and without gradient clipping. After every
step the model is evaluated on all facts.

The appendix figure trains once on Uniform with 500 entities and 3 relations (N = 1,500),
d = 95 and one layer, for 100,000 steps, and plots the loss over all facts.

Run from the repository root:
  python -m experiments.random_batches train --lr LR --batch-size B --seed S --out loss.json
  python -m experiments.random_batches plot loss.json --out fig5_loss_random_batches.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gpt2.modeling.transformer import Transformer

import data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 0.1


@dataclass(frozen=True)
class SampledConfig:
    family: str
    num_entities: int
    num_relations: int
    dim: int
    layers: int
    seed: int
    lr: float
    batch_size: int
    max_batches: int
    stop_at_100: bool = True
    patience: int | None = None


@dataclass(frozen=True)
class SampledResult:
    converged: bool
    steps_to_100: int | None
    best_acc: float
    losses: list[float]


def make_facts(family: str, num_entities: int, num_relations: int) -> list[data.Fact]:
    if family == "uniform":
        return data.uniform(num_entities, num_relations)
    if family == "symmetric":
        return data.symmetric(num_entities, num_relations)
    if family == "two_hop":
        return data.two_hop(num_entities)
    raise ValueError(f"unknown family: {family}")


def train_sampled(cfg: SampledConfig) -> SampledResult:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    facts = torch.tensor(make_facts(cfg.family, cfg.num_entities, cfg.num_relations),
                         dtype=torch.long, device=DEVICE)
    x, y = facts[:, :2], facts[:, 2]
    model = Transformer(layers=cfg.layers, pad_idx=-1,
                        words=cfg.num_entities + cfg.num_relations, seq_len=3, heads=1,
                        dims=cfg.dim, bidirectional=False, dropout=0,
                        positional_embedding=False, kaiming_init=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=WEIGHT_DECAY)
    model.train()

    losses: list[float] = []
    steps_to_100, best_acc, stalled = None, 0.0, 0
    for step in range(1, cfg.max_batches + 1):
        batch = torch.randint(len(facts), (cfg.batch_size,), device=DEVICE)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x[batch])[:, -1, :], y[batch]).backward()
        optimizer.step()

        acc, loss = _evaluate(model, x, y)
        losses.append(loss)
        if acc >= 1.0:
            steps_to_100 = steps_to_100 or step
            if cfg.stop_at_100:
                break
        if acc >= best_acc:
            best_acc, stalled = acc, 0
        else:
            stalled += 1
            if cfg.patience is not None and stalled >= cfg.patience:
                break
    return SampledResult(steps_to_100 is not None, steps_to_100, best_acc, losses)


@torch.no_grad()
def _evaluate(model, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    logits = model(x)[:, -1, :]
    accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    return accuracy, F.cross_entropy(logits, y).item()


def plot(trace_path: Path, out_path: Path) -> None:
    """Draws the loss curve from a {"iter": [...], "loss": [...]} JSON file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linestyle": "--", "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    trace = json.loads(trace_path.read_text())
    loss = np.asarray(trace["loss"], dtype=float)
    exponent = math.floor(math.log10(loss.max()))
    label = "Training loss" + (rf" ($\times10^{{{exponent}}}$)" if exponent else "")

    fig, ax = plt.subplots(figsize=(2.6, 1.75))
    ax.plot(np.asarray(trace["iter"]) / 1000, loss / 10.0 ** exponent, color="#3b7dd8", lw=1.4)
    ax.set_xlabel("Iteration (thousands)", labelpad=1)
    ax.set_ylabel(label, labelpad=2)
    ax.margins(x=0.02)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("train", help="train once and save the loss after every step as JSON")
    run.add_argument("--family", choices=("uniform", "symmetric", "two_hop"), default="uniform")
    run.add_argument("--entities", type=int, default=500)
    run.add_argument("--relations", type=int, default=3)
    run.add_argument("--dim", type=int, default=95)
    run.add_argument("--layers", type=int, default=1)
    run.add_argument("--lr", type=float, required=True)
    run.add_argument("--batch-size", type=int, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--max-batches", type=int, default=100_000)
    run.add_argument("--out", type=Path, required=True)

    draw = sub.add_parser("plot", help="draw the figure from a loss JSON")
    draw.add_argument("trace", type=Path)
    draw.add_argument("--out", type=Path, default=Path("fig5_loss_random_batches.pdf"))

    args = parser.parse_args()
    if args.command == "plot":
        plot(args.trace, args.out)
        return
    result = train_sampled(SampledConfig(
        family=args.family, num_entities=args.entities, num_relations=args.relations,
        dim=args.dim, layers=args.layers, seed=args.seed, lr=args.lr,
        batch_size=args.batch_size, max_batches=args.max_batches, stop_at_100=False))
    args.out.write_text(json.dumps({"iter": list(range(1, len(result.losses) + 1)),
                                    "loss": result.losses}))


if __name__ == "__main__":
    main()
