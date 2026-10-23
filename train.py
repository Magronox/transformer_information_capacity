"""Full-batch training of a transformer to memorize a synthetic fact dataset."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from gpt2.modeling.transformer import Transformer

import data


@dataclass(frozen=True)
class TrainConfig:
    family: str            # "uniform" | "symmetric" | "two_hop" | "powerlaw"
    num_entities: int
    num_relations: int
    dim: int
    layers: int
    seed: int
    lr: float
    weight_decay: float
    warmup_steps: int
    lr_decay_type: str     # "linear" | "cosine" | "wsd" | "flat"
    min_lr: float
    decay_end_ratio: float
    decay_start_ratio: float = 0.0
    max_batches: int = 40_000
    stability_window: int = 50
    clip_norm: float = 1.0
    micro_batch_size: int | None = None   # None = full batch (powerlaw sizes its own)
    alpha: float = 1.0                    # powerlaw only
    min_tail_count: int = 1               # powerlaw only
    heads: int = 1


@dataclass(frozen=True)
class TrainResult:
    converged: bool              # 100% accuracy held for stability_window consecutive steps
    steps_to_100: int | None     # first step (warmup included) with 100% accuracy
    best_acc: float


def gpt2(cfg: TrainConfig, vocab_size: int) -> torch.nn.Module:
    return Transformer(layers=cfg.layers, pad_idx=-1, words=vocab_size, seq_len=3,
                       heads=cfg.heads, dims=cfg.dim, bidirectional=False,
                       dropout=0, positional_embedding=False, kaiming_init=True)


def train(cfg: TrainConfig,
          build_model: Callable[[TrainConfig, int], torch.nn.Module] = gpt2) -> TrainResult:
    _seed_everything(cfg.seed)
    facts = _make_facts(cfg)
    vocab_size = cfg.num_entities + cfg.num_relations

    if cfg.family == "powerlaw":
        x_train, y_train = _powerlaw_batch(facts, cfg)
        micro_batch = _powerlaw_micro_batch(len(x_train), len(facts), vocab_size, cfg)
        eval_set = tuple(t.cuda() for t in _split(torch.tensor(facts, dtype=torch.long)))
    else:
        x_train, y_train = _split(torch.tensor(facts, dtype=torch.long))
        micro_batch = cfg.micro_batch_size
        eval_set = None

    model = build_model(cfg, vocab_size)
    return _fit(model.cuda(), x_train.cuda(), y_train.cuda(), micro_batch, cfg, eval_set)


def _fit(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
         micro_batch: int | None, cfg: TrainConfig,
         eval_set: tuple[torch.Tensor, torch.Tensor] | None) -> TrainResult:
    torch.backends.cuda.matmul.allow_tf32 = True
    n = len(x)
    chunk = n if micro_batch is None or micro_batch >= n else micro_batch
    slices = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay, fused=True)
    model.train()

    steps_to_100, best_acc, streak = None, 0.0, 0
    for step in range(1, cfg.warmup_steps + cfg.max_batches + 2):
        optimizer.zero_grad(set_to_none=True)
        correct = torch.zeros((), dtype=torch.long, device=x.device)
        for s, e in slices:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x[s:e])[:, -1, :]
                loss = F.cross_entropy(logits, y[s:e], reduction="sum") / n
            loss.backward()
            if eval_set is None:
                correct += (logits.argmax(dim=1) == y[s:e]).sum()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        optimizer.step()

        # The new rate takes effect from the next step, so the first step runs at cfg.lr.
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(step, cfg)

        acc = (correct.item() / n if eval_set is None
               else _accuracy(model, *eval_set))
        best_acc = max(best_acc, acc)
        if acc >= 1.0:
            if steps_to_100 is None:
                steps_to_100 = step
            streak += 1
            if streak >= cfg.stability_window:
                return TrainResult(True, steps_to_100, best_acc)
        else:
            streak = 0
    return TrainResult(False, steps_to_100, best_acc)


@torch.no_grad()
def _accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predicted = model(x)[:, -1, :].max(dim=1).indices
    return (predicted == y).sum().item() / len(y)


def _learning_rate(step: int, cfg: TrainConfig) -> float:
    base, low = cfg.lr, cfg.min_lr
    if step < cfg.warmup_steps:
        return low + (base - low) * (step / cfg.warmup_steps)
    t = step - cfg.warmup_steps
    decay_end = int(cfg.max_batches * cfg.decay_end_ratio)
    if t >= decay_end:
        return low
    if cfg.lr_decay_type == "linear":
        return base - (base - low) * (t / decay_end)
    if cfg.lr_decay_type == "cosine":
        return low + (base - low) * 0.5 * (1 + np.cos(np.pi * (t / decay_end)))
    if cfg.lr_decay_type == "wsd":
        decay_start = int(cfg.max_batches * cfg.decay_start_ratio)
        if t < decay_start or decay_start >= decay_end:
            return base
        p = (t - decay_start) / (decay_end - decay_start)
        return base * low / (p * base + (1 - p) * low)
    return base


def _make_facts(cfg: TrainConfig) -> list[data.Fact]:
    if cfg.family in ("uniform", "powerlaw"):
        return data.uniform(cfg.num_entities, cfg.num_relations)
    if cfg.family == "symmetric":
        return data.symmetric(cfg.num_entities, cfg.num_relations)
    if cfg.family == "two_hop":
        return data.two_hop(cfg.num_entities)
    raise ValueError(f"unknown family: {cfg.family}")


def _powerlaw_batch(facts: list[data.Fact],
                    cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    counts = data.powerlaw_counts(len(facts), cfg.alpha, cfg.min_tail_count)
    order = [i for i, c in enumerate(counts) for _ in range(c)]
    random.Random(cfg.seed).shuffle(order)
    return _split(torch.tensor(facts, dtype=torch.long)[torch.tensor(order)])


def _powerlaw_micro_batch(batch: int, num_facts: int, vocab_size: int,
                          cfg: TrainConfig) -> int:
    gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    model_bytes = 16 * (2 * vocab_size * cfg.dim + cfg.layers * 12 * cfg.dim ** 2)
    data_bytes = 24 * (batch + num_facts)
    free = max(int(gpu_bytes * 0.85) - model_bytes - data_bytes, 1024 ** 3)
    return min(max(1000, free // (12 * vocab_size)), batch)


def _split(facts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return facts[:, :2], facts[:, 2]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
