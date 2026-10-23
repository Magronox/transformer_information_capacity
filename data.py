"""Synthetic fact datasets: (subject, relation, object) token-id triples."""
from __future__ import annotations

import random

import numpy as np

Fact = tuple[int, int, int]


def uniform(num_entities: int, num_relations: int) -> list[Fact]:
    """Entities 0..n-1, relations n..n+m-1, one uniformly drawn object per (subject, relation)."""
    entities = list(range(num_entities))
    # random.sample, not random.choice: it consumes a different random stream.
    return [(s, num_entities + r, random.sample(entities, 1)[0])
            for s in entities for r in range(num_relations)]


def symmetric(num_entities: int, num_relations: int) -> list[Fact]:
    """Entities 0..n-1, relations n..n+m-1, each relation a random involution."""
    facts = []
    for r in range(1, num_relations + 1):
        f = _random_involution(num_entities)
        facts.extend((a - 1, num_entities + r - 1, f[a] - 1)
                     for a in range(1, num_entities + 1))
    return facts


def two_hop(num_entities: int) -> list[Fact]:
    """Entities 0..n-1; relations n, n+1 are random functions and n+2 is their composition."""
    f1 = [random.randrange(num_entities) for _ in range(num_entities)]
    f2 = [random.randrange(num_entities) for _ in range(num_entities)]
    facts = []
    for a in range(num_entities):
        n = num_entities
        facts += [(a, n, f1[a]), (a, n + 1, f2[a]), (a, n + 2, f2[f1[a]])]
    return facts


def _random_involution(n: int) -> dict[int, int]:
    items = list(range(1, n + 1))
    random.shuffle(items)
    f = {}
    while len(items) > 1:
        a, b = items.pop(), items.pop()
        f[a], f[b] = b, a
    if items:
        a = items.pop()
        f[a] = a
    return f


def powerlaw_counts(num_facts: int, alpha: float,
                    min_tail_count: int = 1) -> list[int]:
    """Copies of each fact per batch, c_i ∝ i^-alpha, c_i >= 1, rarest fact >= min_tail_count."""
    log_w = -alpha * np.log(np.arange(1, num_facts + 1, dtype=np.float64))
    log_p = log_w - _logsumexp(log_w)
    batch = max(int(np.ceil(min_tail_count / np.exp(log_p[-1]))), num_facts)

    excess = np.maximum(batch * np.exp(log_p) - 1.0, 0.0)
    counts = 1 + np.floor(excess).astype(int)
    remainder = excess - np.floor(excess)
    deficit = batch - int(counts.sum())
    if deficit > 0:
        counts[np.argsort(-remainder)[:deficit]] += 1
    elif deficit < 0:
        candidates = np.where(counts > 1, remainder, np.inf)
        counts[np.argsort(candidates)[:-deficit]] -= 1
    return counts.tolist()


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))
