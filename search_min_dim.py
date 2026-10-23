"""Min-d search: the smallest width d at which a transformer memorizes a dataset."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import optuna
import torch
from optuna.samplers import TPESampler
from optuna.trial import TrialState

import train
from train import TrainConfig

SEEDS = (10, 20, 30)
FAMILIES = ("uniform", "symmetric", "two_hop", "powerlaw")
DECAY_TYPES = ("linear", "cosine")

MAX_BATCHES = 40_000
STABILITY_WINDOW = 50
WARMUP_STEPS_PER_LAYER = 500
START_DIM = 32
DIM_CAP = 256

LR_RANGE = (1e-6, 1.0)
WEIGHT_DECAY_RANGE = (1e-8, 1e-1)
MIN_LR_RATIO_RANGE = (1e-8, 1.0)
DECAY_END_STEP_RANGE = (100, MAX_BATCHES)

N_TRIALS = 60
PATIENCE = 15
FAIL_ALLOWANCE = 10
STARTUP_TRIALS_PER_PARAM = 5
N_STARTUP_TRIALS = STARTUP_TRIALS_PER_PARAM * 5
SAMPLER_SEED = 42


@dataclass(frozen=True)
class Setup:
    family: str
    num_entities: int
    num_relations: int
    layers: int
    alpha: float
    min_tail_count: int

    @property
    def is_powerlaw(self) -> bool:
        return self.family == "powerlaw"


@dataclass(frozen=True)
class Hparams:
    lr: float
    weight_decay: float
    lr_decay_type: str
    min_lr: float
    decay_end_ratio: float


def study_name(setup: Setup, decay_type: str) -> str:
    name = (f"min_dim_{setup.family}_E{setup.num_entities}_R{setup.num_relations}"
            f"_L{setup.layers}_{decay_type}")
    if setup.is_powerlaw:
        name += f"_alpha{setup.alpha:.2f}_mintail{setup.min_tail_count}"
    return name


def train_config(setup: Setup, hp: Hparams, seed: int, dim: int) -> TrainConfig:
    return TrainConfig(
        family=setup.family, num_entities=setup.num_entities,
        num_relations=setup.num_relations, dim=dim, layers=setup.layers,
        seed=seed, lr=hp.lr, weight_decay=hp.weight_decay,
        warmup_steps=WARMUP_STEPS_PER_LAYER * setup.layers, lr_decay_type=hp.lr_decay_type,
        min_lr=hp.min_lr, decay_end_ratio=hp.decay_end_ratio, decay_start_ratio=0.0,
        max_batches=MAX_BATCHES, stability_window=STABILITY_WINDOW,
        alpha=setup.alpha, min_tail_count=setup.min_tail_count, heads=1)


def cache_key(setup: Setup, hp: Hparams, seed: int, dim: int) -> dict:
    return {
        "ent": setup.num_entities, "rel": setup.num_relations, "seed": seed,
        "batch": 0 if setup.is_powerlaw else setup.num_entities * setup.num_relations,
        "lr": hp.lr, "weight_decay": hp.weight_decay,
        "warmup_steps": WARMUP_STEPS_PER_LAYER * setup.layers,
        "lr_decay_type": hp.lr_decay_type, "dim": dim, "min_lr": hp.min_lr,
        "decay_end_ratio": hp.decay_end_ratio, "decay_start_ratio": 0.0,
        "stability_window": STABILITY_WINDOW, "max_batches": MAX_BATCHES,
        "family": setup.family,
        "alpha": setup.alpha if setup.is_powerlaw else None,
        "min_tail_count": setup.min_tail_count if setup.is_powerlaw else None,
        "n_layers": setup.layers, "n_heads": 1,
    }


class ProbeCache:
    """One JSON file per finished training run, written atomically."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: dict) -> str:
        blob = json.dumps(key, sort_keys=True, separators=(",", ":"))
        return os.path.join(self.directory, hashlib.sha256(blob.encode()).hexdigest()[:32] + ".json")

    def load(self, key: dict) -> dict | None:
        try:
            with open(self._path(key)) as f:
                record = json.load(f)
        except (OSError, ValueError):
            return None
        return record.get("result") if record.get("v") == 1 and record.get("key") == key else None

    def store(self, key: dict, result: dict) -> None:
        tmp = os.path.join(self.directory, f".tmp_{uuid.uuid4().hex}")
        try:
            with open(tmp, "w") as f:
                json.dump({"v": 1, "key": key, "result": result}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path(key))
        except OSError as e:
            print(f"probe cache write failed: {e}", flush=True)


def _is_oom(e: RuntimeError) -> bool:
    return "out of memory" in str(e).lower()


TrainFn = Callable[[TrainConfig], train.TrainResult]


def _train_oom_fails_trial(train_fn: TrainFn, cfg: TrainConfig) -> train.TrainResult:
    """Train, retrying once after an out-of-memory error; a second one propagates and fails the trial."""
    try:
        return train_fn(cfg)
    except RuntimeError as e:
        if not _is_oom(e):
            raise
        torch.cuda.empty_cache()
        time.sleep(2)
        return train_fn(cfg)


def probe(setup: Setup, cache: ProbeCache, hp: Hparams, seed: int,
          dim: int, train_fn: TrainFn) -> tuple[bool, int | None, float | None]:
    """Train once at width `dim`, or read the cached run. Returns (converged, steps_to_100, best_acc)."""
    key = cache_key(setup, hp, seed, dim)
    hit = cache.load(key)
    if hit is not None:
        return hit["converged"], hit["batches100"], hit["best_acc"]
    result = _train_oom_fails_trial(train_fn, train_config(setup, hp, seed, dim))
    cache.store(key, {"converged": result.converged, "batches100": result.steps_to_100,
                      "best_acc": result.best_acc})
    return result.converged, result.steps_to_100, result.best_acc


def min_dim_for_seed(setup: Setup, cache: ProbeCache, hp: Hparams, seed: int,
                     train_fn: TrainFn) -> tuple[int | None, int | None, dict[int, float]]:
    """(min dim or None, steps_to_100 there, best accuracy at every probed dim)."""
    acc_by_dim: dict[int, float] = {}

    def passes(dim: int) -> tuple[bool, int | None]:
        converged, steps, best_acc = probe(setup, cache, hp, seed, dim, train_fn)
        if best_acc is not None:
            acc_by_dim[dim] = best_acc
        return converged, steps

    ok, ok_steps = None, None
    dim = START_DIM
    while dim > 0:
        converged, steps = passes(dim)
        if not converged:
            break
        ok, ok_steps = dim, steps
        dim //= 2
    if dim == 0:
        return ok, ok_steps, acc_by_dim

    if ok is None:
        fail, dim = START_DIM, 2 * START_DIM
        while dim <= DIM_CAP:
            converged, steps = passes(dim)
            if converged:
                ok, ok_steps = dim, steps
                break
            fail, dim = dim, 2 * dim
        if ok is None:
            return None, None, acc_by_dim
        lo = fail + 1
    else:
        lo = dim
    hi = ok
    while lo <= hi:
        mid = (lo + hi) // 2
        converged, steps = passes(mid)
        if converged:
            ok, ok_steps, hi = mid, steps, mid - 1
        else:
            lo = mid + 1
    return ok, ok_steps, acc_by_dim


def min_dim_all_seeds(setup: Setup, cache: ProbeCache, hp: Hparams,
                      train_fn: TrainFn) -> list[tuple]:
    """Run the width search for every seed in parallel, one thread per seed, spread over the GPUs."""
    n_gpus = torch.cuda.device_count()

    def run(index_seed: tuple[int, int]) -> tuple:
        index, seed = index_seed
        if n_gpus > 0:
            torch.cuda.set_device(index % n_gpus)
        return min_dim_for_seed(setup, cache, hp, seed, train_fn)

    with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        return list(pool.map(run, enumerate(SEEDS)))


def objective_value(per_seed_dim: dict[int, int | None]) -> float | None:
    """min_i d_i + 0.1 * (failed seeds) + 0.05 * sum_i (d_i - min_j d_j); None if all fail."""
    dims = [d for d in per_seed_dim.values() if d is not None]
    if not dims:
        return None
    best = min(dims)
    n_failed = len(per_seed_dim) - len(dims)
    spread = sum(d - best for d in dims)
    return best + 0.1 * n_failed + 0.05 * spread


def sample_hparams(trial: optuna.Trial, decay_type: str) -> Hparams:
    lr = trial.suggest_float("lr", *LR_RANGE, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", *MIN_LR_RATIO_RANGE, log=True)
    decay_end_step = trial.suggest_int("decay_end_step", *DECAY_END_STEP_RANGE)
    if trial.suggest_categorical("use_weight_decay", [False, True]):
        weight_decay = trial.suggest_float("weight_decay", *WEIGHT_DECAY_RANGE, log=True)
    else:
        weight_decay = 0.0
    return Hparams(lr, weight_decay, decay_type, lr * min_lr_ratio,
                   min(1.0, decay_end_step / MAX_BATCHES))


def first_trial_params(setup: Setup) -> dict:
    return {"lr": 1.5 / math.sqrt(setup.num_entities * setup.num_relations),
            "use_weight_decay": True, "weight_decay": 4e-5,
            "min_lr_ratio": 0.01, "decay_end_step": 20000}


def _trials(study: optuna.Study, *states: TrialState) -> list[optuna.trial.FrozenTrial]:
    return study.get_trials(deepcopy=False, states=states)


def trials_since_best(trials: list[optuna.trial.FrozenTrial]) -> int:
    values = [t.value for t in trials]
    return len(values) - values.index(min(values)) - 1


def is_finished(study: optuna.Study) -> bool:
    done = sorted(_trials(study, TrialState.COMPLETE), key=lambda t: t.number)
    return len(done) >= N_TRIALS and trials_since_best(done) >= PATIENCE


class EarlyStopping:
    """Stop after min_trials COMPLETE trials once the best value is patience trials old."""

    def __init__(self, done: list[optuna.trial.FrozenTrial], min_trials: int, patience: int):
        self.min_trials = min_trials
        self.patience = patience
        self.best = min(t.value for t in done) if done else None
        self.since_best = trials_since_best(done) if done else 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        if self.best is None or study.best_value < self.best:
            self.best = study.best_value
            self.since_best = 0
        else:
            self.since_best += 1
        if len(_trials(study, TrialState.COMPLETE)) >= self.min_trials and self.since_best >= self.patience:
            study.stop()


def requeue_orphans(study: optuna.Study, marker_dir: str, max_trials: int,
                    stale_hours: float) -> None:
    """Re-enqueue the params of RUNNING trials older than stale_hours, once across workers."""
    os.makedirs(marker_dir, exist_ok=True)
    now = datetime.datetime.now()
    budget_left = max_trials - len(_trials(study, TrialState.COMPLETE))
    for t in _trials(study, TrialState.RUNNING):
        if budget_left <= 0:
            break
        if t.datetime_start is None or not t.params:
            continue
        if (now - t.datetime_start).total_seconds() < stale_hours * 3600:
            continue
        signature = hashlib.sha256(json.dumps(
            [t.number, str(t.datetime_start), sorted(t.params.items())],
            default=str, sort_keys=True).encode()).hexdigest()[:16]
        try:
            os.close(os.open(os.path.join(marker_dir, f"t{t.number}_{signature}.enqueued"),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        study.enqueue_trial(dict(t.params))
        budget_left -= 1


def make_objective(setup: Setup, cache: ProbeCache, decay_type: str, train_fn: TrainFn):
    def objective(trial: optuna.Trial) -> float:
        if is_finished(trial.study):
            trial.study.stop()
            raise optuna.TrialPruned()
        hp = sample_hparams(trial, decay_type)
        results = min_dim_all_seeds(setup, cache, hp, train_fn)
        per_seed_dim = {s: r[0] for s, r in zip(SEEDS, results)}
        trial.set_user_attr("per_seed_min_dim", per_seed_dim)
        trial.set_user_attr("per_seed_batches100", {s: r[1] for s, r in zip(SEEDS, results)})
        trial.set_user_attr("per_seed_acc_by_dim", {s: r[2] for s, r in zip(SEEDS, results)})
        value = objective_value(per_seed_dim)
        if value is None:
            raise optuna.TrialPruned()
        return value
    return objective


def run_study(setup: Setup, decay_type: str, db_dir: str, cache_dir: str,
              orphan_stale_hours: float, train_fn: TrainFn = train.train) -> optuna.Study:
    """Run the remaining trial count (PRUNED and FAIL trials use it up); rerunning tops the study up."""
    name = study_name(setup, decay_type)
    os.makedirs(db_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(db_dir, name + '.db')}"
    cache = ProbeCache(os.path.join(cache_dir, name))
    # hash() of a str is salted per process unless PYTHONHASHSEED is set.
    sampler = TPESampler(seed=SAMPLER_SEED + hash(decay_type) % 1000,
                         n_startup_trials=N_STARTUP_TRIALS, multivariate=True, group=True)
    objective = make_objective(setup, cache, decay_type, train_fn)

    try:
        study = optuna.load_study(study_name=name, storage=storage, sampler=sampler)
    except KeyError:
        study = optuna.create_study(study_name=name, storage=storage, direction="minimize",
                                    sampler=sampler, load_if_exists=True)
        study.enqueue_trial(first_trial_params(setup))
        study.optimize(objective, n_trials=N_TRIALS,
                       callbacks=[EarlyStopping([], N_TRIALS, PATIENCE)], catch=(Exception,))
        return study

    requeue_orphans(study, os.path.join(cache.directory, "orphans"), N_TRIALS, orphan_stale_hours)
    n_complete = len(_trials(study, TrialState.COMPLETE))
    n_fail = len(_trials(study, TrialState.FAIL))
    n_left = max(0, min(N_TRIALS - n_complete, N_TRIALS + FAIL_ALLOWANCE - (n_complete + n_fail)))
    if n_left > 0 and not is_finished(study):
        early_stop = EarlyStopping(_trials(study, TrialState.COMPLETE), N_TRIALS, PATIENCE)
        study.optimize(objective, n_trials=n_left, callbacks=[early_stop], catch=(Exception,))
    return study


def report(study: optuna.Study) -> None:
    if not _trials(study, TrialState.COMPLETE):
        print(f"{study.study_name}: no completed trials", flush=True)
        return
    print(f"{study.study_name}: best objective = {study.best_value:.2f} "
          f"params = {study.best_params}", flush=True)


def _parse_dataset(text: str) -> tuple[int, int]:
    entities, relations = text.split(",")
    return int(entities), int(relations)


def main(argv: list[str] | None = None, train_fn: TrainFn = train.train) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, default="uniform")
    parser.add_argument("--datasets", nargs="+", type=_parse_dataset, required=True,
                        metavar="E,R", help="datasets as entities,relations")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr-decay-types", nargs="+", choices=DECAY_TYPES, default=list(DECAY_TYPES))
    parser.add_argument("--alpha", type=float, default=1.0, help="powerlaw exponent")
    parser.add_argument("--min-tail-count", type=int, default=1,
                        help="powerlaw: copies of the rarest fact per batch")
    parser.add_argument("--db-dir", default="studies")
    parser.add_argument("--cache-dir", default="probe_cache")
    parser.add_argument("--orphan-stale-hours", type=float, default=18.0,
                        help="RUNNING trials older than this are re-run")
    args = parser.parse_args(argv)

    for entities, relations in args.datasets:
        setup = Setup(args.family, entities, relations, args.layers, args.alpha,
                      args.min_tail_count)
        for decay_type in args.lr_decay_types:
            report(run_study(setup, decay_type, args.db_dir, args.cache_dir,
                             args.orphan_stale_hours, train_fn))


if __name__ == "__main__":
    main()
